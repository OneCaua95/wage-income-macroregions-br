#!/usr/bin/env python3
"""Testa o modelo M5 do notebook ``analise`` na base de 25 a 65 anos.

O M5 é uma regressão linear de ``log(Salario_Hora)`` com escolaridade,
formalidade, setor público, sexo, dummies de setor, ocupação, cor, região e
efeitos fixos de ano. O ajuste usa a mesma divisão temporal do benchmark:
treino 2013--2022, validação 2023--2024 e teste 2025.

Além das métricas preditivas, o script calcula R² ajustado, estatística F,
condição numérica e erros-padrão agrupados em duas vias (coorte de nascimento
e UF) no conjunto de treino. Como o M5 original foi escrito para uma base
com faixas de nascimento mais curtas, as faixas são estendidas até 2003 para
cobrir corretamente indivíduos de 25 anos em 2025.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config.settings import setup_logging


TARGET_COLUMNS = [
    "Ano", "Ano_Nascimento", "Idade", "UF", "Sexo", "Cor", "Anos_de_Estudo",
    "Categoria_emprego", "Contribuicao_previdencia",
    "Grupo_atv_princ_empreedimento", "Grupo_ocupacional_no_emprego",
    "Condicao_Ocupacao", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas",
    "Regiao",
]

SETORES = {
    2: "Setor_Industria_Geral",
    3: "Setor_Construcao",
    4: "Setor_Comercio",
    5: "Setor_Transporte_Armazenagem_Correio",
    6: "Setor_Alojamento_Alimentacao",
    7: "Setor_Info_Comunicacao_Financ_Imob_Prof_Adm",
    8: "Setor_Adm_Publica",
    9: "Setor_Educacao_Saude",
    10: "Setor_Outros_Servicos",
}
OCUPACOES = {
    1: "Ocup_Diretores_Gerentes",
    2: "Ocup_Profissionais_Ciencias_Intelectuais",
    3: "Ocup_Tecnicos_Nivel_Medio",
    4: "Ocup_Apoio_Administrativo",
    5: "Ocup_Servicos_Vendedores_Comercio_Mercado",
    6: "Ocup_Qualificados_Agropecuaria_Pesca",
    7: "Ocup_Qualificados_Construcao_Mecanica_Oficios",
    8: "Ocup_Operadores_Instalacoes_Maquinas",
    9: "Ocup_Forcas_Armadas_Policiais_Bombeiros",
}
CORES = {2: "Cor_Preta", 3: "Cor_Amarela", 4: "Cor_Parda", 5: "Cor_Indigena"}
REGIOES = ["Norte", "Sul", "Centro-Oeste", "Sudeste"]

# Faixas de nascimento de cinco anos, estendidas para abranger a janela 25--65.
BIRTH_BINS = [1947, 1952, 1957, 1962, 1967, 1972, 1977, 1982, 1988, 1993, 1998, 2003]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testa o modelo M5 do notebook analise.")
    parser.add_argument(
        "--entrada",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet",
        help="Parquet consolidado, que contém ano de nascimento e grupo ocupacional.",
    )
    parser.add_argument(
        "--saida-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "modelos_salario_hora",
    )
    parser.add_argument("--max-treino", type=int, default=150_000)
    parser.add_argument("--max-avaliacao", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def amostrar(df: pd.DataFrame, limite: int, seed: int) -> pd.DataFrame:
    if len(df) <= limite:
        return df.copy()
    return df.sample(n=limite, random_state=seed).copy()


def carregar_base(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Base consolidada não encontrada: {caminho}")
    df = pd.read_parquet(caminho, columns=TARGET_COLUMNS)
    df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")
    df = df[df["Idade"].between(25, 65)].copy()
    df = df[
        (df["Condicao_Ocupacao"] == 1)
        & (df["Rendimento_hab_Trab_princ"] > 0)
        & (df["Horas_hab_trabalhadas"] > 0)
        & df["Anos_de_Estudo"].between(0, 16)
        & df["Regiao"].notna()
        & df["UF"].notna()
    ].copy()
    df["Salario_Hora"] = (
        df["Rendimento_hab_Trab_princ"].astype(float)
        / (df["Horas_hab_trabalhadas"].astype(float) * 4.33)
    )
    df = df[df["Salario_Hora"] > 0].copy()

    # Variáveis binárias usadas na formulação do notebook.
    df["is_public"] = df["Categoria_emprego"].isin([5, 6, 7]).astype(float)
    df["is_formal"] = (
        df["Categoria_emprego"].isin([1, 3, 5, 7])
        | (df["Categoria_emprego"].isin([8, 9]) & (df["Contribuicao_previdencia"] == 1))
    ).astype(float)
    df["Sexo_Dummy"] = (df["Sexo"] == 2).astype(float)

    feature_names = ["Anos_de_Estudo", "is_formal", "is_public", "Sexo_Dummy"]
    for code, name in SETORES.items():
        df[name] = (df["Grupo_atv_princ_empreedimento"] == code).astype(float)
        feature_names.append(name)
    for code, name in OCUPACOES.items():
        df[name] = (df["Grupo_ocupacional_no_emprego"] == code).astype(float)
        feature_names.append(name)
    for code, name in CORES.items():
        df[name] = (df["Cor"] == code).astype(float)
        feature_names.append(name)
    for region in REGIOES:
        name = f"Reg_{region}"
        df[name] = (df["Regiao"].astype(str) == region).astype(float)
        feature_names.append(name)

    # Tratamento de efeitos fixos de ano com 2013 como categoria de referência.
    years = list(range(2013, 2026))
    for year in years[1:]:
        name = f"Ano_{year}"
        df[name] = (df["Ano"] == year).astype(float)
        feature_names.append(name)

    birth_year = df["Ano_Nascimento"].where(
        df["Ano_Nascimento"].between(1947, 2003), df["Ano"] - df["Idade"]
    )
    birth_band = pd.cut(
        birth_year, bins=BIRTH_BINS, labels=False, include_lowest=True,
    )
    df["Coorte_Cluster"] = (
        df["Sexo_Dummy"].astype(int).astype(str) + "-"
        + birth_band.astype("string") + "-"
        + df["Cor"].astype("string")
    )
    df["UF_Cluster"] = df["UF"].astype(str)
    df["y"] = np.log(df["Salario_Hora"].astype(float))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", *feature_names, "Coorte_Cluster"])
    columns = ["Ano", "y", "Salario_Hora", "Coorte_Cluster", "UF_Cluster", *feature_names]
    return df[columns]


def metricas(y_true_log: pd.Series, pred_log: np.ndarray) -> dict[str, float]:
    y_true_log = np.asarray(y_true_log, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    y_true = np.exp(y_true_log)
    pred = np.exp(np.clip(pred_log, -20, 20))
    return {
        "mae_log": float(mean_absolute_error(y_true_log, pred_log)),
        "rmse_log": float(mean_squared_error(y_true_log, pred_log) ** 0.5),
        "r2_log": float(r2_score(y_true_log, pred_log)),
        "mae_salario_hora": float(mean_absolute_error(y_true, pred)),
        "rmse_salario_hora": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2_salario_hora": float(r2_score(y_true, pred)),
    }


def cluster_covariance(X: np.ndarray, residuals: np.ndarray, clusters: pd.Series) -> tuple[np.ndarray, int]:
    """Calcula a matriz sanduíche de uma via e retorna a matriz e G."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    labels = clusters.astype(str).to_numpy()
    sums = []
    for label in np.unique(labels):
        xgu = X[labels == label].T @ residuals[labels == label]
        sums.append(xgu)
    meat = np.vstack(sums).T @ np.vstack(sums)
    g = len(sums)
    correction = (g / max(g - 1, 1)) * ((len(X) - 1) / max(len(X) - X.shape[1], 1))
    variance = correction * (xtx_inv @ meat @ xtx_inv)
    return variance, g


def diagnosticos_m5(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    treino: pd.DataFrame,
    feature_names: list[str],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    pred = X @ beta
    residuals = y - pred
    n, k = X.shape
    sse = float(residuals @ residuals)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst
    r2_adj = 1 - (sse / max(n - k, 1)) / (sst / max(n - 1, 1))
    f_stat = ((sst - sse) / max(k - 1, 1)) / (sse / max(n - k, 1))
    v_coorte, g_coorte = cluster_covariance(X, residuals, treino["Coorte_Cluster"])
    v_uf, g_uf = cluster_covariance(X, residuals, treino["UF_Cluster"])
    intersection = treino["Coorte_Cluster"].astype(str) + "|" + treino["UF_Cluster"].astype(str)
    v_intersection, g_intersection = cluster_covariance(X, residuals, intersection)
    v_two_way = v_coorte + v_uf - v_intersection
    eigenvalues, eigenvectors = np.linalg.eigh(v_two_way)
    v_two_way = eigenvectors @ np.diag(np.clip(eigenvalues, 0, None)) @ eigenvectors.T
    se_two_way = np.sqrt(np.clip(np.diag(v_two_way), 0, None))
    nomes = ["Intercept"] + feature_names
    graus_liberdade = max(min(g_coorte, g_uf) - 1, 1)
    t_stats = beta / np.where(se_two_way > 0, se_two_way, np.nan)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), graus_liberdade))
    coeficientes = pd.DataFrame(
        {
            "variavel": nomes,
            "coeficiente": beta,
            "erro_padrao_cluster_duplo": se_two_way,
            "t": t_stats,
            "p_valor": p_values,
            "efeito_percentual_aprox": np.where(
                np.arange(len(beta)) == 0, np.nan, np.expm1(beta) * 100
            ),
        }
    )
    sorted_residuals = residuals[np.argsort(treino["Ano"].to_numpy(), kind="stable")]
    dw = float(np.sum(np.diff(sorted_residuals) ** 2) / max(np.sum(sorted_residuals ** 2), np.finfo(float).eps))
    jb = stats.jarque_bera(residuals)
    singular_values = np.linalg.svd(X, compute_uv=False)
    positive_singular_values = singular_values[singular_values > 1e-12]
    condition = float(
        positive_singular_values.max() / positive_singular_values.min()
        if len(positive_singular_values) else np.inf
    )
    diagnosticos = {
        "n_treino_m5": int(n),
        "k_regressores_m5": int(k),
        "r2_treino_m5": float(r2),
        "r2_ajustado_treino_m5": float(r2_adj),
        "f_stat_treino_m5": float(f_stat),
        "jarque_bera_m5": float(jb.statistic),
        "jarque_bera_p_m5": float(jb.pvalue),
        "durbin_watson_ano_m5": dw,
        "condicao_numerica_m5": condition,
        "clusters_coorte_m5": int(g_coorte),
        "clusters_uf_m5": int(g_uf),
        "clusters_intersecao_m5": int(g_intersection),
        "se_media_duas_vias_m5": float(np.nanmean(se_two_way)),
    }
    return diagnosticos, coeficientes


def ajustar_e_avaliar(treino: pd.DataFrame, avaliacao: dict[str, pd.DataFrame], seed: int) -> tuple[list[dict], dict]:
    feature_names = [c for c in treino.columns if c not in {"Ano", "y", "Salario_Hora", "Coorte_Cluster", "UF_Cluster"}]
    x_treino = np.column_stack([np.ones(len(treino)), treino[feature_names].to_numpy(dtype=float)])
    y_treino = treino["y"].to_numpy(dtype=float)
    inicio = time.perf_counter()
    beta, *_ = np.linalg.lstsq(x_treino, y_treino, rcond=None)
    tempo = time.perf_counter() - inicio
    resultados = []
    for conjunto, dados in avaliacao.items():
        x = np.column_stack([np.ones(len(dados)), dados[feature_names].to_numpy(dtype=float)])
        resultado = metricas(dados["y"], x @ beta)
        resultado.update({"modelo": "M5_ajustado", "tipo": conjunto, "tempo_treino_seg": tempo})
        resultados.append(resultado)
    diagnosticos, coeficientes = diagnosticos_m5(
        x_treino, y_treino, beta, treino, feature_names
    )
    diagnosticos["seed_m5"] = int(seed)
    diagnosticos["faixas_coorte_m5"] = (
        "1947-2003 em intervalos de cinco anos; ano de nascimento inválido "
        "reconstruído por Ano - Idade"
    )
    return resultados, diagnosticos, coeficientes


def atualizar_arquivos(
    args: argparse.Namespace,
    m5_resultados: list[dict],
    diagnosticos: dict,
    coeficientes: pd.DataFrame,
) -> None:
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    metricas_path = args.saida_dir / "metricas_modelos.csv"
    metricas_df = pd.read_csv(metricas_path) if metricas_path.exists() else pd.DataFrame()
    metricas_df = metricas_df[metricas_df["modelo"] != "M5_ajustado"] if not metricas_df.empty else metricas_df
    metricas_df = pd.concat([metricas_df, pd.DataFrame(m5_resultados)], ignore_index=True)
    metricas_df.to_csv(metricas_path, index=False)
    coeficientes.to_csv(args.saida_dir / "coeficientes_m5.csv", index=False)

    teste = metricas_df[metricas_df["tipo"] == "teste"].sort_values("rmse_log")
    melhor = teste.iloc[0].to_dict()
    resumo_path = args.saida_dir / "resumo_modelos.json"
    resumo = json.loads(resumo_path.read_text(encoding="utf-8")) if resumo_path.exists() else {}
    resumo.update({
        "melhor_modelo_por_rmse_log_teste": melhor["modelo"],
        "metricas_melhor_modelo": melhor,
        "m5_diagnosticos": diagnosticos,
        "m5_especificacao": "log(Salario_Hora) ~ escolaridade + formalidade + setor público + sexo + dummies de setor, ocupação, cor e região + FE de ano; SE agrupado em coorte e UF no treino",
    })
    resumo_path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    tabela = teste[["modelo", "rmse_log", "mae_log", "r2_log", "mae_salario_hora", "rmse_salario_hora"]].copy()
    tabela.columns = ["Modelo", "RMSE log", "MAE log", "R² log", "MAE salário-hora", "RMSE salário-hora"]
    linhas = [
        "# Comparação de modelos para o salário-hora", "",
        "A comparação usa `log(Salario_Hora)`, com treino em 2013–2022, validação em 2023–2024 e teste em 2025.", "",
        f"O melhor modelo no teste foi **{melhor['modelo']}**, pelo menor RMSE no log do salário-hora.", "",
        tabela.to_markdown(index=False), "",
        "## M5 do notebook", "",
        "O M5 foi reproduzido com escolaridade, formalidade, setor público, sexo, dummies de setor, ocupação, cor, região e efeitos fixos de ano. A avaliação preditiva usa a mesma divisão temporal dos demais modelos. Os erros-padrão agrupados por coorte de nascimento e UF são calculados no treino.", "",
        "## Limitações", "",
        "- A base tratada não contém pesos amostrais da PNAD; os resultados são não ponderados.",
        "- O M5 usa amostras fixas de 150 mil observações no treino e 50 mil em validação e teste.",
        "- As faixas de nascimento do M5 foram estendidas até 2003 para cobrir a faixa etária de 25 a 65 anos em 2025.",
    ]
    (args.saida_dir / "relatorio_modelos.md").write_text("\n".join(linhas), encoding="utf-8")


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    print(f"Carregando base do M5: {args.entrada}")
    df = carregar_base(args.entrada)
    treino = amostrar(df[df["Ano"].between(2013, 2022)], args.max_treino, args.seed)
    validacao = amostrar(df[df["Ano"].between(2023, 2024)], args.max_avaliacao, args.seed + 1)
    teste = amostrar(df[df["Ano"] == 2025], args.max_avaliacao, args.seed + 2)
    print(f"Amostras M5: treino={len(treino):,}, validação={len(validacao):,}, teste={len(teste):,}")
    resultados, diagnosticos, coeficientes = ajustar_e_avaliar(
        treino, {"validacao": validacao, "teste": teste}, args.seed
    )
    atualizar_arquivos(args, resultados, diagnosticos, coeficientes)
    print("M5 salvo em metricas_modelos.csv, coeficientes_m5.csv, resumo_modelos.json e relatorio_modelos.md")
    print(pd.DataFrame(resultados).to_string(index=False))
    print(json.dumps(diagnosticos, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
