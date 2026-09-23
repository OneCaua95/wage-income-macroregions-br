#!/usr/bin/env python3
"""Estima o M5 agrupado com efeitos de macroregião e estado.

A especificação usa uma única regressão Ridge para toda a amostra. Os efeitos
de estado são codificados como desvios dentro da macroregião, com restrição de
soma zero. Assim, o coeficiente da macroregião representa o efeito médio dos
estados daquele grupo, enquanto os desvios mostram a posição de cada estado
dentro da própria macroregião.

O modelo também permite que escolaridade, formalidade, sexo e setor público
tenham relações diferentes por macroregião. A escolha de alpha é feita na
validação temporal e o teste de 2025 é usado apenas uma vez para o alpha
selecionado.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SETORES = {
    2: "Setor_Industria_Geral", 3: "Setor_Construcao", 4: "Setor_Comercio",
    5: "Setor_Transporte_Armazenagem_Correio", 6: "Setor_Alojamento_Alimentacao",
    7: "Setor_Info_Comunicacao_Financ_Imob_Prof_Adm", 8: "Setor_Adm_Publica",
    9: "Setor_Educacao_Saude", 10: "Setor_Outros_Servicos",
}
OCUPACOES = {
    1: "Ocup_Diretores_Gerentes", 2: "Ocup_Profissionais_Ciencias_Intelectuais",
    3: "Ocup_Tecnicos_Nivel_Medio", 4: "Ocup_Apoio_Administrativo",
    5: "Ocup_Servicos_Vendedores_Comercio_Mercado", 6: "Ocup_Qualificados_Agropecuaria_Pesca",
    7: "Ocup_Qualificados_Construcao_Mecanica_Oficios", 8: "Ocup_Operadores_Instalacoes_Maquinas",
    9: "Ocup_Forcas_Armadas_Policiais_Bombeiros",
}
CORES = {2: "Cor_Preta", 3: "Cor_Amarela", 4: "Cor_Parda", 5: "Cor_Indigena"}
MACROREGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
MACRO_REFERENCIA = "Nordeste"
BIRTH_BINS = [1947, 1952, 1957, 1962, 1967, 1972, 1977, 1982, 1988, 1993, 1998, 2003]
RAW_COLUMNS = [
    "Ano", "Trimestre", "Ano_Nascimento", "Idade", "UF", "Sexo", "Cor",
    "Anos_de_Estudo", "Categoria_emprego", "Contribuicao_previdencia",
    "Grupo_atv_princ_empreedimento", "Grupo_ocupacional_no_emprego",
    "Condicao_Ocupacao", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas",
    "Regiao",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=PROJECT_ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "modelos_salario_hora")
    parser.add_argument("--max-treino", type=int, default=150_000)
    parser.add_argument("--max-avaliacao", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[100.0, 300.0, 1000.0, 3000.0])
    return parser.parse_args()


def amostrar(df: pd.DataFrame, limite: int, seed: int) -> pd.DataFrame:
    return df.sample(n=limite, random_state=seed).copy() if len(df) > limite else df.copy()


def carregar(caminho: Path, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(caminho, columns=RAW_COLUMNS)
    df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")
    df["Anos_de_Estudo"] = pd.to_numeric(df["Anos_de_Estudo"], errors="coerce")
    mascara = (
        df["Idade"].between(25, 65)
        & df["Condicao_Ocupacao"].eq(1)
        & df["Rendimento_hab_Trab_princ"].gt(0)
        & df["Horas_hab_trabalhadas"].gt(0)
        & df["Anos_de_Estudo"].between(0, 16)
        & df["Regiao"].isin(MACROREGIOES)
        & df["UF"].notna()
        & df["Cor"].notna()
    )
    nascimento = df["Ano_Nascimento"].where(df["Ano_Nascimento"].between(1947, 2003), df["Ano"] - df["Idade"])
    mascara &= pd.cut(nascimento, bins=BIRTH_BINS, labels=False, include_lowest=True).notna()
    df = df.loc[mascara].copy()
    df["Salario_Hora"] = df["Rendimento_hab_Trab_princ"] / (df["Horas_hab_trabalhadas"] * 4.33)
    df["y"] = np.log(df["Salario_Hora"])
    return {
        "treino": amostrar(df[df["Ano"].between(2013, 2022)], args.max_treino, args.seed),
        "validacao": amostrar(df[df["Ano"].between(2023, 2024)], args.max_avaliacao, args.seed + 1),
        "teste": amostrar(df[df["Ano"].eq(2025)], args.max_avaliacao, args.seed + 2),
    }


def criar_matriz(df: pd.DataFrame, estados_por_macro: dict[str, list], incluir_ano: bool = True) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["Anos_de_Estudo"] = df["Anos_de_Estudo"].astype(float)
    x["is_formal"] = (df["Categoria_emprego"].isin([1, 3, 5, 7]) | (df["Categoria_emprego"].isin([8, 9]) & df["Contribuicao_previdencia"].eq(1))).astype(float)
    x["is_public"] = df["Categoria_emprego"].isin([5, 6, 7]).astype(float)
    x["Sexo_Dummy"] = df["Sexo"].eq(2).astype(float)
    for code, name in SETORES.items():
        x[name] = df["Grupo_atv_princ_empreedimento"].eq(code).astype(float)
    for code, name in OCUPACOES.items():
        x[name] = df["Grupo_ocupacional_no_emprego"].eq(code).astype(float)
    for code, name in CORES.items():
        x[name] = df["Cor"].eq(code).astype(float)
    if incluir_ano:
        for year in range(2014, 2023):
            x[f"Ano_{year}"] = df["Ano"].eq(year).astype(float)
    x["Idade_centrada"] = df["Idade"].astype(float) - 45.0
    x["Idade_centrada_2"] = x["Idade_centrada"] ** 2

    # Efeitos de macroregião: Nordeste é a referência.
    for macro in MACROREGIOES:
        if macro != MACRO_REFERENCIA:
            x[f"Macro_{macro}"] = df["Regiao"].eq(macro).astype(float)

    # Desvios de estado codificados contra o último estado de cada macroregião.
    # Essa parametrização mantém a média dos efeitos estaduais igual a zero.
    for macro, estados in estados_por_macro.items():
        referencia_estado = estados[-1]
        for estado in estados[:-1]:
            coluna = f"EstadoDev_{estado}"
            x[coluna] = np.select(
                [df["UF"].eq(estado), df["UF"].eq(referencia_estado)], [1.0, -1.0], default=0.0
            )

    # Interações selecionadas para permitir heterogeneidade regional.
    for macro in MACROREGIOES:
        if macro == MACRO_REFERENCIA:
            continue
        indicador = df["Regiao"].eq(macro).astype(float)
        x[f"Estudo_x_Macro_{macro}"] = x["Anos_de_Estudo"] * indicador
        x[f"Formal_x_Macro_{macro}"] = x["is_formal"] * indicador
        x[f"Sexo_x_Macro_{macro}"] = x["Sexo_Dummy"] * indicador
        x[f"Publico_x_Macro_{macro}"] = x["is_public"] * indicador
    return x


def metricas(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    y_real, p_real = np.exp(y), np.exp(np.clip(p, -20, 20))
    return {
        "mae_log": float(np.mean(np.abs(y - p))),
        "rmse_log": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2_log": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "mae_salario_hora": float(np.mean(np.abs(y_real - p_real))),
        "rmse_salario_hora": float(np.sqrt(np.mean((y_real - p_real) ** 2))),
        "r2_salario_hora": float(1 - np.sum((y_real - p_real) ** 2) / np.sum((y_real - y_real.mean()) ** 2)),
    }


def nomes_legiveis(nome: str, estados_por_macro: dict[str, list]) -> tuple[str, str]:
    if nome.startswith("Macro_"):
        return nome, "efeito médio da macroregião; Nordeste é a referência"
    if nome.startswith("EstadoDev_"):
        estado = nome.replace("EstadoDev_", "")
        for macro, estados in estados_por_macro.items():
            if estado in estados:
                return nome, f"desvio dentro de {macro}; o último estado é a referência da macroregião"
    if nome.startswith("Estudo_x_Macro_"):
        return nome, "variação do retorno da escolaridade na macroregião"
    if nome.startswith("Formal_x_Macro_"):
        return nome, "variação da associação da formalidade na macroregião"
    if nome.startswith("Sexo_x_Macro_"):
        return nome, "variação da associação do sexo na macroregião"
    if nome.startswith("Publico_x_Macro_"):
        return nome, "variação da associação do setor público na macroregião"
    return nome, "efeito condicional às demais variáveis"


def gerar_coeficientes(modelo: Ridge, scaler: StandardScaler, x: pd.DataFrame, estados_por_macro: dict[str, list]) -> pd.DataFrame:
    beta = modelo.coef_ / scaler.scale_
    intercepto = modelo.intercept_ - np.sum(modelo.coef_ * scaler.mean_ / scaler.scale_)
    valores = dict(zip(x.columns, beta))
    linhas = [{
        "nivel": "modelo", "variavel": "Intercept", "coeficiente_linear": float(intercepto),
        "efeito_percentual_aprox": np.nan, "observacao": "nível de referência",
    }]
    for nome, valor in valores.items():
        label, obs = nomes_legiveis(nome, estados_por_macro)
        tipo = "macroregião" if nome.startswith("Macro_") else "desvio estadual" if nome.startswith("EstadoDev_") else "interação regional" if "_x_Macro_" in nome else "variável individual"
        efeito = np.expm1(valor) * 100 if tipo in {"macroregião", "desvio estadual"} else np.nan
        linhas.append({"nivel": tipo, "variavel": label, "coeficiente_linear": float(valor), "efeito_percentual_aprox": float(efeito) if np.isfinite(efeito) else np.nan, "observacao": obs})

    # Efeito estadual completo = efeito médio da macroregião + desvio estadual.
    for macro, estados in estados_por_macro.items():
        macro_beta = 0.0 if macro == MACRO_REFERENCIA else valores[f"Macro_{macro}"]
        desvios = {estado: valores[f"EstadoDev_{estado}"] for estado in estados[:-1]}
        desvios[estados[-1]] = -sum(desvios.values())
        for estado, desvio in desvios.items():
            efeito = macro_beta + desvio
            linhas.append({
                "nivel": "efeito estadual", "variavel": f"UF {estado} ({macro})",
                "coeficiente_linear": float(efeito), "efeito_percentual_aprox": float(np.expm1(efeito) * 100),
                "observacao": "efeito médio da macroregião + desvio do estado; média estadual dentro da macroregião é zero",
            })
    return pd.DataFrame(linhas)


def main() -> int:
    args = parse_args()
    amostras = carregar(args.entrada, args)
    estados_por_macro = {
        macro: sorted(amostras["treino"].loc[amostras["treino"]["Regiao"].eq(macro), "UF"].unique().tolist())
        for macro in MACROREGIOES
    }
    matrizes = {nome: criar_matriz(df, estados_por_macro) for nome, df in amostras.items()}
    x_treino = matrizes["treino"]
    scaler = StandardScaler().fit(x_treino)
    y_treino = amostras["treino"]["y"]

    validacoes = []
    modelos = {}
    for alpha in args.alphas:
        modelo = Ridge(alpha=alpha).fit(scaler.transform(x_treino), y_treino)
        pred = modelo.predict(scaler.transform(matrizes["validacao"]))
        resultado = metricas(amostras["validacao"]["y"], pred)
        resultado.update({"modelo": "M5_hierarquico_Ridge", "alpha": float(alpha), "tipo": "validacao"})
        validacoes.append(resultado)
        modelos[float(alpha)] = modelo
    melhor = min(validacoes, key=lambda row: row["rmse_log"])
    alpha = float(melhor["alpha"])
    modelo = modelos[alpha]
    resultados = [melhor]
    for conjunto in ("teste",):
        pred = modelo.predict(scaler.transform(matrizes[conjunto]))
        resultado = metricas(amostras[conjunto]["y"], pred)
        resultado.update({"modelo": "M5_hierarquico_Ridge", "alpha": alpha, "tipo": conjunto})
        resultados.append(resultado)

    args.saida_dir.mkdir(parents=True, exist_ok=True)
    metricas_path = args.saida_dir / "metricas_modelos.csv"
    metricas_df = pd.read_csv(metricas_path) if metricas_path.exists() else pd.DataFrame()
    if not metricas_df.empty:
        metricas_df = metricas_df[metricas_df["modelo"] != "M5_hierarquico_Ridge"]
    pd.concat([metricas_df, pd.DataFrame(resultados)], ignore_index=True).to_csv(metricas_path, index=False)
    coeficientes = gerar_coeficientes(modelo, scaler, x_treino, estados_por_macro)
    coeficientes.to_csv(args.saida_dir / "coeficientes_m5_hierarquico_ridge.csv", index=False)
    resumo = {
        "modelo": "M5_hierarquico_Ridge", "alpha_selecionado": alpha, "alphas_testados": args.alphas,
        "macroregiao_referencia": MACRO_REFERENCIA, "estados_por_macroregiao": estados_por_macro,
        "metricas_validacao": melhor, "metricas_teste": resultados[1], "n_coeficientes": int(len(coeficientes)),
        "estrutura": "macroregião + desvios de estado com soma zero dentro da macroregião + interações regionais de escolaridade, formalidade, sexo e setor público",
        "limitacao": "Ridge não fornece p-valores OLS convencionais; os coeficientes descrevem associações preditivas condicionais.",
    }
    (args.saida_dir / "resumo_m5_hierarquico_ridge.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
