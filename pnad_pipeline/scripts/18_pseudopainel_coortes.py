#!/usr/bin/env python3
"""Constrói um pseudo-painel de coortes a partir das seções transversais PNAD.

As coortes têm composição fixa por sexo e faixa de nascimento de cinco anos.
Cada observação do painel é uma média ponderada V1028 de uma coorte-trimestre;
ela não acompanha indivíduos. O modelo usa efeitos fixos de coorte e trimestre,
controles de composição, tendências lineares diferenciais e sensibilidade ao
tamanho/efetivo das células. A variância em duas vias é exploratória, não
substitui a variância oficial pelo desenho amostral da PNAD.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

M5_SPEC = importlib.util.spec_from_file_location("_pnad_pseudopainel_m5", SCRIPT_DIR / "08_inferencia_m5.py")
if M5_SPEC is None or M5_SPEC.loader is None:
    raise RuntimeError("Não foi possível carregar a normalização da análise M5.")
M5 = importlib.util.module_from_spec(M5_SPEC)
sys.modules[M5_SPEC.name] = M5
M5_SPEC.loader.exec_module(M5)

from pnad_pipeline.analise.amostragem import exigir_colunas_desenho, pesos_amostrais

INPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet"
OUTPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/pseudopainel"
MINIMUMS = [(100, 50), (200, 100), (500, 250)]
MIN_PERIODS_PER_COHORT = 8
REGIONS = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=INPUT)
    parser.add_argument("--saida-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def labels_coortes() -> dict[int, str]:
    edges = M5.BIRTH_BINS
    return {i: f"{edges[i] if i == 0 else edges[i] + 1}–{edges[i+1]}" for i in range(len(edges)-1)}


def agregar_lote(batch: pd.DataFrame) -> pd.DataFrame:
    df = M5.normalizar_lote(batch)
    if df.empty:
        return pd.DataFrame()
    birth = pd.to_numeric(df["Ano_Nascimento"], errors="coerce").where(
        pd.to_numeric(df["Ano_Nascimento"], errors="coerce").between(1947, 2003),
        df["Ano"] - df["Idade"],
    )
    band = pd.cut(birth, bins=M5.BIRTH_BINS, labels=False, include_lowest=True).astype(int)
    df["coorte_id"] = df["Sexo"].astype(int) * 100 + band
    df["trimestre_id"] = df["Ano"].astype(int) * 4 + df["Trimestre"].astype(int) - 1
    df["w"] = pesos_amostrais(df)
    df["w2"] = df["w"] ** 2
    df["wy"] = df["w"] * df["y"]
    df["wsalario"] = df["w"] * np.exp(np.clip(df["y"], -50, 50))
    df["westudo"] = df["w"] * df["Anos_de_Estudo"]
    formal = (df["Categoria_emprego"].isin([1, 3, 5, 7]) |
              (df["Categoria_emprego"].isin([8, 9]) & df["Contribuicao_previdencia"].eq(1)))
    df["wformal"] = df["w"] * formal.to_numpy(dtype=float)
    df["wpublico"] = df["w"] * df["Categoria_emprego"].isin([5, 6, 7]).to_numpy(dtype=float)
    for code, name in [(2, "wpreta"), (3, "wamarela"), (4, "wparda"), (5, "windigena")]:
        df[name] = df["w"] * df["Cor"].eq(code).to_numpy(dtype=float)
    for region in REGIONS:
        df["wregiao_" + region.lower().replace("-", "_")] = df["w"] * df["Regiao"].eq(region).to_numpy(dtype=float)
    raw_birth = pd.to_numeric(df["Ano_Nascimento"], errors="coerce")
    df["w_nascimento_aproximado"] = df["w"] * (~raw_birth.between(1947, 2003)).to_numpy(dtype=float)
    grouped = df.groupby(["coorte_id", "trimestre_id"], observed=True, sort=False).agg(
        n=("y", "size"), w=("w", "sum"), w2=("w2", "sum"), wy=("wy", "sum"),
        wsalario=("wsalario", "sum"), westudo=("westudo", "sum"), wformal=("wformal", "sum"),
        wpublico=("wpublico", "sum"), wpreta=("wpreta", "sum"), wamarela=("wamarela", "sum"),
        wparda=("wparda", "sum"), windigena=("windigena", "sum"),
        w_nascimento_aproximado=("w_nascimento_aproximado", "sum"),
        **{"wregiao_" + r.lower().replace("-", "_"): ("wregiao_" + r.lower().replace("-", "_"), "sum") for r in REGIONS},
    ).reset_index()
    del df
    return grouped


def montar_cells(partes: list[pd.DataFrame]) -> pd.DataFrame:
    totals = pd.concat(partes, ignore_index=True).groupby(
        ["coorte_id", "trimestre_id"], observed=True, sort=True
    ).sum(numeric_only=True).reset_index()
    bands = labels_coortes()
    totals["sexo_codigo"] = totals["coorte_id"] // 100
    totals["faixa_nascimento_codigo"] = totals["coorte_id"] % 100
    totals["sexo"] = totals["sexo_codigo"].map({1: "Homens", 2: "Mulheres"})
    totals["faixa_nascimento"] = totals["faixa_nascimento_codigo"].map(bands)
    totals["periodo_ordem"] = totals["trimestre_id"]
    totals["Ano"] = (totals["trimestre_id"] // 4).astype(int)
    totals["Trimestre"] = (totals["trimestre_id"] % 4 + 1).astype(int)
    totals["periodo"] = totals["Ano"].astype(str) + "T" + totals["Trimestre"].astype(str)
    totals["n_efetivo_kish"] = totals["w"] ** 2 / totals["w2"]
    totals["media_log_salario_hora"] = totals["wy"] / totals["w"]
    totals["media_geometrica_salario_hora"] = np.exp(totals["media_log_salario_hora"])
    totals["media_salario_hora"] = totals["wsalario"] / totals["w"]
    for target, numerator in [("media_anos_estudo", "westudo"), ("proporcao_formal", "wformal"), ("proporcao_setor_publico", "wpublico"), ("proporcao_preta", "wpreta"), ("proporcao_amarela", "wamarela"), ("proporcao_parda", "wparda"), ("proporcao_indigena", "windigena")]:
        totals[target] = totals[numerator] / totals["w"]
    for region in REGIONS:
        suffix = region.lower().replace("-", "_")
        totals["proporcao_regiao_" + suffix] = totals["wregiao_" + suffix] / totals["w"]
    totals["proporcao_nascimento_aproximado"] = totals["w_nascimento_aproximado"] / totals["w"]
    totals["coorte"] = totals["sexo"] + " · " + totals["faixa_nascimento"]
    return totals


def design_matrix(cells: pd.DataFrame, specification: str) -> tuple[np.ndarray, list[str]]:
    cohorts = sorted(cells.coorte_id.unique())
    periods = sorted(cells.trimestre_id.unique())
    cohort_base, period_base = cohorts[0], periods[0]
    columns = [np.ones(len(cells))]
    names = ["Intercepto"]
    for cohort in cohorts[1:]:
        columns.append(cells.coorte_id.eq(cohort).to_numpy(dtype=float))
        names.append(f"Coorte_{cohort}")
    for period in periods[1:]:
        columns.append(cells.trimestre_id.eq(period).to_numpy(dtype=float))
        names.append(f"Trimestre_{period}")
    if specification in {"composicao", "tendencias"}:
        controls = ["media_anos_estudo", "proporcao_formal", "proporcao_setor_publico",
                    "proporcao_preta", "proporcao_amarela", "proporcao_parda", "proporcao_indigena",
                    "proporcao_regiao_nordeste", "proporcao_regiao_sudeste",
                    "proporcao_regiao_sul", "proporcao_regiao_centro_oeste"]
        for control in controls:
            columns.append(cells[control].to_numpy(dtype=float))
            names.append(control)
    if specification == "tendencias":
        trend = (cells.trimestre_id.to_numpy(dtype=float) - min(periods)) / 4.0
        for cohort in cohorts[1:]:
            columns.append(trend * cells.coorte_id.eq(cohort).to_numpy(dtype=float))
            names.append(f"Tendencia_anual_coorte_{cohort}_vs_{cohort_base}")
    return np.column_stack(columns), names


def cluster_meat(scores: np.ndarray, groups: np.ndarray, correction: float) -> np.ndarray:
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.zeros((len(unique), scores.shape[1]), dtype=float)
    np.add.at(sums, inverse, scores)
    meat = sums.T @ sums
    count = len(unique)
    if count > 1:
        meat *= count / (count - 1) * correction
    return meat


def fit_model(cells: pd.DataFrame, specification: str, minimum_n: int, minimum_ess: int) -> tuple[dict, list[dict]]:
    eligible = cells.loc[(cells.n >= minimum_n) & (cells.n_efetivo_kish >= minimum_ess)].copy()
    periods_per = eligible.groupby("coorte_id").trimestre_id.nunique()
    included = periods_per[periods_per >= MIN_PERIODS_PER_COHORT].index
    eligible = eligible.loc[eligible.coorte_id.isin(included)].copy()
    eligible.sort_values(["trimestre_id", "coorte_id"], inplace=True)
    if len(eligible) < 30 or eligible.coorte_id.nunique() < 3 or eligible.trimestre_id.nunique() < 8:
        raise ValueError(f"Células insuficientes no limiar n≥{minimum_n}, Kish≥{minimum_ess} para ajustar o pseudo-painel.")
    X, names = design_matrix(eligible, specification)
    y = eligible.media_log_salario_hora.to_numpy(dtype=float)
    weights = eligible.n_efetivo_kish.to_numpy(dtype=float)
    sqrt_w = np.sqrt(weights)
    beta, _, rank, _ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
    if rank < X.shape[1]:
        raise ValueError(f"Matriz não identificada no limiar n≥{minimum_n}, Kish≥{minimum_ess}, modelo {specification}: posto {rank}/{X.shape[1]}.")
    residual = y - X @ beta
    bread = np.linalg.inv((X * weights[:, None]).T @ X)
    scores = X * (weights * residual)[:, None]
    n, k = X.shape
    correction = (n - 1) / max(n - k, 1)
    g_cohort, g_period = eligible.coorte_id.nunique(), eligible.trimestre_id.nunique()
    meat_cohort = cluster_meat(scores, eligible.coorte_id.to_numpy(), correction)
    meat_period = cluster_meat(scores, eligible.trimestre_id.to_numpy(), correction)
    meat_cell = cluster_meat(scores, eligible["coorte_id"].to_numpy() * 10_000 + eligible["trimestre_id"].to_numpy(), correction)
    covariance = bread @ (meat_cohort + meat_period - meat_cell) @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    df = max(min(g_cohort, g_period) - 1, 1)
    critical = float(stats.t.ppf(.975, df))
    pvalues = np.where(se > 0, 2 * stats.t.sf(np.abs(np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)), df), np.nan)
    pvalues = np.minimum(pvalues, 1.0)
    sse = float(weights @ (residual ** 2))
    weighted_mean = float(weights @ y / weights.sum())
    sst = float(weights @ ((y - weighted_mean) ** 2))
    summary = {
        "limiar_n": minimum_n, "limiar_kish": minimum_ess, "especificacao": specification,
        "n_celulas": int(n), "n_coortes": int(g_cohort), "n_trimestres": int(g_period),
        "n_parametros": int(k), "graus_liberdade_aproximados": df,
        "rmse_log_ponderado": float(np.sqrt(sse / weights.sum())),
        "r2_ponderado": float(1 - sse / sst) if sst > 0 else None,
        "soma_pesos_efetivos": float(weights.sum()),
        "nota_ajuste": "Ajuste na amostra, com peso de precisão aproximado pelo n efetivo de Kish; não é validação preditiva.",
    }
    coefficients = [{
        "limiar_n": minimum_n, "limiar_kish": minimum_ess, "especificacao": specification,
        "termo": name, "coeficiente": float(b), "erro_padrao_2vias": float(s),
        "ic95_inferior": float(b - critical * s), "ic95_superior": float(b + critical * s),
        "p_valor_aproximado": float(p), "identificavel": True,
    } for name, b, s, p in zip(names, beta, se, pvalues)]
    return summary, coefficients


def executar(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    if not args.entrada.exists():
        raise FileNotFoundError(f"Base histórica ponderada não encontrada: {args.entrada}")
    parquet = M5.pq.ParquetFile(args.entrada)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") == b"false":
        raise ValueError("Use a base consolidada completa, anterior aos filtros de domínio.")
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    print(f"Agregando células por coorte-trimestre; leitura em lotes de {args.batch_size:,} linhas.", flush=True)
    partials = []
    for index, batch in enumerate(M5.pq.ParquetFile(args.entrada).iter_batches(batch_size=args.batch_size, columns=M5.RAW_COLUMNS), 1):
        grouped = agregar_lote(batch)
        if not grouped.empty:
            partials.append(grouped)
        if index % 25 == 0:
            print(f"Lotes lidos: {index}; parciais agregados: {len(partials)}.", flush=True)
        del batch, grouped
        gc.collect()
    cells = montar_cells(partials)
    del partials
    gc.collect()
    if cells.empty:
        raise ValueError("Nenhuma observação elegível para formar as coortes.")
    all_summaries, all_coefficients = [], []
    for minimum_n, minimum_ess in MINIMUMS:
        print(f"Estimando sensibilidade dos limiares: n>={minimum_n}, Kish>={minimum_ess}.", flush=True)
        for specification in ("efeitos_fixos", "composicao", "tendencias"):
            summary, coefficients = fit_model(cells, specification, minimum_n, minimum_ess)
            all_summaries.append(summary)
            all_coefficients.extend(coefficients)
    primary_n, primary_ess = MINIMUMS[0]
    counts = cells.loc[(cells.n >= primary_n) & (cells.n_efetivo_kish >= primary_ess)].groupby("coorte_id").trimestre_id.nunique()
    cells["celula_qualificada_principal"] = cells.apply(
        lambda r: bool(r.n >= primary_n and r.n_efetivo_kish >= primary_ess and counts.get(r.coorte_id, 0) >= MIN_PERIODS_PER_COHORT), axis=1
    )
    cell_columns = ["coorte_id", "coorte", "sexo", "sexo_codigo", "faixa_nascimento", "faixa_nascimento_codigo",
                    "Ano", "Trimestre", "periodo", "periodo_ordem", "n", "w", "n_efetivo_kish",
                    "media_log_salario_hora", "media_geometrica_salario_hora", "media_salario_hora",
                    "media_anos_estudo", "proporcao_formal", "proporcao_setor_publico",
                    "proporcao_regiao_norte", "proporcao_regiao_nordeste", "proporcao_regiao_sudeste",
                    "proporcao_regiao_sul", "proporcao_regiao_centro_oeste", "proporcao_nascimento_aproximado",
                    "celula_qualificada_principal"]
    cells[cell_columns].to_csv(args.saida_dir / "medias_coorte_trimestre.csv", index=False)
    comparison = pd.DataFrame(all_summaries)
    comparison.to_csv(args.saida_dir / "comparacao_especificacoes.csv", index=False)
    pd.DataFrame(all_coefficients).to_csv(args.saida_dir / "coeficientes_pseudopainel.csv", index=False)
    primary_cells = cells.loc[cells.celula_qualificada_principal]
    summary = {
        "status": "ok", "periodo": f"{cells.Ano.min()}T{cells.loc[cells.Ano.eq(cells.Ano.min()), 'Trimestre'].min()}–{cells.Ano.max()}T{cells.loc[cells.Ano.eq(cells.Ano.max()), 'Trimestre'].max()}",
        "n_observacoes_pessoa_trimestre": int(cells.n.sum()), "n_celulas_total": int(len(cells)),
        "n_celulas_qualificadas_principal": int(len(primary_cells)),
        "n_coortes_qualificadas_principal": int(primary_cells.coorte_id.nunique()),
        "lista_coortes_principal": primary_cells[["coorte_id", "coorte", "sexo", "faixa_nascimento"]]
            .drop_duplicates().sort_values("coorte_id").to_dict("records"),
        "n_trimestres_qualificados_principal": int(primary_cells.trimestre_id.nunique()),
        "limiar_principal": {"n_amostral_minimo": primary_n, "kish_minimo": primary_ess,
                             "trimestres_minimos_por_coorte": MIN_PERIODS_PER_COHORT},
        "limiares_sensibilidade": [{"n_amostral_minimo": n, "kish_minimo": e} for n, e in MINIMUMS],
        "referencia_coorte": "Primeira coorte identificável na amostra de cada limiar.",
        "efeito_fixo_coorte": "Sexo × faixa fixa de nascimento; nascimento fora do intervalo válido é aproximado por ano − idade.",
        "desfecho": "Média ponderada de log do salário-hora real (reais de janeiro/2026) por coorte-trimestre.",
        "ponderacao": "Média de cada célula com peso final V1028; WLS entre células ponderada pelo tamanho efetivo de Kish.",
        "modelos": all_summaries,
        "observacao_nascimento_aproximado_pct": float(100 * np.average(primary_cells.proporcao_nascimento_aproximado, weights=primary_cells.w)) if len(primary_cells) else None,
        "notas": [
            "Pseudo-painel de cortes transversais repetidos: médias de grupos, não trajetórias das mesmas pessoas.",
            "Coortes fixas por sexo e faixa de nascimento; escolaridade, formalidade, setor público e macroregião são características médias que podem mudar no tempo.",
            "As células e tendências são afetadas por erro amostral nas médias; os limiares são escolhas analíticas, sem valor universal.",
            "ICs/p-valores usam sanduíche clusterizado por coorte e trimestre, aproximação com poucos grupos e sem linearização oficial por estrato/UPA; trate como exploratórios.",
            "A PNAD tem rotação e domicílios revisitados; a covariância temporal não é integralmente identificada neste pseudo-painel.",
            "Modelos descritivos/associativos; não identificam efeito causal nem retorno individual da escolaridade.",
        ],
    }
    (args.saida_dir / "resumo_pseudopainel.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Pseudo-painel de coortes — PNAD Contínua", "",
        f"Série: {summary['periodo']} · células principais: {summary['n_celulas_qualificadas_principal']:,} · coortes: {summary['n_coortes_qualificadas_principal']} · trimestres: {summary['n_trimestres_qualificados_principal']}.", "",
        "## Construção", "",
        "As coortes são definidas por sexo e faixas fixas de nascimento compatíveis com a especificação histórica. Para cada coorte e trimestre, o arquivo registra média ponderada de log do salário-hora real, tamanho amostral, tamanho efetivo de Kish e médias/proporções de composição. A idade e a composição geográfica não definem a entrada/saída da coorte.", "",
        "A especificação principal usa células com pelo menos 100 observações e tamanho efetivo de Kish de pelo menos 50, e exige pelo menos oito trimestres qualificados por coorte. As sensibilidades usam (n, Kish) de (200, 100) e (500, 250). Não existe um limiar universal; valores maiores reduzem células ruidosas, mas removem parte do painel.", "",
        "## Modelos", "",
        "1. Efeitos fixos de coorte e trimestre; 2. adiciona escolaridade média e proporções formais, setor público, cor/raça e macroregião; 3. adiciona tendências lineares específicas por coorte, relativas à coorte de referência. WLS usa o tamanho efetivo de Kish como peso de precisão aproximado. A comparação de ajuste é dentro da amostra.", "",
        "## Limites de interpretação", "",
        *[f"- {note}" for note in summary["notas"]], "",
        "Erro de amostragem nas médias da coorte pode atenuar associações e contaminar as tendências. Os intervalos e p-valores são aproximações exploratórias clusterizadas por coorte e trimestre; eles não correspondem à linearização oficial da PNAD. Verifique `comparacao_especificacoes.csv` ao comparar os limiares.", "",
    ]
    (args.saida_dir / "relatorio_pseudopainel.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Concluído: {len(cells):,} células; {summary['n_coortes_qualificadas_principal']} coortes no painel principal.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
