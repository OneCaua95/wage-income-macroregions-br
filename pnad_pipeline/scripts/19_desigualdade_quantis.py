#!/usr/bin/env python3
"""Calcula desigualdade salarial e regressões quantílicas para a série PNAD.

Os indicadores distributivos usam toda a série ponderada por V1028. A
regressão quantílica usa um ponto de partida ponderado e estratificado por
trimestre × macroregião, seguido por duas correções dos scores no arquivo
completo. Os erros-padrão usam linearização por UPA dentro de estrato e uma
aproximação escalar para a densidade condicional no quantil.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

M5_SPEC = importlib.util.spec_from_file_location("_pnad_desigualdade_m5", SCRIPT_DIR / "08_inferencia_m5.py")
if M5_SPEC is None or M5_SPEC.loader is None:
    raise RuntimeError("Não foi possível carregar a normalização da análise M5.")
M5 = importlib.util.module_from_spec(M5_SPEC)
sys.modules[M5_SPEC.name] = M5
M5_SPEC.loader.exec_module(M5)

from pnad_pipeline.analise.amostragem import LinearizacaoUPA, exigir_colunas_desenho, pesos_amostrais

INPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet"
OUTPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/desigualdade"
QUANTIS = (0.10, 0.25, 0.50, 0.75, 0.90)
REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
SEXOS = {1: "Homens", 2: "Mulheres"}
CORES = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena"}
COLUNAS_AMOSTRA = [
    "Ano", "Trimestre", "UF", "Estrato", "UPA", "Regiao", "Sexo", "Cor", "Idade",
    "Anos_de_Estudo", "Categoria_emprego", "Contribuicao_previdencia", "y", "V1028",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=INPUT)
    parser.add_argument("--saida-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--amostra-quantis", type=int, default=30_000,
                        help="Máximo de linhas uniformemente amostradas por trimestre × macroregião.")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--upa-unica", choices=["erro", "ajustar"], default="ajustar")
    parser.add_argument("--upa-multiestrato", choices=["erro", "separar"], default="separar")
    return parser.parse_args()


def period_fields(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    year = pd.to_numeric(df["Ano"], errors="coerce").to_numpy(dtype=np.int32)
    quarter = pd.to_numeric(df["Trimestre"], errors="coerce").to_numpy(dtype=np.int32)
    order = year * 4 + quarter - 1
    labels = np.char.add(np.char.add(year.astype(str), "T"), quarter.astype(str))
    return year, quarter, labels


def metricas_distribuicao(values: np.ndarray, weights: np.ndarray, key: tuple) -> tuple[dict, list[dict]]:
    valid = np.isfinite(values) & np.isfinite(weights) & (values >= 0) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values) or weights.sum() <= 0:
        return {}, []
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    total_weight = float(weights.sum())
    weighted_income = values * weights
    total_income = float(weighted_income.sum())
    cumulative_weight = np.cumsum(weights)
    cumulative_income = np.cumsum(weighted_income)
    qs = np.asarray([.10, .25, .50, .75, .90])
    q_indexes = np.searchsorted(cumulative_weight, qs * total_weight, side="left")
    q_values = values[np.minimum(q_indexes, len(values) - 1)]
    x_lorenz = np.concatenate(([0.0], cumulative_weight / total_weight))
    y_lorenz = np.concatenate(([0.0], cumulative_income / total_income if total_income > 0 else np.zeros(len(values))))
    deciles = np.linspace(0, 1, 11)
    curve = np.interp(deciles, x_lorenz, y_lorenz)
    area = float(np.sum(np.diff(x_lorenz) * (y_lorenz[:-1] + y_lorenz[1:]) / 2))
    row = {
        "periodo": key[0], "dimensao": key[1], "grupo": key[2],
        "n_observacoes": int(len(values)), "peso_v1028": total_weight,
        "P10": float(q_values[0]), "P25": float(q_values[1]), "P50": float(q_values[2]),
        "P75": float(q_values[3]), "P90": float(q_values[4]),
        "razao_P90_P10": float(q_values[4] / q_values[0]) if q_values[0] > 0 else None,
        "gini": float(np.clip(1 - 2 * area, 0, 1)) if total_income > 0 else None,
    }
    lorenz = [{"periodo": key[0], "dimensao": key[1], "grupo": key[2],
               "participacao_populacao": float(x), "participacao_renda": float(y)}
              for x, y in zip(deciles, curve)]
    return row, lorenz


def colunas_design(df: pd.DataFrame) -> dict[str, np.ndarray]:
    age = pd.to_numeric(df["Idade"], errors="coerce").to_numpy(dtype=np.float64) - 45.0
    education = pd.to_numeric(df["Anos_de_Estudo"], errors="coerce").to_numpy(dtype=np.int16)
    sex = pd.to_numeric(df["Sexo"], errors="coerce").to_numpy(dtype=np.int16)
    color = pd.to_numeric(df["Cor"], errors="coerce").to_numpy(dtype=np.int16)
    year = pd.to_numeric(df["Ano"], errors="coerce").to_numpy(dtype=np.int16)
    quarter = pd.to_numeric(df["Trimestre"], errors="coerce").to_numpy(dtype=np.int16)
    employment = pd.to_numeric(df["Categoria_emprego"], errors="coerce")
    contribution = pd.to_numeric(df["Contribuicao_previdencia"], errors="coerce")
    formal = (employment.isin([1, 3, 5, 7]) |
              (employment.isin([8, 9]) & contribution.eq(1))).to_numpy(dtype=np.float64)
    public = employment.isin([5, 6, 7]).to_numpy(dtype=np.float64)
    regions = df["Regiao"].astype(str).to_numpy()
    columns: dict[str, np.ndarray] = {"Intercepto": np.ones(len(df), dtype=np.float64)}
    for level in range(1, 17):
        columns[f"Escolaridade_{level}"] = (education == level).astype(np.float64)
    columns["Idade_centrada"] = age / 10.0
    columns["Idade_centrada_2"] = (age / 10.0) ** 2
    columns["Sexo_Mulher"] = (sex == 2).astype(np.float64)
    for level in range(2, 6):
        columns[f"Cor_{level}"] = (color == level).astype(np.float64)
    for region in REGIOES[1:]:
        columns["Regiao_" + region.replace("-", "_")] = (regions == region).astype(np.float64)
    columns["Formal"] = formal
    columns["Setor_Publico"] = public
    for year_value in range(2014, 2026):
        columns[f"Ano_{year_value}"] = (year == year_value).astype(np.float64)
    for quarter_value in (2, 3, 4):
        columns[f"Trimestre_{quarter_value}"] = (quarter == quarter_value).astype(np.float64)
    return columns


def matriz_design(df: pd.DataFrame, terms: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    columns = colunas_design(df)
    if terms is None:
        terms = [name for name, values in columns.items()
                 if name == "Intercepto" or np.ptp(values) > 0]
    return np.column_stack([columns[name] for name in terms]), terms


def codigos_upa(desenho: LinearizacaoUPA, df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    keys = pd.MultiIndex.from_frame(df[["UF", "Estrato", "UPA"]])
    codes, unique = pd.factorize(keys, sort=False)
    tuples = [tuple(int(value) for value in key) for key in unique]
    return codes, tuples


def acumular_score(desenho: LinearizacaoUPA, df: pd.DataFrame, score: np.ndarray, offset: int) -> None:
    codes, tuples = codigos_upa(desenho, df)
    n_unique, k = len(tuples), score.shape[1]
    flat_index = codes[:, None] * k + np.arange(k, dtype=np.int64)[None, :]
    sums = np.bincount(flat_index.ravel(), weights=score.ravel(), minlength=n_unique * k).reshape(n_unique, k)
    ids = [desenho.grupos[key] for key in tuples]
    desenho.scores[np.asarray(ids, dtype=np.int64), offset:offset + k] += sums


def carne_estratificada(desenho: LinearizacaoUPA, scores: np.ndarray) -> np.ndarray:
    meat = np.zeros((scores.shape[1], scores.shape[1]), dtype=np.float64)
    global_mean = scores.mean(axis=0) if len(scores) else np.zeros(scores.shape[1])
    for ids in desenho.estratos.values():
        m = len(ids)
        if m < 2:
            if desenho.politica_upa_unica == "erro":
                raise ValueError("Há estrato com uma única UPA; use --upa-unica ajustar de forma explícita.")
            centered = scores[ids[0]] - global_mean
            meat += np.outer(centered, centered)
        else:
            centered = scores[ids] - scores[ids].mean(axis=0)
            meat += (m / (m - 1)) * (centered.T @ centered)
    return meat


def salvar_amostra_quantis(parts: dict, totals: dict, weight_totals: dict) -> pd.DataFrame:
    if not parts:
        raise ValueError("Não foi possível formar a amostra para as regressões quantílicas.")
    sample = pd.concat(parts.values(), ignore_index=True)
    sample.drop(columns=["_chave_aleatoria"], inplace=True, errors="ignore")
    sample["_ordem_periodo"] = sample["Ano"].astype(int) * 4 + sample["Trimestre"].astype(int) - 1
    sample["_grupo_regiao"] = sample["Regiao"].astype(str)
    sample["_peso_calibrado"] = sample["V1028"].astype(float)
    for key, index in sample.groupby(["_ordem_periodo", "_grupo_regiao"], sort=False).groups.items():
        key = (int(key[0]), str(key[1]))
        sample_weight = float(sample.loc[index, "V1028"].sum())
        if sample_weight <= 0 or key not in weight_totals:
            continue
        sample.loc[index, "_peso_calibrado"] *= weight_totals[key] / sample_weight
    return sample


def quantil_ponderado(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_weights = values[order], weights[order]
    index = np.searchsorted(np.cumsum(ordered_weights), quantile * ordered_weights.sum(), side="left")
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def ajuste_quantile_ponderado(x: np.ndarray, y: np.ndarray, weights: np.ndarray, tau: float,
                              max_iter: int = 1000, tolerance: float = 1e-6) -> tuple[np.ndarray, int, bool]:
    """IRLS para a função pinball com peso individual no objetivo."""
    sqrt_w = np.sqrt(weights)
    beta = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
    converged = False
    for iteration in range(1, max_iter + 1):
        residual = y - x @ beta
        near_zero = np.abs(residual) < 1e-6
        residual[near_zero] = np.where(residual[near_zero] >= 0, 1e-6, -1e-6)
        tilted_abs = np.abs(np.where(residual < 0, tau * residual, (1 - tau) * residual))
        irls_weight = weights / tilted_abs
        xtx = x.T @ (x * irls_weight[:, None])
        xty = x.T @ (y * irls_weight)
        updated = np.linalg.pinv(xtx, rcond=1e-11) @ xty
        difference = float(np.max(np.abs(updated - beta)))
        scale = max(float(np.max(np.abs(beta))), 1.0)
        beta = updated
        if difference <= tolerance * scale:
            converged = True
            break
    return beta, iteration, converged


def estimar_quantis(sample: pd.DataFrame, sample_limit: int) -> tuple[dict[float, np.ndarray], list[str], dict[float, float], dict]:
    values, terms = matriz_design(sample)
    y = sample["y"].to_numpy(dtype=np.float64)
    weights = sample["_peso_calibrado"].to_numpy(dtype=np.float64)
    weights = weights / weights.mean()
    estimates: dict[float, np.ndarray] = {}
    densities: dict[float, float] = {}
    iterations: dict[str, int] = {}
    convergence: dict[str, bool] = {}
    print(f"Ajustando regressões quantílicas ponderadas em {len(sample):,} registros, limite {sample_limit:,}.", flush=True)
    for tau in QUANTIS:
        estimates[tau], iterations[str(tau)], convergence[str(tau)] = ajuste_quantile_ponderado(values, y, weights, tau)
        residual = y - values @ estimates[tau]
        sigma = float(np.sqrt(np.average(residual ** 2, weights=weights)))
        iqr = quantil_ponderado(residual, weights, .75) - quantil_ponderado(residual, weights, .25)
        scale = min(sigma, iqr / 1.34) if iqr > 0 else sigma
        n_eff = float(weights.sum() ** 2 / np.square(weights).sum())
        bandwidth = max(1e-6, 1.06 * scale * n_eff ** (-.2))
        density = np.exp(-.5 * (residual / bandwidth) ** 2) / (np.sqrt(2 * np.pi) * bandwidth)
        densities[tau] = float(np.average(density, weights=weights))
        iterations[str(tau)] = int(iterations[str(tau)])
        convergence[str(tau)] = bool(convergence[str(tau)])
        print(f"  quantil {int(tau * 100)}: {iterations[str(tau)]} iterações; convergiu={convergence[str(tau)]}.", flush=True)
    info = {
        "n_amostra_quantis": int(len(sample)),
        "limite_amostra_quantis": int(sample_limit),
        "n_termos": int(len(terms)),
        "termos": terms,
        "quantis_ajustados": list(QUANTIS),
        "iteracoes_por_quantil": iterations,
        "convergiu_por_quantil": convergence,
        "densidade_media_por_quantil": {str(key): value for key, value in densities.items()},
    }
    return estimates, terms, densities, info


def executar_passada_scores(
    entrada: Path,
    batch_size: int,
    desenho: LinearizacaoUPA,
    terms: list[str],
    betas: dict[float, np.ndarray],
    densities: dict[float, float],
    xtwx: np.ndarray | None,
    accumulate: bool,
) -> tuple[dict[float, np.ndarray], np.ndarray | None]:
    k = len(terms)
    score_total = {tau: np.zeros(k, dtype=np.float64) for tau in QUANTIS}
    desenho.scores = np.zeros((len(desenho.grupos), k * len(QUANTIS)), dtype=np.float64)
    total_xtwx = np.zeros((k, k), dtype=np.float64) if xtwx is None and accumulate else xtwx
    reader = pq.ParquetFile(entrada)
    for index, batch in enumerate(reader.iter_batches(batch_size=batch_size, columns=M5.RAW_COLUMNS), 1):
        df = M5.normalizar_lote(batch)
        if not df.empty:
            x, _ = matriz_design(df, terms)
            y = df["y"].to_numpy(dtype=np.float64)
            w = pesos_amostrais(df)
            if accumulate:
                total_xtwx += x.T @ (x * w[:, None])
            for q_index, tau in enumerate(QUANTIS):
                residual = y - x @ betas[tau]
                below = np.where(residual < -1e-9, 1.0, np.where(residual > 1e-9, 0.0, 0.5))
                row_scores = x * (w * (tau - below))[:, None]
                score_total[tau] += row_scores.sum(axis=0)
                acumular_score(desenho, df, row_scores, q_index * k)
                del residual, below, row_scores
            del df, x, y, w
        if index % 25 == 0:
            print(f"  Lotes de inferência processados: {index}.", flush=True)
        del batch
    return score_total, total_xtwx


def format_label_education(code: int) -> str:
    if code == 1:
        return "1 ano"
    if code == 16:
        return "16 anos ou mais"
    return f"{code} anos"


def executar(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or args.amostra_quantis < 5_000:
        raise ValueError("O lote deve ser positivo e a amostra quantílica deve ter pelo menos 5.000 registros.")
    if not args.entrada.exists():
        raise FileNotFoundError(f"Base histórica ponderada não encontrada: {args.entrada}")
    parquet = pq.ParquetFile(args.entrada)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") == b"false":
        raise ValueError("Use a base consolidada completa, anterior aos filtros de domínio.")
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sample_cap = max(1, int(np.ceil(args.amostra_quantis / (52 * len(REGIOES)))))
    sample_parts: dict[tuple[int, str], pd.DataFrame] = {}
    row_counts: dict[tuple[int, str], int] = defaultdict(int)
    weight_totals: dict[tuple[int, str], float] = defaultdict(float)
    distribution_parts: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    design = LinearizacaoUPA(politica_upa_unica=args.upa_unica,
                             politica_upa_multiestrato=args.upa_multiestrato)
    n_eligible = 0
    total_weight = 0.0
    period_set: set[str] = set()

    print(f"Lendo a série ponderada em lotes de {args.batch_size:,}; calculando a distribuição completa e formando a amostra quantílica.", flush=True)
    for index, batch in enumerate(parquet.iter_batches(batch_size=args.batch_size, columns=M5.RAW_COLUMNS), 1):
        raw = batch.to_pandas()
        design.registrar(raw)
        df = M5.normalizar_lote(batch)
        if not df.empty:
            year, quarter, period = period_fields(df)
            df["_periodo"] = period
            df["_ordem_periodo"] = year * 4 + quarter - 1
            weights = pesos_amostrais(df)
            wages = np.exp(df["y"].to_numpy(dtype=np.float64))
            n_eligible += len(df)
            total_weight += float(weights.sum())
            period_set.update(np.unique(period).tolist())

            base = pd.DataFrame({"periodo": period, "salario_hora": wages, "peso": weights,
                                 "Regiao": df["Regiao"].astype(str).to_numpy(),
                                 "Sexo": pd.to_numeric(df["Sexo"], errors="coerce").to_numpy(dtype=np.int16),
                                 "Cor": pd.to_numeric(df["Cor"], errors="coerce").to_numpy(dtype=np.int16)})
            for (period_label, region), part in base.groupby(["periodo", "Regiao"], observed=True, sort=False):
                distribution_parts[(str(period_label), "regiao", str(region))].append(
                    (part.salario_hora.to_numpy(copy=True), part.peso.to_numpy(copy=True)))
            for (period_label, sex), part in base.groupby(["periodo", "Sexo"], observed=True, sort=False):
                distribution_parts[(str(period_label), "sexo", SEXOS.get(int(sex), str(sex)))].append(
                    (part.salario_hora.to_numpy(copy=True), part.peso.to_numpy(copy=True)))
            for (period_label, color), part in base.groupby(["periodo", "Cor"], observed=True, sort=False):
                distribution_parts[(str(period_label), "cor", CORES.get(int(color), str(color)))].append(
                    (part.salario_hora.to_numpy(copy=True), part.peso.to_numpy(copy=True)))
            for period_label, part in base.groupby("periodo", observed=True, sort=False):
                distribution_parts[(str(period_label), "geral", "Brasil")].append(
                    (part.salario_hora.to_numpy(copy=True), part.peso.to_numpy(copy=True)))

            for key, part in df.groupby(["_ordem_periodo", "Regiao"], observed=True, sort=False):
                stratum = (int(key[0]), str(key[1]))
                row_counts[stratum] += len(part)
                weight_totals[stratum] += float(part["V1028"].sum())
                candidate = part[COLUNAS_AMOSTRA].copy()
                candidate["_chave_aleatoria"] = rng.random(len(candidate))
                if stratum in sample_parts:
                    candidate = pd.concat([sample_parts[stratum], candidate], ignore_index=True)
                sample_parts[stratum] = candidate.nlargest(sample_cap, "_chave_aleatoria")
            del base, df, wages, weights
        if index % 25 == 0:
            print(f"Lotes de entrada processados: {index}; registros elegíveis: {n_eligible:,}.", flush=True)
        del batch, raw
        if index % 10 == 0:
            gc.collect()

    if n_eligible == 0:
        raise ValueError("Nenhum salário-hora válido foi encontrado no domínio analítico.")
    print(f"Distribuição apurada em {n_eligible:,} registros elegíveis. Calculando percentis, Gini e Lorenz.", flush=True)
    distribution_rows, lorenz_rows = [], []
    for key, fragments in distribution_parts.items():
        values = np.concatenate([piece[0] for piece in fragments])
        weights = np.concatenate([piece[1] for piece in fragments])
        row, lorenz = metricas_distribuicao(values, weights, key)
        if row:
            distribution_rows.append(row)
            lorenz_rows.extend(lorenz)
    distribution = pd.DataFrame(distribution_rows)
    lorenz = pd.DataFrame(lorenz_rows)
    del distribution_parts
    gc.collect()

    sample = salvar_amostra_quantis(sample_parts, row_counts, weight_totals)
    del sample_parts
    estimates, terms, densities, quantile_info = estimar_quantis(sample, args.amostra_quantis)
    k = len(terms)

    print("Primeira correção dos coeficientes pelos scores de toda a série.", flush=True)
    score_initial, xtwx = executar_passada_scores(
        args.entrada, args.batch_size, design, terms, estimates, densities, None, accumulate=True)
    betas_corrected = {}
    bread = {}
    first_steps = {}
    for tau in QUANTIS:
        a = densities[tau] * xtwx
        inverse = np.linalg.pinv(a, rcond=1e-11)
        step = inverse @ score_initial[tau]
        betas_corrected[tau] = estimates[tau] + step
        bread[tau] = inverse
        first_steps[str(tau)] = float(np.max(np.abs(step)))
    del score_initial

    print("Refinando os scores e calculando a variância por UPA dentro de estrato.", flush=True)
    score_final, _ = executar_passada_scores(
        args.entrada, args.batch_size, design, terms, betas_corrected, densities, xtwx, accumulate=False)
    coefficients = []
    alpha = 0.05
    df_design = max(int(design.graus_liberdade), 1)
    critical = float(stats.t.ppf(1 - alpha / 2, df_design))
    second_steps = {}
    for q_index, tau in enumerate(QUANTIS):
        offset = q_index * k
        cluster_scores = design.scores[:, offset:offset + k]
        meat = carne_estratificada(design, cluster_scores)
        covariance = bread[tau] @ meat @ bread[tau]
        covariance = (covariance + covariance.T) / 2
        second_step = bread[tau] @ score_final[tau]
        beta_final = betas_corrected[tau] + second_step
        second_steps[str(tau)] = float(np.max(np.abs(second_step)))
        for level in range(1, 17):
            name = f"Escolaridade_{level}"
            if name not in terms:
                continue
            term_index = terms.index(name)
            beta = float(beta_final[term_index])
            se = float(np.sqrt(max(float(covariance[term_index, term_index]), 0.0)))
            lower, upper = beta - critical * se, beta + critical * se
            p_value = float(2 * stats.t.sf(abs(beta / se), df_design)) if se > 0 else None
            coefficients.append({
                "quantil": tau, "quantil_rotulo": f"P{int(tau * 100)}",
                "anos_de_estudo_codigo": level, "anos_de_estudo": format_label_education(level),
                "referencia": "Código 0: sem instrução ou menos de 1 ano",
                "beta_log": beta, "erro_padrao": se,
                "ic95_inferior_log": float(lower), "ic95_superior_log": float(upper),
                "associacao_pct": float(100 * np.expm1(beta)),
                "ic95_inferior_pct": float(100 * np.expm1(lower)),
                "ic95_superior_pct": float(100 * np.expm1(upper)),
                "p_valor": p_value,
                "n_observacoes_elegiveis": int(n_eligible),
                "n_upas_desenho": int(len(design.grupos)),
                "n_estratos_desenho": int(len(design.estratos)),
                "graus_liberdade_desenho": int(df_design),
            })

    distribution.sort_values(["periodo", "dimensao", "grupo"], inplace=True)
    lorenz.sort_values(["periodo", "dimensao", "grupo", "participacao_populacao"], inplace=True)
    coefficients_frame = pd.DataFrame(coefficients)
    distribution.to_csv(args.saida_dir / "distribuicao_desigualdade.csv", index=False)
    lorenz.to_csv(args.saida_dir / "curva_lorenz.csv", index=False)
    coefficients_frame.to_csv(args.saida_dir / "coeficientes_quantis.csv", index=False)

    periods = sorted(period_set)
    summary = {
        "status": "ok", "periodo": f"{periods[0]}–{periods[-1]}",
        "n_observacoes_elegiveis": int(n_eligible), "peso_v1028_total": float(total_weight),
        "n_periodos": int(len(periods)), "n_upas_desenho": int(len(design.grupos)),
        "n_estratos_desenho": int(len(design.estratos)), "graus_liberdade_desenho": int(df_design),
        "dominio": "Pessoas ocupadas de 25 a 65 anos com rendimento e horas habituais positivos.",
        "desfecho": "Salário-hora real do trabalho principal, em reais de janeiro/2026.",
        "ponderacao_distribuicao": "Peso final V1028; percentis pela inversa da distribuição empírica ponderada, sem interpolação.",
        "desigualdade": "Gini e curva de Lorenz do salário-hora individual entre pessoas ocupadas; não é desigualdade de renda domiciliar.",
        "modelo_quantis": "log(salário-hora real) por escolaridade categórica, idade e idade², sexo, cor/raça, macroregião, formalidade, setor público, ano e trimestre.",
        "quantis": quantile_info,
        "linearizacao": "Scores ponderados por V1028, UPA dentro de estrato, aproximação com reposição; sem FPC nem recalibração dos pesos.",
        "densidade_condicional": "A matriz do quantil usa densidade escalar aproximada por kernel gaussiano dos resíduos ponderados; a heterogeneidade da densidade por covariável não é modelada.",
        "correcao_scores_max_abs": {"primeira": first_steps, "segunda": second_steps},
        "notas": [
            "Os indicadores descritivos usam o arquivo completo; os registros correspondem a pessoas-trimestre, não a pessoas únicas ao longo de 2013–2025.",
            "As regressões quantílicas são associações condicionais por posição na distribuição de log do salário-hora e não identificam causalidade.",
            "O ponto inicial das regressões usa amostra aleatória uniforme por trimestre × macroregião, calibrada pelos totais V1028; duas correções de scores percorrem a série completa.",
            "Os ICs são aproximações de linearização por UPA/estrato e usam densidade condicional escalar; não substituem variâncias oficiais específicas para regressão quantílica.",
            "Gini e Lorenz referem-se ao salário-hora de trabalhadores ocupados com rendimento e horas válidos; não medem renda familiar ou bem-estar do domicílio.",
        ],
    }
    (args.saida_dir / "resumo_desigualdade.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Desigualdade salarial e regressões quantílicas — PNAD Contínua", "",
        f"Série: {summary['periodo']} · observações elegíveis: {n_eligible:,} · UPAs: {len(design.grupos):,} · estratos: {len(design.estratos):,}.", "",
        "## Distribuição", "",
        "P10, P25, P50, P75 e P90 usam V1028 e a inversa da distribuição empírica ponderada, sem interpolação. A razão P90/P10 e o Gini descrevem dispersão do salário-hora individual entre trabalhadores ocupados. A curva de Lorenz foi amostrada em decis da distribuição ponderada.", "",
        "## Regressões quantílicas", "",
        "O desfecho é log do salário-hora real. A escolaridade entra por categorias (código 0 como referência), com controles de idade, sexo, cor/raça, macroregião, formalidade, setor público e efeitos de ano e trimestre. A amostra de otimização é estratificada por trimestre × macroregião e calibrada para os totais V1028; duas atualizações dos scores são calculadas sobre o arquivo completo.", "",
        "Os erros-padrão usam scores agrupados por UPA dentro de estrato e densidade escalar estimada por kernel gaussiano dos resíduos ponderados. A densidade varia por quantil, mas não por covariável. Essa é uma aproximação para viabilizar a série histórica; não substitui uma implementação oficial de regressão quantílica para amostras complexas. Leia os ICs junto às notas em `resumo_desigualdade.json`.", "",
        "## Limites", "",
        *[f"- {note}" for note in summary["notas"]], "",
    ]
    (args.saida_dir / "relatorio_desigualdade.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Concluído: {len(distribution):,} grupos de distribuição, {len(coefficients_frame):,} coeficientes quantílicos.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
