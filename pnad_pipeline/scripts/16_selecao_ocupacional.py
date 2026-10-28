#!/usr/bin/env python3
"""Analisa seleção para a amostra de salário-hora da PNAD Contínua.

Estima a probabilidade de observar salário-hora positivo entre pessoas de
25–65 anos com escolaridade e covariáveis demográficas válidas. Aplica pesos
inversos dessa probabilidade como sensibilidade da associação educação–salário.
Não estima Heckman: os dados disponíveis não oferecem uma restrição de exclusão
defensável para identificar seleção por fatores não observados.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

M5_SPEC = importlib.util.spec_from_file_location("_pnad_selecao_m5_helpers", SCRIPT_DIR / "08_inferencia_m5.py")
if M5_SPEC is None or M5_SPEC.loader is None:
    raise RuntimeError("Não foi possível carregar os auxiliares da inferência M5.")
M5 = importlib.util.module_from_spec(M5_SPEC)
sys.modules[M5_SPEC.name] = M5
M5_SPEC.loader.exec_module(M5)

EDU_SPEC = importlib.util.spec_from_file_location("_pnad_selecao_educacao_helpers", SCRIPT_DIR / "15_inferencia_retorno_educacao.py")
if EDU_SPEC is None or EDU_SPEC.loader is None:
    raise RuntimeError("Não foi possível carregar os auxiliares da análise de escolaridade.")
EDU = importlib.util.module_from_spec(EDU_SPEC)
sys.modules[EDU_SPEC.name] = EDU
EDU_SPEC.loader.exec_module(EDU)

from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais

INPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet"
ANALYTIC_INPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_analise_25_65.parquet"
EDUCATION_RESULTS = PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/retorno_educacao/efeitos_retorno_educacao.csv"
OUTPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/selecao_ocupacional"

ANALYTIC_COLUMNS = [
    "Ano", "Trimestre", "UF", "V1028", "Sexo", "Idade", "Cor",
    "Anos_de_Estudo", "Condicao_Ocupacao", "Ocupado",
    "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas", "Regiao",
]
PISO_PROPENSAO = 0.05
CHECKPOINT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=INPUT)
    parser.add_argument("--entrada-analise", type=Path, default=ANALYTIC_INPUT)
    parser.add_argument("--efeitos-educacao", type=Path, default=EDUCATION_RESULTS)
    parser.add_argument("--saida-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--tolerancia", type=float, default=1e-7)
    parser.add_argument("--piso-propensao", type=float, default=PISO_PROPENSAO)
    parser.add_argument("--recalcular", action="store_true", help="Ignora e substitui o checkpoint desta estimação.")
    parser.add_argument("--upa-unica", choices=["erro", "ajustar"], default="ajustar")
    parser.add_argument("--upa-multiestrato", choices=["erro", "separar"], default="separar")
    return parser.parse_args()


def _numeric(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def dominio_comum(df: pd.DataFrame) -> np.ndarray:
    idade, estudo = _numeric(df, "Idade"), _numeric(df, "Anos_de_Estudo")
    sexo, cor, uf = _numeric(df, "Sexo"), _numeric(df, "Cor"), _numeric(df, "UF")
    ano, trimestre = _numeric(df, "Ano"), _numeric(df, "Trimestre")
    regiao = df["Regiao"].astype(str).to_numpy()
    return (
        np.isfinite(idade) & (idade >= 25) & (idade <= 65)
        & np.isfinite(estudo) & (estudo >= 0) & (estudo <= 16)
        & np.isin(sexo, [1, 2]) & np.isin(cor, [1, 2, 3, 4, 5])
        & np.isin(uf, M5.UFS) & np.isin(regiao, M5.MACROREGIOES)
        & np.isin(ano, np.arange(2013, 2026)) & np.isin(trimestre, [1, 2, 3, 4])
    )


def salario_observado(df: pd.DataFrame) -> np.ndarray:
    condicao = _numeric(df, "Condicao_Ocupacao")
    renda = _numeric(df, "Rendimento_hab_Trab_princ")
    horas = _numeric(df, "Horas_hab_trabalhadas")
    return (
        np.isfinite(condicao) & (condicao == 1)
        & np.isfinite(renda) & (renda > 0)
        & np.isfinite(horas) & (horas > 0)
    )


def nomes_selecao() -> list[str]:
    nomes = ["Intercept", "Sexo_Mulher"]
    nomes += [f"Escolaridade_{k}" for k in range(1, 17)]
    nomes += [f"Cor_{k}" for k in [2, 3, 4, 5]]
    nomes += ["Idade_centrada", "Idade_centrada_2", "Idade_x_mulher", "Idade2_x_mulher"]
    nomes += [f"Escolaridade_{k}_x_mulher" for k in range(1, 17)]
    nomes += [f"UF_{uf}" for uf in M5.UFS[1:]]
    nomes += [f"Ano_{ano}" for ano in range(2014, 2026)]
    nomes += [f"Trimestre_{tri}" for tri in (2, 3, 4)]
    return nomes


def matriz_selecao(df: pd.DataFrame) -> np.ndarray:
    idade = _numeric(df, "Idade") - 45.0
    sexo = _numeric(df, "Sexo")
    mulher = (sexo == 2).astype(np.float64)
    estudo, cor, uf = _numeric(df, "Anos_de_Estudo"), _numeric(df, "Cor"), _numeric(df, "UF")
    ano, trimestre = _numeric(df, "Ano"), _numeric(df, "Trimestre")
    colunas: list[np.ndarray] = [np.ones(len(df)), mulher]
    educacao = [(estudo == k).astype(np.float64) for k in range(1, 17)]
    colunas.extend(educacao)
    colunas.extend([(cor == k).astype(np.float64) for k in [2, 3, 4, 5]])
    idade2 = idade * idade / 100.0
    colunas.extend([idade / 10.0, idade2, mulher * idade / 10.0, mulher * idade2])
    colunas.extend([x * mulher for x in educacao])
    colunas.extend([(uf == code).astype(np.float64) for code in M5.UFS[1:]])
    colunas.extend([(ano == code).astype(np.float64) for code in range(2014, 2026)])
    colunas.extend([(trimestre == code).astype(np.float64) for code in (2, 3, 4)])
    return np.column_stack(colunas)


def _update_rate(group: dict, key: str, mask: np.ndarray, w: np.ndarray, occupied: np.ndarray, selected: np.ndarray) -> None:
    row = group.setdefault(key, {"n_alvo": 0, "n_ocupados": 0, "n_salario_valido": 0,
                                 "peso_alvo": 0.0, "peso_ocupados": 0.0, "peso_salario_valido": 0.0})
    row["n_alvo"] += int(mask.sum())
    row["n_ocupados"] += int((mask & occupied).sum())
    row["n_salario_valido"] += int((mask & selected).sum())
    row["peso_alvo"] += float(w[mask].sum())
    row["peso_ocupados"] += float(w[mask & occupied].sum())
    row["peso_salario_valido"] += float(w[mask & selected].sum())


def _labels(dimension: str, code: str) -> str:
    if dimension == "geral":
        return "Brasil"
    if dimension == "escolaridade":
        level = int(float(code))
        return "Sem instrução ou menos de 1 ano" if level == 0 else "16 anos ou mais" if level == 16 else f"{level} ano" if level == 1 else f"{level} anos"
    if dimension == "sexo":
        return {"1": "Homem", "2": "Mulher"}.get(str(int(float(code))), code)
    if dimension == "cor":
        return {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}.get(str(int(float(code))), code)
    return code


def _collect_rates(args: argparse.Namespace) -> tuple[dict, float]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(args.entrada_analise)
    required = set(ANALYTIC_COLUMNS)
    missing = sorted(required.difference(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"A base de análise não contém as colunas necessárias: {missing}")
    stats: dict[str, dict] = {key: {} for key in ["geral", "escolaridade", "sexo", "cor", "regiao"]}
    n_rows = 0
    for batch in parquet.iter_batches(columns=ANALYTIC_COLUMNS, batch_size=args.batch_size):
        df = batch.to_pandas()
        weights = pesos_amostrais(df)
        target = dominio_comum(df)
        occupied = _numeric(df, "Condicao_Ocupacao") == 1
        selected = target & occupied & salario_observado(df)
        _update_rate(stats["geral"], "Brasil", target, weights, occupied, selected)
        values_by_dimension = {
            "escolaridade": _numeric(df, "Anos_de_Estudo"),
            "sexo": _numeric(df, "Sexo"),
            "cor": _numeric(df, "Cor"),
            "regiao": df["Regiao"].astype(str).to_numpy(),
        }
        for dimension, values in values_by_dimension.items():
            for value in pd.unique(values[target]):
                code = str(value)
                _update_rate(stats[dimension], code, target & (values == value), weights, occupied, selected)
        n_rows += len(df)
        del df, target, values_by_dimension
    domain = stats["geral"]["Brasil"]
    if domain["n_alvo"] == 0:
        raise ValueError("A base não contém pessoas de 25 a 65 anos com covariáveis válidas.")
    mean_weight = domain["peso_alvo"] / domain["n_alvo"]
    if mean_weight <= 0:
        raise ValueError("Peso médio do domínio inválido.")
    return {"grupos": stats, "n_linhas_base": n_rows}, mean_weight


def _fit_logit(args: argparse.Namespace, mean_weight: float) -> tuple[np.ndarray, np.ndarray, int, bool]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(args.entrada_analise)
    gamma = np.zeros(len(nomes_selecao()), dtype=np.float64)
    converged = False
    information = np.zeros((len(gamma), len(gamma)), dtype=np.float64)
    for iteration in range(1, args.max_iter + 1):
        score = np.zeros(len(gamma), dtype=np.float64)
        information.fill(0.0)
        n = 0
        for batch in parquet.iter_batches(columns=ANALYTIC_COLUMNS, batch_size=args.batch_size):
            df = batch.to_pandas()
            mask = dominio_comum(df)
            if not mask.any():
                continue
            sub = df.loc[mask]
            x = matriz_selecao(sub)
            w = pesos_amostrais(sub) / mean_weight
            y = (sub["Ocupado"].to_numpy(dtype=bool)
                 & salario_observado(sub)).astype(np.float64)
            p = expit(x @ gamma)
            variance = np.maximum(p * (1.0 - p), 1e-10)
            score += x.T @ (w * (y - p))
            information += x.T @ (x * (w * variance)[:, None])
            n += len(sub)
            del df, sub, x, w, y, p, variance
        try:
            delta = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(information, score, rcond=1e-11)[0]
        largest = float(np.max(np.abs(delta)))
        damping = min(1.0, 2.0 / largest) if largest > 0 else 1.0
        gamma += delta * damping
        print(f"Logit de seleção · iteração {iteration}: max |passo|={largest * damping:.3g}; {n:,} pessoas no domínio")
        if largest * damping < args.tolerancia:
            converged = True
            break
        gc.collect()
    if not converged:
        raise RuntimeError(f"Logit de seleção não convergiu em {args.max_iter} iterações.")
    # Recalcula a informação no parâmetro final para a variância de duas etapas.
    information.fill(0.0)
    for batch in parquet.iter_batches(columns=ANALYTIC_COLUMNS, batch_size=args.batch_size):
        df = batch.to_pandas()
        mask = dominio_comum(df)
        if not mask.any():
            continue
        sub = df.loc[mask]
        x = matriz_selecao(sub)
        w = pesos_amostrais(sub) / mean_weight
        p = expit(x @ gamma)
        information += x.T @ (x * (w * np.maximum(p * (1.0 - p), 1e-10))[:, None])
        del df, sub, x, w, p
    return gamma, information, iteration, converged


def _weighted_quantile_from_hist(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    total = hist.sum()
    if total <= 0:
        return float("nan")
    i = min(int(np.searchsorted(np.cumsum(hist), total * q, side="left")), len(hist) - 1)
    return float((edges[i] + edges[i + 1]) / 2)


def _estimate_outcomes(args: argparse.Namespace, gamma: np.ndarray, mean_weight: float):
    base = EDU.ModeloEducacao("categorias", "geral", EDU.nomes_modelo("categorias", "geral"))
    ipw = EDU.ModeloEducacao("categorias", "geral", EDU.nomes_modelo("categorias", "geral"))
    k_gamma, k_beta = len(gamma), len(base.nomes)
    cross = np.zeros((k_beta, k_gamma), dtype=np.float64)
    hist = np.zeros(1000, dtype=np.float64)
    histogram_edges = np.linspace(0.0, 1.0, len(hist) + 1)
    p_min, p_max = 1.0, 0.0
    target_n = floor_n = floor_selected_n = 0
    target_w = floor_w = floor_selected_w = floor_selected_ipw_w = 0.0
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(args.entrada)
    design = LinearizacaoUPA(
        politica_upa_unica=args.upa_unica,
        politica_upa_multiestrato=args.upa_multiestrato,
    )
    for batch in M5.iterar_lotes(args.entrada, args.batch_size):
        raw = batch.to_pandas()
        design.registrar(raw)
        mask = dominio_comum(raw)
        if mask.any():
            target = raw.loc[mask]
            x = matriz_selecao(target)
            p = expit(x @ gamma)
            w = pesos_amostrais(target) / mean_weight
            bins = np.minimum((p * len(hist)).astype(int), len(hist) - 1)
            hist += np.bincount(bins, weights=w, minlength=len(hist))
            p_min, p_max = min(p_min, float(p.min())), max(p_max, float(p.max()))
            floor_mask = p < args.piso_propensao
            floor_n += int(floor_mask.sum())
            floor_w += float(w[floor_mask].sum())
            target_n += len(target)
            target_w += float(w.sum())
            del target, x, p, w, bins, floor_mask
        selected = M5.normalizar_lote(batch)
        if not selected.empty:
            xsel = matriz_selecao(selected)
            psel = expit(xsel @ gamma)
            pused = np.maximum(psel, args.piso_propensao)
            wsel = pesos_amostrais(selected) / mean_weight
            selected_floor = psel < args.piso_propensao
            floor_selected_n += int(selected_floor.sum())
            floor_selected_w += float(wsel[selected_floor].sum())
            floor_selected_ipw_w += float((wsel[selected_floor] / args.piso_propensao).sum())
            y = selected["y"].to_numpy(dtype=np.float64)
            z = EDU.matriz_modelo(selected, "categorias", "geral")
            base.acumular_normal(z, y, wsel)
            ipw_weights = wsel / pused
            ipw.acumular_normal(z, y, ipw_weights)
            del selected, xsel, psel, pused, wsel, y, z, ipw_weights
        del raw
        gc.collect()
    base.resolver()
    ipw.resolver()
    # A informação do segundo estágio e o bloco cruzado são acumulados com os
    # coeficientes finais na passada de scores imediatamente seguinte.
    return (base, ipw, cross, hist, histogram_edges, p_min, p_max, floor_n, floor_w,
            floor_selected_n, floor_selected_w, floor_selected_ipw_w, target_n, target_w, design)


def _count_limited_salary_weights(args: argparse.Namespace, gamma: np.ndarray, mean_weight: float):
    """Conta pesos IPW realmente limitados na amostra salarial observada."""
    n_limited = 0
    base_weight_limited = ipw_weight_limited = 0.0
    for batch in M5.iterar_lotes(args.entrada, args.batch_size):
        selected = M5.normalizar_lote(batch)
        if selected.empty:
            continue
        p = expit(matriz_selecao(selected) @ gamma)
        low = p < args.piso_propensao
        if low.any():
            w = pesos_amostrais(selected) / mean_weight
            n_limited += int(low.sum())
            base_weight_limited += float(w[low].sum())
            ipw_weight_limited += float((w[low] / args.piso_propensao).sum())
    return n_limited, base_weight_limited, ipw_weight_limited


def _score_pass(args: argparse.Namespace, gamma: np.ndarray, mean_weight: float,
                ipw: EDU.ModeloEducacao, information_gamma: np.ndarray,
                design: LinearizacaoUPA):
    k_gamma, k_beta = len(gamma), len(ipw.nomes)
    design.iniciar_scores(k_gamma + k_beta)
    cross = np.zeros((k_beta, k_gamma), dtype=np.float64)
    for batch in M5.iterar_lotes(args.entrada, args.batch_size):
        raw = batch.to_pandas()
        mask = dominio_comum(raw)
        if not mask.any():
            continue
        target = raw.loc[mask]
        x = matriz_selecao(target)
        p = expit(x @ gamma)
        w = pesos_amostrais(target) / mean_weight
        selected = M5.normalizar_lote(batch)
        idx = np.searchsorted(target.index.to_numpy(), selected.index.to_numpy()) if not selected.empty else np.array([], dtype=int)
        if len(idx) and ((idx >= len(target)).any() or not np.array_equal(target.index.to_numpy()[idx], selected.index.to_numpy())):
            raise RuntimeError("Não foi possível alinhar os salários observados com o domínio da seleção.")
        r = np.zeros(len(target), dtype=bool)
        r[idx] = True
        scores = np.zeros((len(target), k_gamma + k_beta), dtype=np.float64)
        scores[:, :k_gamma] = x * (w * (r.astype(np.float64) - p))[:, None]
        if not selected.empty:
            xsel = matriz_selecao(selected)
            psel = expit(xsel @ gamma)
            pused = np.maximum(psel, args.piso_propensao)
            wsel = pesos_amostrais(selected) / mean_weight
            y = selected["y"].to_numpy(dtype=np.float64)
            z = EDU.matriz_modelo(selected, "categorias", "geral")
            residual = y - z @ ipw.beta
            scores[idx, k_gamma:] = z * (wsel / pused * residual)[:, None]
            active_derivative = psel >= args.piso_propensao
            cross += z.T @ (xsel * (wsel * residual * np.where(active_derivative, (1.0 - psel) / pused, 0.0))[:, None])
            del xsel, psel, pused, wsel, y, z, residual, active_derivative
        design.acumular(target, scores)
        del raw, target, x, p, w, selected, idx, r, scores
        gc.collect()
    if ipw.n != 9_064_317:
        print(f"Amostra IPW selecionada: {ipw.n:,} observações (referência anterior: 9,064,317)")
    try:
        inv_gamma = np.linalg.inv(information_gamma)
    except np.linalg.LinAlgError:
        inv_gamma = np.linalg.pinv(information_gamma, rcond=1e-11)
    bread = np.zeros((k_gamma + k_beta, k_gamma + k_beta), dtype=np.float64)
    bread[:k_gamma, :k_gamma] = inv_gamma
    bread[k_gamma:, k_gamma:] = ipw.bread
    bread[k_gamma:, :k_gamma] = -ipw.bread @ cross @ inv_gamma
    meat = design.covariancia(np.eye(k_gamma + k_beta, dtype=np.float64))
    covariance = bread @ meat @ bread.T
    covariance = (covariance + covariance.T) / 2
    return design, covariance


def _rates_frame(stats: dict) -> pd.DataFrame:
    rows = []
    for dimension, groups in stats.items():
        for code, values in groups.items():
            den = values["peso_alvo"]
            rows.append({
                "dimensao": dimension, "grupo_codigo": code,
                "grupo": _labels(dimension, code),
                **values,
                "taxa_ocupacao_pct": 100.0 * values["peso_ocupados"] / den if den else None,
                "taxa_salario_valido_pct": 100.0 * values["peso_salario_valido"] / den if den else None,
                "taxa_salario_valido_entre_ocupados_pct": 100.0 * values["peso_salario_valido"] / values["peso_ocupados"] if values["peso_ocupados"] else None,
            })
    frame = pd.DataFrame(rows)
    frame["_ordem"] = pd.to_numeric(frame["grupo_codigo"], errors="coerce")
    frame = frame.sort_values(["dimensao", "_ordem", "grupo"], kind="stable", na_position="last")
    return frame.drop(columns="_ordem").reset_index(drop=True)


def _effects_frame(base: EDU.ModeloEducacao, ipw: EDU.ModeloEducacao, covariance: np.ndarray,
                   education_path: Path, df_design: int, confidence: float) -> pd.DataFrame:
    previous = pd.read_csv(education_path)
    previous = previous.loc[
        previous.especificacao.eq("categorias")
        & previous.dimensao.eq("geral")
        & previous.contraste.eq("acumulado")
    ].set_index("anos_de_estudo_codigo")
    if len(previous) != 16:
        raise ValueError("A tabela anterior de retorno da educação não contém os 16 contrastes esperados.")
    crit = float(stats.t.ppf(1 - (1 - confidence) / 2, max(df_design, 1)))
    rows, pvals = [], []
    offset = covariance.shape[0] - len(ipw.beta)
    cov_beta = covariance[offset:, offset:]
    for level in range(1, 17):
        idx = ipw.nomes.index(f"Escolaridade_{level}")
        beta = float(ipw.beta[idx])
        se = float(np.sqrt(max(cov_beta[idx, idx], 0.0)))
        lo, hi = beta - crit * se, beta + crit * se
        p = float(2 * stats.t.sf(abs(beta / se), max(df_design, 1))) if se > 0 else np.nan
        pvals.append(p)
        before = previous.loc[level]
        rows.append({
            "anos_de_estudo_codigo": level,
            "anos_de_estudo": _labels("escolaridade", str(level)),
            "beta_base_log": float(before.beta_log),
            "associacao_base_pct": float(before.associacao_pct),
            "ic_base_inferior_pct": float(before.ic_inferior_pct),
            "ic_base_superior_pct": float(before.ic_superior_pct),
            "beta_ipw_log": beta,
            "erro_padrao_ipw_log": se,
            "associacao_ipw_pct": float(100.0 * np.expm1(beta)),
            "ic_ipw_inferior_pct": float(100.0 * np.expm1(lo)),
            "ic_ipw_superior_pct": float(100.0 * np.expm1(hi)),
            "mudanca_associacao_pp": float(100.0 * np.expm1(beta) - before.associacao_pct),
            "p_valor_ipw": p,
            "p_valor_base": float(before.p_valor),
            "p_valor_base_bh": float(before.p_valor_bh),
        })
    frame = pd.DataFrame(rows)
    frame["p_valor_ipw_bh"] = M5.ajustar_bh(np.asarray(pvals, dtype=np.float64))
    return frame


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def executar(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or args.max_iter < 1 or not 0 < args.piso_propensao < 1:
        raise ValueError("Tamanho de lote, iterações ou piso de propensão inválido.")
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.saida_dir / "checkpoint_selecao_ocupacional.pkl"
    signature = {
        "versao": CHECKPOINT_VERSION,
        "entradas": [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                     for path in [args.entrada, args.entrada_analise, args.efeitos_educacao]],
        "batch_size": args.batch_size, "piso": args.piso_propensao,
        "upa_unica": args.upa_unica, "upa_multiestrato": args.upa_multiestrato,
    }
    state = None
    if checkpoint_path.exists() and not args.recalcular:
        with checkpoint_path.open("rb") as stream:
            candidate = pickle.load(stream)
        if candidate.get("signature") == signature and candidate.get("stage") == "modelo":
            state = candidate
            print("Retomando da etapa de modelo já estimada.")

    if state is None:
        print("Passagem 1: taxas ponderadas de ocupação e de salário-hora válido")
        descriptive, mean_weight = _collect_rates(args)
        target = descriptive["grupos"]["geral"]["Brasil"]
        print(f"Domínio comum: {target['n_alvo']:,}; seleção salarial: {target['n_salario_valido']:,}")

        print("Passagens seguintes: estimação da probabilidade de observar salário-hora")
        gamma, information_gamma, iterations, converged = _fit_logit(args, mean_weight)
        print("Passagem seguinte: estimação do modelo salarial com pesos inversos")
        (base, ipw, _, hist, hist_edges, p_min, p_max, floor_n, floor_w,
         floor_selected_n, floor_selected_w, floor_selected_ipw_w,
         target_n, target_w, design) = _estimate_outcomes(args, gamma, mean_weight)
        if target_n != target["n_alvo"]:
            raise RuntimeError(f"Domínio diferente entre arquivos: {target_n:,} vs {target['n_alvo']:,}.")
        state = {
            "signature": signature, "stage": "modelo", "descriptive": descriptive,
            "mean_weight": mean_weight, "gamma": gamma, "information_gamma": information_gamma,
            "iterations": iterations, "converged": converged, "base": base, "ipw": ipw,
            "hist": hist, "hist_edges": hist_edges, "p_min": p_min, "p_max": p_max,
            "floor_n": floor_n, "floor_w": floor_w, "target_n": target_n,
            "floor_selected_n": floor_selected_n, "floor_selected_w": floor_selected_w,
            "floor_selected_ipw_w": floor_selected_ipw_w,
            "target_w": target_w, "design": design,
        }
        with checkpoint_path.open("wb") as stream:
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        descriptive, mean_weight = state["descriptive"], state["mean_weight"]
        gamma, information_gamma = state["gamma"], state["information_gamma"]
        iterations, converged = state["iterations"], state["converged"]
        base, ipw = state["base"], state["ipw"]
        hist, hist_edges = state["hist"], state["hist_edges"]
        p_min, p_max = state["p_min"], state["p_max"]
        floor_n, floor_w = state["floor_n"], state["floor_w"]
        target_n, target_w, design = state["target_n"], state["target_w"], state["design"]
        target = descriptive["grupos"]["geral"]["Brasil"]
    if "floor_selected_n" not in state:
        print("Contando pesos limitados efetivamente usados na amostra salarial")
        (floor_selected_n, floor_selected_w,
         floor_selected_ipw_w) = _count_limited_salary_weights(args, gamma, mean_weight)
        state["floor_selected_n"] = floor_selected_n
        state["floor_selected_w"] = floor_selected_w
        state["floor_selected_ipw_w"] = floor_selected_ipw_w
        with checkpoint_path.open("wb") as stream:
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        floor_selected_n = state["floor_selected_n"]
        floor_selected_w = state["floor_selected_w"]
        floor_selected_ipw_w = state["floor_selected_ipw_w"]
    print("Passagem final: variância por UPA e estrato, incluindo a estimação da propensão")
    design, covariance = _score_pass(args, gamma, mean_weight, ipw, information_gamma, design)
    if base.n != target["n_salario_valido"] or ipw.n != target["n_salario_valido"]:
        raise RuntimeError(f"A amostra salarial não coincide com o domínio publicado: base={base.n:,}, IPW={ipw.n:,}, esperado={target['n_salario_valido']:,}.")

    rates = _rates_frame(descriptive["grupos"])
    rates_path = args.saida_dir / "taxas_selecao.csv"
    rates.to_csv(rates_path, index=False)
    effects = _effects_frame(base, ipw, covariance, args.efeitos_educacao, design.graus_liberdade, 0.95)
    effects_path = args.saida_dir / "sensibilidade_ipw_escolaridade.csv"
    effects.to_csv(effects_path, index=False)
    min_change = float(effects["mudanca_associacao_pp"].min())
    max_change = float(effects["mudanca_associacao_pp"].max())

    median_p = _weighted_quantile_from_hist(hist, hist_edges, .50)
    p05 = _weighted_quantile_from_hist(hist, hist_edges, .05)
    p95 = _weighted_quantile_from_hist(hist, hist_edges, .95)
    eff_size = ipw.soma_pesos ** 2 / ipw.soma_pesos2 if ipw.soma_pesos2 else None
    summary = {
        "entrada": str(args.entrada), "entrada_analise": str(args.entrada_analise),
        "periodo": "2013–2025, 52 trimestres empilhados", "ponderacao": "V1028",
        "dominio_alvo": "Pessoas de 25–65 anos com escolaridade (0–16), sexo, cor/raça e UF válidos; amostra pessoa-trimestre, não pessoas únicas.",
        "evento_selecao_ipw": "Salário-hora observável: ocupada (VD4002=1), rendimento habitual positivo e horas habituais positivas, mesma elegibilidade do modelo de retorno da escolaridade.",
        "n_linhas_base": descriptive["n_linhas_base"], "n_alvo": target["n_alvo"],
        "peso_alvo": target["peso_alvo"], "n_ocupados": target["n_ocupados"],
        "peso_ocupados": target["peso_ocupados"], "n_salario_valido": target["n_salario_valido"],
        "peso_salario_valido": target["peso_salario_valido"],
        "taxa_ocupacao_pct": 100 * target["peso_ocupados"] / target["peso_alvo"],
        "taxa_salario_valido_pct": 100 * target["peso_salario_valido"] / target["peso_alvo"],
        "propensao": {"n_coeficientes": len(gamma), "iteracoes": iterations, "convergiu": converged,
                      "piso": args.piso_propensao, "min": p_min, "p05_ponderado": p05,
                      "mediana_ponderada": median_p, "p95_ponderado": p95, "max": p_max,
                      "n_abaixo_piso": floor_n, "peso_abaixo_piso_escala_normalizada": floor_w,
                      "peso_abaixo_piso_pct": 100.0 * floor_w / target_w if target_w else None,
                      "n_pesos_salariais_limitados": floor_selected_n,
                      "pct_amostra_salario_com_peso_limitado": 100.0 * floor_selected_n / ipw.n if ipw.n else None,
                      "peso_base_salario_limitado_escala_normalizada": floor_selected_w,
                      "peso_ipw_salario_limitado_escala_normalizada": floor_selected_ipw_w,
                      "maximo_fator_ipw": 1 / args.piso_propensao},
        "modelo_salario": {"n": ipw.n, "n_parametros": len(ipw.ativos),
                           "n_efetivo_kish_ipw": eff_size,
                           "n_efetivo_kish_pesos_v1028": base.soma_pesos ** 2 / base.soma_pesos2,
                           "n_upas": len(design.grupos), "n_estratos": len(design.estratos),
                           "graus_liberdade_desenho": design.graus_liberdade},
        "sensibilidade": {"mudanca_minima_pp": min_change, "mudanca_maxima_pp": max_change,
                          "n_associacoes_maiores_que_zero": int((effects["mudanca_associacao_pp"] > 0).sum())},
        "desenho": "Linearização por UPA dentro de estrato; aproximação com reposição, sem FPC ou recalibração. A covariância IPW incorpora a estimação da propensão pelo sistema empilhado de scores.",
        "identificacao": "Não há variável de exclusão defensável entre as colunas disponíveis para uma correção de Heckman. O IPW é apenas uma sensibilidade sob seleção explicada pelas covariáveis observadas; seleção por fatores não observados continua não identificada.",
        "limitacao_temporal": "Os 52 trimestres são empilhados; rotação de domicílios, mudanças de UPA/estrato e renovação gradual da Amostra Mestra limitam a variância temporal. A estimação não é a variância oficial do IBGE.",
        "arquivos": {"taxas": rates_path.name, "sensibilidade": effects_path.name},
    }
    summary = _json_clean(summary)
    (args.saida_dir / "resumo_selecao_ocupacional.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Seleção ocupacional e disponibilidade do salário-hora", "",
        "## O que foi analisado", "",
        "A amostra de renda observa apenas pessoas ocupadas com rendimento e horas positivos. Para mostrar como essa seleção ocorre, o relatório separa a taxa de ocupação da taxa de inclusão na amostra salarial. A análise cobre 25–65 anos e os 52 trimestres de 2013–2025; cada linha é uma observação pessoa-trimestre, não uma pessoa única.", "",
        f"No domínio com covariáveis válidas, há {target['n_alvo']:,} observações. A taxa ponderada de ocupação é {summary['taxa_ocupacao_pct']:.2f}%; {summary['taxa_salario_valido_pct']:.2f}% têm salário-hora positivo e entram na regressão. O grupo selecionado coincide com as {ipw.n:,} observações usadas na análise anterior da educação.", "",
        "A taxa de ocupação varia muito: mulheres têm participação menor que homens, e os grupos com maior escolaridade entram mais frequentemente na amostra salarial. Essas diferenças mostram por que a seleção precisa ser examinada; não provam que a seleção por si só cause as diferenças salariais.", "",
        "## Sensibilidade por ponderação inversa", "",
        "Foi estimada uma regressão logística ponderada por V1028 para a probabilidade de a observação pertencer à amostra com salário-hora válido. A equação usa escolaridade categórica, sexo e interações sexo–escolaridade, cor/raça, idade e idade² com interação por sexo, UF, ano e trimestre. Não usa setor, formalidade ou ocupação, que só estão definidos depois da ocupação.", "",
        "Como sensibilidade, a regressão salarial por categorias da análise anterior foi reestimada com pesos `V1028 / probabilidade estimada`, limitando o inverso a 20 (piso de propensão de 0,05). O intervalo de confiança do segundo estágio incorpora a estimação da propensão por uma linearização conjunta dos scores dos dois modelos. As taxas por grupo são descritivas e ponderadas por V1028.", "",
        f"No domínio-alvo, {floor_n:,} previsões ficaram abaixo de 5%; entre as {ipw.n:,} observações da amostra salarial, {floor_selected_n:,} pesos inversos foram efetivamente limitados a 20 ({100 * floor_selected_n / ipw.n:.4f}%).", "",
        f"A reponderação aumentou as 16 associações categóricas entre {min_change:+.2f} e {max_change:+.2f} pontos percentuais; a mudança maior foi para 16 anos ou mais. Essa variação é descritiva: não foi testada como diferença entre dois estimadores dependentes.", "",
        "A comparação só ajusta a seleção associada às covariáveis observadas. Não substitui a identificação de salário potencial fora da ocupação e não é um efeito causal.", "",
        "## Identificação e limites", "",
        "Uma correção clássica de Heckman requer uma variável que altere a probabilidade de ocupação, mas não afete diretamente o salário condicional. As colunas disponíveis para esta série não contêm uma restrição de exclusão que possa ser defendida; por isso, não foi estimado Heckman nem alegado que a seleção não observada foi corrigida. O IPW deve ser lido como análise de sensibilidade sob a hipótese, não verificável, de seleção explicada pelas covariáveis observadas.", "",
        "Os erros-padrão usam linearização por UPA dentro de estrato, aproximação com reposição, sem FPC ou recalibração. O empilhamento ignora covariância entre trimestres de domicílios revisitados e entre versões de UPA reclassificada; a variância não reproduz integralmente a variância oficial do IBGE.", "",
        "## Arquivos", "",
        "- `taxas_selecao.csv`: taxas ponderadas de ocupação e salário-hora válido por escolaridade, sexo, cor/raça e macrorregião.",
        "- `sensibilidade_ipw_escolaridade.csv`: comparação dos retornos da educação com V1028 e IPW, IC95% do IPW e variação em pontos percentuais.",
        "- `resumo_selecao_ocupacional.json`: amostra, sobreposição da propensão, desenho e limites de identificação.",
        "", "Referência de classificação do status de ocupação: documentação de variáveis derivadas da [PNAD Contínua do IBGE](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Anual/Outras_Formas_de_Trabalho_2018/Selecao_de_Variaveis.pdf).",
    ]
    (args.saida_dir / "relatorio_selecao_ocupacional.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"n_alvo": target["n_alvo"], "n_salario_valido": ipw.n,
                      "taxa_ocupacao_pct": summary["taxa_ocupacao_pct"],
                      "taxa_salario_valido_pct": summary["taxa_salario_valido_pct"],
                      "propensao": summary["propensao"], "n_upas": len(design.grupos),
                      "n_estratos": len(design.estratos),
                      "saida": str(args.saida_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
