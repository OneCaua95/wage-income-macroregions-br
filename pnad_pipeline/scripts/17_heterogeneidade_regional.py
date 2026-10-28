#!/usr/bin/env python3
"""Reestima heterogeneidade regional do salário-hora com desenho da PNAD.

O modelo conjunto permite que os retornos categóricos da escolaridade e a
associação da formalidade variem por macrorregião. Também estima cinco modelos
regionais separados para comparar curvas e ajuste dentro da amostra. A
inferência usa V1028 e linearização por UPA/estrato; as associações não são
causais e o contraste de ajuste não é validação fora da amostra.
"""

from __future__ import annotations

import argparse
import copy
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

M15_SPEC = importlib.util.spec_from_file_location(
    "_pnad_regional_education_helpers", SCRIPT_DIR / "15_inferencia_retorno_educacao.py"
)
if M15_SPEC is None or M15_SPEC.loader is None:
    raise RuntimeError("Não foi possível carregar os auxiliares da inferência educacional.")
M15 = importlib.util.module_from_spec(M15_SPEC)
sys.modules[M15_SPEC.name] = M15
M15_SPEC.loader.exec_module(M15)

from pnad_pipeline.analise.amostragem import LinearizacaoUPA, exigir_colunas_desenho, pesos_amostrais

INPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet"
OUTPUT = PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/heterogeneidade_regional"
REGIONS = [label for _, label in M15.GRUPOS["regiao"]]
BASE_REGION = REGIONS[0]
OTHER_REGIONS = REGIONS[1:]
EDUCATION_TERMS = [
    f"Escolaridade_{level}_x_regiao_{region.replace('-', '_')}"
    for region in OTHER_REGIONS for level in range(1, 17)
]
FORMALITY_TERMS = [f"is_formal_x_regiao_{region.replace('-', '_')}" for region in OTHER_REGIONS]
POOLED_NAMES = [*M15.nomes_modelo("categorias", "regiao"), *FORMALITY_TERMS]
COMMON_NAMES = M15.nomes_modelo("categorias", "geral")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=INPUT)
    parser.add_argument("--saida-dir", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--nivel-confianca", type=float, default=0.95)
    parser.add_argument("--upa-unica", choices=["erro", "ajustar"], default="ajustar")
    parser.add_argument("--upa-multiestrato", choices=["erro", "separar"], default="separar")
    return parser.parse_args()


def matriz_pooled_regional(df: pd.DataFrame) -> np.ndarray:
    """Escolaridade e formalidade variam por macro; demais controles são comuns."""
    base = M15.matriz_modelo(df, "categorias", "regiao")
    formal = (
        df["Categoria_emprego"].isin([1, 3, 5, 7])
        | (df["Categoria_emprego"].isin([8, 9]) & df["Contribuicao_previdencia"].eq(1))
    ).to_numpy(dtype=np.float64)
    interacoes = [
        formal * df["Regiao"].eq(region).to_numpy(dtype=np.float64)
        for region in OTHER_REGIONS
    ]
    return np.column_stack([base, *interacoes])


def _modelo(especificacao: str, dimensao: str, nomes: list[str]):
    return M15.ModeloEducacao(especificacao, dimensao, nomes)


def _acumular_scores(model, x: np.ndarray, y: np.ndarray, w: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    assert model.beta is not None and model.desenho is not None
    residual = y - x @ model.beta
    model.desenho.acumular(df, x * (w * residual)[:, None])
    return residual


def _metrics(sse: float, stats_region: dict) -> tuple[float | None, float | None]:
    total_weight = stats_region["peso"]
    if total_weight <= 0:
        return None, None
    rmse = float(np.sqrt(max(sse, 0.0) / total_weight))
    sst = stats_region["soma_y2"] - stats_region["soma_y"] ** 2 / total_weight
    r2 = float(1.0 - sse / sst) if sst > 0 else None
    return rmse, r2


def _fit_comparison(region_stats: dict[str, dict], sse: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        stats_region = region_stats[region]
        common_rmse, common_r2 = _metrics(sse["pooled_comum"][region], stats_region)
        pooled_rmse, pooled_r2 = _metrics(sse["pooled_interacoes"][region], stats_region)
        separate_rmse, separate_r2 = _metrics(sse["regional_separado"][region], stats_region)
        rows.append({
            "regiao": region,
            "n_observacoes": stats_region["n"],
            "peso_v1028": stats_region["peso"],
            "n_efetivo_kish": stats_region["peso"] ** 2 / stats_region["peso2"],
            "rmse_log_pooled_comum": common_rmse,
            "r2_ponderado_pooled_comum": common_r2,
            "rmse_log_pooled_interacoes": pooled_rmse,
            "r2_ponderado_pooled_interacoes": pooled_r2,
            "rmse_log_separado": separate_rmse,
            "r2_ponderado_separado": separate_r2,
            "reducao_rmse_interacoes_pct": 100 * (common_rmse - pooled_rmse) / common_rmse if common_rmse else None,
            "reducao_rmse_separado_pct": 100 * (common_rmse - separate_rmse) / common_rmse if common_rmse else None,
            "n_upas_dominio": len(stats_region["upas_set"]),
            "n_estratos_dominio": len(stats_region["estratos_set"]),
        })
    return pd.DataFrame(rows)


def _education_contrast(model, level: int, region: str, marginal: bool) -> np.ndarray:
    """Categorial schooling contrast, including the pooled regional interaction."""
    contrast = np.zeros(len(model.nomes), dtype=np.float64)
    if level > 0:
        contrast[model.nomes.index(f"Escolaridade_{level}")] = 1.0
    if marginal and level > 1:
        contrast[model.nomes.index(f"Escolaridade_{level - 1}")] = -1.0
    if model.dimensao == "regiao" and region != BASE_REGION:
        suffix = region.replace("-", "_")
        if level > 0:
            contrast[model.nomes.index(f"Escolaridade_{level}_x_regiao_{suffix}")] = 1.0
        if marginal and level > 1:
            contrast[model.nomes.index(f"Escolaridade_{level - 1}_x_regiao_{suffix}")] = -1.0
    return contrast


def _educational_effects(models: dict[str, object], covariances: dict[str, np.ndarray], confidence: float) -> pd.DataFrame:
    rows = []
    alpha = 1.0 - confidence
    approach_models = [("Pooled com interações", "pooled_interacoes", None)]
    approach_models.extend(("Modelos regionais separados", f"regional_{region}", region) for region in REGIONS)
    for approach, model_key, fixed_region in approach_models:
        model = models[model_key]
        cov = covariances[model_key]
        assert model.beta is not None and model.ativos is not None and model.desenho is not None
        df = max(int(model.desenho.graus_liberdade), 1)
        critico = float(stats.t.ppf(1 - alpha / 2, df))
        regions = [fixed_region] if fixed_region else REGIONS
        for region in regions:
            for contrast in ("acumulado", "marginal"):
                marginal = contrast == "marginal"
                for level in range(1, 17):
                    c = _education_contrast(model, level, region, marginal=marginal)
                    needed = np.flatnonzero(np.abs(c) > 1e-12)
                    estimable = not np.any(~np.isin(needed, model.ativos))
                    beta = float(c @ model.beta) if estimable else np.nan
                    se = float(np.sqrt(max(float(c @ cov @ c), 0.0))) if estimable else np.nan
                    lower, upper = (beta - critico * se, beta + critico * se) if estimable else (np.nan, np.nan)
                    p_value = float(2 * stats.t.sf(abs(beta / se), df)) if estimable and se > 0 else np.nan
                    level_label = "16 anos ou mais" if level == 16 else ("1 ano" if level == 1 else f"{level} anos")
                    label = f"{level_label} vs {level - 1} anos" if marginal else f"{level_label} vs código 0"
                    rows.append({
                        "abordagem": approach,
                        "regiao": region,
                        "contraste": contrast,
                        "contraste_rotulo": label,
                        "anos_de_estudo_codigo": level,
                        "anos_de_estudo": level_label,
                        "referencia": f"Nível imediatamente anterior ({level - 1} anos)" if marginal else "Código 0: sem instrução ou menos de 1 ano",
                        "beta_log": beta if estimable else None,
                        "erro_padrao": se if estimable else None,
                        "ic_inferior_log": lower if estimable else None,
                        "ic_superior_log": upper if estimable else None,
                        "associacao_pct": float(100 * np.expm1(beta)) if estimable else None,
                        "ic_inferior_pct": float(100 * np.expm1(lower)) if estimable else None,
                        "ic_superior_pct": float(100 * np.expm1(upper)) if estimable else None,
                        "p_valor": p_value if np.isfinite(p_value) else None,
                        "graus_liberdade": df,
                        "identificavel": bool(estimable),
                    })
    out = pd.DataFrame(rows)
    out["p_valor_bh"] = np.nan
    for _, indices in out.groupby(["abordagem", "contraste"], sort=False).groups.items():
        index_list = list(indices)
        p = out.loc[index_list, "p_valor"].to_numpy(dtype=float)
        out.loc[index_list, "p_valor_bh"] = M15.M5.ajustar_bh(p)
    return out


def _formality_effects(models: dict[str, object], covariances: dict[str, np.ndarray], confidence: float) -> pd.DataFrame:
    rows = []
    alpha = 1.0 - confidence
    for approach, model_key, fixed_region in [
        ("Pooled com interações", "pooled_interacoes", None),
        *[("Modelos regionais separados", f"regional_{region}", region) for region in REGIONS],
    ]:
        model = models[model_key]
        cov = covariances[model_key]
        assert model.beta is not None and model.ativos is not None and model.desenho is not None
        df = max(int(model.desenho.graus_liberdade), 1)
        critical = float(stats.t.ppf(1 - alpha / 2, df))
        for region in ([fixed_region] if fixed_region else REGIONS):
            contrast = np.zeros(len(model.nomes), dtype=np.float64)
            contrast[model.nomes.index("is_formal")] = 1.0
            if fixed_region is None and region != BASE_REGION:
                name = f"is_formal_x_regiao_{region.replace('-', '_')}"
                contrast[model.nomes.index(name)] = 1.0
            needed = np.flatnonzero(np.abs(contrast) > 1e-12)
            estimable = not np.any(~np.isin(needed, model.ativos))
            beta = float(contrast @ model.beta) if estimable else np.nan
            se = float(np.sqrt(max(float(contrast @ cov @ contrast), 0.0))) if estimable else np.nan
            lower, upper = (beta - critical * se, beta + critical * se) if estimable else (np.nan, np.nan)
            p_value = float(2 * stats.t.sf(abs(beta / se), df)) if estimable and se > 0 else np.nan
            rows.append({
                "abordagem": approach, "regiao": region,
                "beta_log": beta if estimable else None,
                "erro_padrao": se if estimable else None,
                "ic_inferior_log": lower if estimable else None,
                "ic_superior_log": upper if estimable else None,
                "associacao_pct": float(100 * np.expm1(beta)) if estimable else None,
                "ic_inferior_pct": float(100 * np.expm1(lower)) if estimable else None,
                "ic_superior_pct": float(100 * np.expm1(upper)) if estimable else None,
                "p_valor": p_value if np.isfinite(p_value) else None,
                "graus_liberdade": df, "identificavel": bool(estimable),
                "interpretacao": "Diferença condicional entre formal e informal; associação, não efeito causal.",
            })
    out = pd.DataFrame(rows)
    out["p_valor_bh"] = np.nan
    for _, indices in out.groupby("abordagem", sort=False).groups.items():
        index_list = list(indices)
        p = out.loc[index_list, "p_valor"].to_numpy(dtype=float)
        out.loc[index_list, "p_valor_bh"] = M15.M5.ajustar_bh(p)
    return out


def executar(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or not 0 < args.nivel_confianca < 1:
        raise ValueError("Lote deve ser positivo e nível de confiança deve estar entre 0 e 1.")
    parquet = M15.M5.pq.ParquetFile(args.entrada)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") == b"false":
        raise ValueError("Use a base consolidada completa, anterior ao filtro de idade.")
    args.saida_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "pooled_comum": _modelo("categorias", "geral", COMMON_NAMES),
        "pooled_interacoes": _modelo("categorias", "regiao", POOLED_NAMES),
    }
    for region in REGIONS:
        models[f"regional_{region}"] = _modelo("categorias", "geral", COMMON_NAMES)
    design_all = LinearizacaoUPA(
        politica_upa_unica=args.upa_unica,
        politica_upa_multiestrato=args.upa_multiestrato,
    )

    print(f"Primeira passagem: modelos ponderados em lotes de {args.batch_size:,} linhas")
    for batch in M15.M5.iterar_lotes(args.entrada, args.batch_size):
        raw = batch.to_pandas()
        design_all.registrar(raw)
        df = M15.M5.normalizar_lote(batch)
        if not df.empty:
            y = df["y"].to_numpy(dtype=np.float64)
            w = pesos_amostrais(df)
            x = M15.matriz_modelo(df, "categorias", "geral")
            models["pooled_comum"].acumular_normal(x, y, w)
            del x
            x = matriz_pooled_regional(df)
            models["pooled_interacoes"].acumular_normal(x, y, w)
            del x
            for region in REGIONS:
                sample = df.loc[df["Regiao"].eq(region)]
                if sample.empty:
                    continue
                xr = M15.matriz_modelo(sample, "categorias", "geral")
                models[f"regional_{region}"].acumular_normal(
                    xr, sample["y"].to_numpy(dtype=np.float64), pesos_amostrais(sample)
                )
                del sample, xr
        del raw, df
        gc.collect()

    pooled_n = models["pooled_comum"].n
    if models["pooled_interacoes"].n != pooled_n or sum(models[f"regional_{r}"].n for r in REGIONS) != pooled_n:
        raise RuntimeError("As especificações não usaram a mesma amostra salarial.")

    for model in models.values():
        model.resolver()
        model.desenho = copy.deepcopy(design_all)
        model.desenho.iniciar_scores(len(model.nomes))
        print(f"{model.dimensao}: {model.n:,} observações; {len(model.ativos)} termos identificáveis")
    del design_all
    gc.collect()

    region_stats = {
        region: {"n": 0, "peso": 0.0, "peso2": 0.0, "soma_y": 0.0, "soma_y2": 0.0,
                 "upas_set": set(), "estratos_set": set()}
        for region in REGIONS
    }
    sse = {approach: {region: 0.0 for region in REGIONS} for approach in
           ("pooled_comum", "pooled_interacoes", "regional_separado")}

    print("Segunda passagem: scores por UPA/estrato e ajuste dentro da amostra")
    for batch in M15.M5.iterar_lotes(args.entrada, args.batch_size):
        df = M15.M5.normalizar_lote(batch)
        if df.empty:
            continue
        y = df["y"].to_numpy(dtype=np.float64)
        w = pesos_amostrais(df)
        common_model = models["pooled_comum"]
        common_x = M15.matriz_modelo(df, "categorias", "geral")
        common_resid = _acumular_scores(common_model, common_x, y, w, df)
        del common_x
        pooled_model = models["pooled_interacoes"]
        pooled_x = matriz_pooled_regional(df)
        pooled_resid = _acumular_scores(pooled_model, pooled_x, y, w, df)
        del pooled_x

        for region in REGIONS:
            mask = df["Regiao"].eq(region).to_numpy()
            if not mask.any():
                continue
            stats_region = region_stats[region]
            wr, yr = w[mask], y[mask]
            stats_region["n"] += int(mask.sum())
            stats_region["peso"] += float(wr.sum())
            stats_region["peso2"] += float(wr @ wr)
            stats_region["soma_y"] += float(wr @ yr)
            stats_region["soma_y2"] += float(wr @ (yr ** 2))
            sample = df.loc[mask]
            keys = sample[["UF", "Estrato", "UPA"]].drop_duplicates()
            stats_region["upas_set"].update(tuple(int(v) for v in row) for row in keys.itertuples(index=False, name=None))
            stats_region["estratos_set"].update(tuple(int(v) for v in row) for row in keys[["UF", "Estrato"]].drop_duplicates().itertuples(index=False, name=None))
            sse["pooled_comum"][region] += float(wr @ (common_resid[mask] ** 2))
            sse["pooled_interacoes"][region] += float(wr @ (pooled_resid[mask] ** 2))

            regional_model = models[f"regional_{region}"]
            regional_x = M15.matriz_modelo(sample, "categorias", "geral")
            regional_resid = _acumular_scores(
                regional_model, regional_x, sample["y"].to_numpy(dtype=np.float64),
                pesos_amostrais(sample), sample,
            )
            regional_w = pesos_amostrais(sample)
            sse["regional_separado"][region] += float(regional_w @ (regional_resid ** 2))
            del sample, keys, regional_x, regional_resid, regional_w
        del df, y, w, common_resid, pooled_resid
        gc.collect()

    covariances = {key: model.covariancia() for key, model in models.items()}
    tests_raw = {
        "heterogeneidade_escolaridade": M15._wald(
            models["pooled_interacoes"], M15.restricoes_indices(models["pooled_interacoes"], EDUCATION_TERMS),
            covariances["pooled_interacoes"],
        ),
        "heterogeneidade_formalidade": M15._wald(
            models["pooled_interacoes"], M15.restricoes_indices(models["pooled_interacoes"], FORMALITY_TERMS),
            covariances["pooled_interacoes"],
        ),
        "interacoes_conjuntas": M15._wald(
            models["pooled_interacoes"], M15.restricoes_indices(models["pooled_interacoes"], EDUCATION_TERMS + FORMALITY_TERMS),
            covariances["pooled_interacoes"],
        ),
    }
    primary_p = M15.M5.ajustar_bh(np.array([
        tests_raw["heterogeneidade_escolaridade"]["p_valor"],
        tests_raw["heterogeneidade_formalidade"]["p_valor"],
    ], dtype=float))
    tests = {
        "heterogeneidade_escolaridade": {**tests_raw["heterogeneidade_escolaridade"], "p_valor_bh": float(primary_p[0])},
        "heterogeneidade_formalidade": {**tests_raw["heterogeneidade_formalidade"], "p_valor_bh": float(primary_p[1])},
        "interacoes_conjuntas": {**tests_raw["interacoes_conjuntas"], "p_valor_bh": None},
    }
    for result in tests.values():
        p = result["p_valor"]
        result["p_valor_apresentacao"] = "< 1e-16" if p is not None and p < 1e-16 else (f"{p:.6g}" if p is not None else None)
        q = result["p_valor_bh"]
        result["p_valor_bh_apresentacao"] = "< 1e-16" if q is not None and q < 1e-16 else (f"{q:.6g}" if q is not None else None)
    education = _educational_effects(models, covariances, args.nivel_confianca)
    formal = _formality_effects(models, covariances, args.nivel_confianca)
    comparison = _fit_comparison(region_stats, sse)
    education.to_csv(args.saida_dir / "retornos_escolaridade_regiao.csv", index=False)
    formal.to_csv(args.saida_dir / "associacao_formalidade_regiao.csv", index=False)
    comparison.to_csv(args.saida_dir / "comparacao_pooled_separado.csv", index=False)

    model_summary = {}
    for key, model in models.items():
        model_summary[key] = {
            "n_observacoes": model.n,
            "k_regressores": len(model.nomes),
            "k_identificaveis": len(model.ativos),
            "n_upas_no_desenho_completo": len(model.desenho.grupos),
            "n_estratos_no_desenho_completo": len(model.desenho.estratos),
            "graus_liberdade_desenho": model.desenho.graus_liberdade,
            "peso_v1028": model.soma_pesos,
            "n_efetivo_kish": model.soma_pesos ** 2 / model.soma_pesos2,
        }
    region_summary = {}
    for region in REGIONS:
        s = region_stats[region]
        region_summary[region] = {
            "n_observacoes": s["n"], "peso_v1028": s["peso"],
            "n_efetivo_kish": s["peso"] ** 2 / s["peso2"],
            "n_upas_dominio": len(s["upas_set"]), "n_estratos_dominio": len(s["estratos_set"]),
        }
    summary = {
        "periodo": "2013–2025, 52 trimestres empilhados",
        "ponderacao": "V1028",
        "dominio": "Pessoas ocupadas de 25 a 65 anos, com rendimento e horas positivos, escolaridade e controles válidos; mesma amostra em todas as especificações.",
        "desenho": "Linearização por UPA dentro de estrato; aproximação com reposição, sem FPC ou recalibração. Cadastro completo das UPAs antes da seleção do domínio.",
        "politica_upa_unica": args.upa_unica,
        "politica_upa_multiestrato": args.upa_multiestrato,
        "upas_multiestrato": [{"UF": uf, "UPA": upa, "estratos": estratos}
                              for (uf, upa), estratos in sorted(models["pooled_interacoes"].desenho.upas_multiestrato.items())],
        "referencia_regional": BASE_REGION,
        "modelos": model_summary,
        "regioes": region_summary,
        "testes": tests,
        "ajuste_multiplo": "BH aplicado às duas hipóteses primárias (escolaridade × macrorregião e formalidade × macrorregião) e, separadamente, às estimativas por nível dentro de cada abordagem e tipo de contraste.",
        "comparacao_modelos": "RMSE log e R² ponderado são calculados dentro da amostra e por macroregião. Servem como comparação descritiva de ajuste; não são evidência de previsão fora da amostra nem teste de seleção entre especificações.",
        "limitacao_temporal": "Os 52 trimestres são empilhados; rotação de domicílios, mudanças de UPA/estrato e renovação gradual da Amostra Mestra limitam a variância temporal. A estimação não reproduz integralmente a variância oficial do IBGE.",
        "interpretacao": "As estimativas descrevem associações condicionais entre pessoas ocupadas; não identificam efeitos causais nem salários potenciais fora da ocupação.",
        "arquivos": {
            "retornos_escolaridade": "retornos_escolaridade_regiao.csv",
            "formalidade": "associacao_formalidade_regiao.csv",
            "comparacao_modelos": "comparacao_pooled_separado.csv",
        },
    }
    (args.saida_dir / "resumo_heterogeneidade_regional.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Heterogeneidade regional do salário-hora", "",
        "## Especificações e amostra", "",
        "A análise usa a mesma amostra salarial em todas as especificações, ponderada por V1028. O desfecho é o log do salário-hora real. Os controles incluem formalidade, setor público, sexo, setor, ocupação, cor/raça, idade e idade², UF, ano e trimestre.", "",
        f"O modelo conjunto usa {pooled_n:,} observações. Escolaridade categórica e associação da formalidade podem variar por macroregião; Norte é a referência para os termos de interação. Também são estimados cinco modelos regionais separados, com os mesmos controles.", "",
        "## Testes de heterogeneidade", "",
        "Os testes Wald-F avaliam conjuntamente se os coeficientes de escolaridade ou formalidade diferem entre macrorregiões, usando a covariância por UPA dentro de estrato. Os p-valores dos dois testes primários recebem ajuste Benjamini-Hochberg.", "",
        "| Hipótese | F | gl numerador | gl denominador | p-valor | p ajustado BH |", "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("heterogeneidade_escolaridade", "Escolaridade × macrorregião"),
                       ("heterogeneidade_formalidade", "Formalidade × macrorregião"),
                       ("interacoes_conjuntas", "Todas as interações em conjunto")):
        t = tests[key]
        adjusted = t["p_valor_bh_apresentacao"] or "—"
        report.append(f"| {label} | {t['f'] if t['f'] is not None else '—'} | {t['gl_numerador']} | {t['gl_denominador']} | {t['p_valor_apresentacao'] or '—'} | {adjusted} |")
    report.extend([
        "", "## Modelos conjuntos e separados", "",
        "A tabela de ajuste compara, em cada macroregião, o modelo pooled com coeficientes comuns, o pooled com interações para escolaridade e formalidade, e a regressão regional estimada separadamente. RMSE log e R² são ponderados e calculados na própria amostra de estimação; descrevem ajuste e não substituem validação fora da amostra.", "",
        "Os arquivos de efeitos incluem associações acumuladas contra o código 0 e diferenças adjacentes por escolaridade, além da associação condicional da formalidade. Os intervalos de 95% são pontuais e transformados da escala log. Os testes não convertem associações em efeitos causais.", "",
        "## Limitações", "",
        "A variância usa linearização por UPA dentro de estrato, aproximação com reposição, sem FPC ou recalibração. O desenho completo é registrado antes da seleção do domínio; a política para UPA única é ajustar, e versões de UPA reclassificadas entre estratos são separadas. O empilhamento temporal não estima toda a covariância de domicílios revisitados nem reproduz integralmente a variância oficial do IBGE.", "",
        "## Arquivos", "",
        "- `retornos_escolaridade_regiao.csv`: estimativas acumuladas e marginais por abordagem e macroregião.",
        "- `associacao_formalidade_regiao.csv`: associação da formalidade por abordagem e macroregião.",
        "- `comparacao_pooled_separado.csv`: qualidade de ajuste ponderada dentro da amostra.",
        "- `resumo_heterogeneidade_regional.json`: amostra, desenho, modelos e testes.",
    ])
    (args.saida_dir / "relatorio_heterogeneidade_regional.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"n": pooled_n, "testes": tests, "saida": str(args.saida_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
