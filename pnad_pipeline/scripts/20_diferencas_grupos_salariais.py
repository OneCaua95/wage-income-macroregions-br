#!/usr/bin/env python3
"""Compara diferenças salariais brutas e ajustadas por sexo e cor/raça.

As duas medidas usam a amostra elegível do M5 histórico e o peso final V1028.
O contraste bruto é a diferença entre médias ponderadas de log(salário-hora).
O contraste ajustado é uma predição padronizada do M5 na distribuição comum
de escolaridade, com variância linearizada por UPA dentro de estrato.

Os resultados são associações descritivas/condicionais, não efeitos causais.
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

M5_SPEC = importlib.util.spec_from_file_location("_pnad_m5_group_gap_helpers", SCRIPT_DIR / "08_inferencia_m5.py")
if M5_SPEC is None or M5_SPEC.loader is None:
    raise RuntimeError("Não foi possível carregar a especificação M5.")
M5 = importlib.util.module_from_spec(M5_SPEC)
sys.modules[M5_SPEC.name] = M5
M5_SPEC.loader.exec_module(M5)

from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais


CONTRASTS = [
    {"dimensao": "sexo", "codigo": 2, "grupo": "Mulheres", "referencia_codigo": 1, "referencia": "Homens", "rotulo": "Mulheres vs homens"},
    {"dimensao": "cor", "codigo": 2, "grupo": "Preta", "referencia_codigo": 1, "referencia": "Branca", "rotulo": "Preta vs branca"},
    {"dimensao": "cor", "codigo": 3, "grupo": "Amarela", "referencia_codigo": 1, "referencia": "Branca", "rotulo": "Amarela vs branca"},
    {"dimensao": "cor", "codigo": 4, "grupo": "Parda", "referencia_codigo": 1, "referencia": "Branca", "rotulo": "Parda vs branca"},
    {"dimensao": "cor", "codigo": 5, "grupo": "Indígena", "referencia_codigo": 1, "referencia": "Branca", "rotulo": "Indígena vs branca"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/diferencas_grupos")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--nivel-confianca", type=float, default=0.95)
    parser.add_argument("--upa-unica", choices=["erro", "ajustar"], default="ajustar")
    parser.add_argument("--upa-multiestrato", choices=["erro", "separar"], default="separar")
    return parser.parse_args()


def indicador(df: pd.DataFrame, dimensao: str, codigo: int) -> np.ndarray:
    coluna = "Sexo" if dimensao == "sexo" else "Cor"
    return df[coluna].eq(codigo).to_numpy(dtype=bool)


def percentual(delta_log: float) -> float:
    return float(100.0 * np.expm1(delta_log))


def intervalo_percentual(delta_log: float, erro_padrao: float, critico: float) -> tuple[float, float]:
    return percentual(delta_log - critico * erro_padrao), percentual(delta_log + critico * erro_padrao)


def contrastes_ajustados(modelo, covariancia: np.ndarray, media_escolaridade: float, graus_liberdade: int, nivel: float) -> list[dict]:
    assert modelo.beta is not None
    nomes = ["Intercept", *M5.nomes_features(modelo.ufs)]
    indices = {nome: i for i, nome in enumerate(nomes)}
    critico = float(stats.t.ppf(1 - (1 - nivel) / 2, max(graus_liberdade, 1)))
    linhas = []
    for item in CONTRASTS:
        c = np.zeros(len(nomes), dtype=np.float64)
        if item["dimensao"] == "sexo":
            c[indices["Sexo_Dummy"]] = 1.0
            c[indices["Estudo_x_Sexo_Dummy"]] = media_escolaridade
        else:
            c[indices[M5.CORES[item["codigo"]]]] = 1.0
        delta = float(c @ modelo.beta)
        variancia = float(c @ covariancia @ c)
        se = float(np.sqrt(max(variancia, 0.0)))
        low, high = intervalo_percentual(delta, se, critico)
        linhas.append({
            "dimensao": item["dimensao"], "grupo": item["grupo"], "referencia": item["referencia"],
            "rotulo": item["rotulo"], "ajustada_delta_log": delta,
            "ajustada_erro_padrao_log": se, "ajustada_pct": percentual(delta),
            "ajustada_ic95_inferior_pct": low, "ajustada_ic95_superior_pct": high,
        })
    return linhas


def executar(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or not 0 < args.nivel_confianca < 1:
        raise ValueError("O lote deve ser positivo e o nível de confiança deve estar entre 0 e 1.")
    if not args.entrada.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.entrada}")

    parquet = M5.pq.ParquetFile(args.entrada)
    desenho_base = LinearizacaoUPA(
        politica_upa_unica=args.upa_unica,
        politica_upa_multiestrato=args.upa_multiestrato,
    )
    modelo = M5.ResultadoModelo("geral", M5.UFS)
    grupos = {(item["dimensao"], item["codigo"]): {"n": 0, "peso": 0.0, "soma_log": 0.0} for item in CONTRASTS}
    grupos.update({("sexo", 1): {"n": 0, "peso": 0.0, "soma_log": 0.0}, ("cor", 1): {"n": 0, "peso": 0.0, "soma_log": 0.0}})
    soma_peso = 0.0
    soma_peso_escolaridade = 0.0
    periodos: set[tuple[int, int]] = set()

    print("Primeira passada: ajustando o M5 e acumulando as médias brutas", flush=True)
    for lote_num, batch in enumerate(M5.iterar_lotes(args.entrada, args.batch_size), 1):
        desenho_base.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        if not df.empty:
            y = df["y"].to_numpy(dtype=np.float64)
            w = pesos_amostrais(df)
            x = M5.matriz_m5(df, M5.UFS)
            modelo.acumular_normal(x, y, w)
            soma_peso += float(w.sum())
            soma_peso_escolaridade += float(w @ df["Anos_de_Estudo"].to_numpy(dtype=np.float64))
            periodos.update((int(a), int(t)) for a, t in df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
            for chave, stats_grupo in grupos.items():
                dim, codigo = chave
                mask = indicador(df, dim, codigo)
                if mask.any():
                    stats_grupo["n"] += int(mask.sum())
                    stats_grupo["peso"] += float(w[mask].sum())
                    stats_grupo["soma_log"] += float(w[mask] @ y[mask])
            del x, y, w
        del df
        if lote_num % 10 == 0:
            print(f"  {lote_num:,} lotes percorridos", flush=True)

    if not modelo.n or soma_peso <= 0:
        raise ValueError("A amostra comum do M5 ficou vazia.")
    for chave, stats_grupo in grupos.items():
        if stats_grupo["peso"] <= 0:
            raise ValueError(f"Grupo sem peso válido na amostra comum: {chave}.")
        stats_grupo["media_log"] = stats_grupo["soma_log"] / stats_grupo["peso"]

    media_escolaridade = soma_peso_escolaridade / soma_peso
    modelo.resolver()
    modelo.desenho = copy.deepcopy(desenho_base)
    modelo.desenho.iniciar_scores(modelo.k)
    desenho_bruto = copy.deepcopy(desenho_base)
    desenho_bruto.iniciar_scores(len(CONTRASTS))
    print(f"M5: {modelo.n:,} observações; {len(modelo.indices_ativos)} parâmetros identificáveis", flush=True)
    print("Segunda passada: variâncias por UPA dentro de estrato", flush=True)
    for lote_num, batch in enumerate(M5.iterar_lotes(args.entrada, args.batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        y = df["y"].to_numpy(dtype=np.float64)
        w = pesos_amostrais(df)
        x = M5.matriz_m5(df, M5.UFS)
        residuo = y - x @ modelo.beta
        modelo.desenho.acumular(df, x * (w * residuo)[:, None])

        scores_brutos = np.zeros((len(df), len(CONTRASTS)), dtype=np.float64)
        for j, item in enumerate(CONTRASTS):
            dim = item["dimensao"]
            grupo_mask = indicador(df, dim, item["codigo"])
            referencia_mask = indicador(df, dim, item["referencia_codigo"])
            g = grupos[(dim, item["codigo"])]
            ref = grupos[(dim, item["referencia_codigo"])]
            scores_brutos[:, j] = w * (
                grupo_mask * (y - g["media_log"]) / g["peso"]
                - referencia_mask * (y - ref["media_log"]) / ref["peso"]
            )
        desenho_bruto.acumular(df, scores_brutos)
        del df, y, w, x, residuo, scores_brutos
        if lote_num % 10 == 0:
            print(f"  {lote_num:,} lotes percorridos", flush=True)
        if lote_num % 20 == 0:
            gc.collect()

    cov_ajustada = modelo.desenho.covariancia(modelo.bread)
    cov_bruta = desenho_bruto.covariancia(np.eye(len(CONTRASTS)))
    gl = int(desenho_base.graus_liberdade)
    critico = float(stats.t.ppf(1 - (1 - args.nivel_confianca) / 2, max(gl, 1)))
    ajustadas = contrastes_ajustados(modelo, cov_ajustada, media_escolaridade, gl, args.nivel_confianca)
    linhas = []
    for j, item in enumerate(CONTRASTS):
        g = grupos[(item["dimensao"], item["codigo"])]
        ref = grupos[(item["dimensao"], item["referencia_codigo"])]
        delta = float(g["media_log"] - ref["media_log"])
        se = float(np.sqrt(max(float(cov_bruta[j, j]), 0.0)))
        low, high = intervalo_percentual(delta, se, critico)
        adjusted = ajustadas[j]
        linhas.append({
            **adjusted,
            "bruta_delta_log": delta, "bruta_erro_padrao_log": se,
            "bruta_pct": percentual(delta), "bruta_ic95_inferior_pct": low,
            "bruta_ic95_superior_pct": high,
            "n_grupo": g["n"], "n_referencia": ref["n"],
            "peso_grupo_v1028": g["peso"], "peso_referencia_v1028": ref["peso"],
            "salario_hora_geometrico_bruto_grupo": float(np.exp(g["media_log"])),
            "salario_hora_geometrico_bruto_referencia": float(np.exp(ref["media_log"])),
        })

    args.saida_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.saida_dir / "diferencas_brutas_ajustadas.csv"
    pd.DataFrame(linhas).to_csv(output_csv, index=False, float_format="%.12g")
    resumo = {
        "status": "ok", "periodo": "2013–2025 (empilhado)", "n_periodos": len(periodos),
        "primeiro_periodo": f"{min(periodos)[0]}T{min(periodos)[1]}",
        "ultimo_periodo": f"{max(periodos)[0]}T{max(periodos)[1]}",
        "n_observacoes_elegiveis": int(modelo.n), "peso_v1028_total": soma_peso,
        "media_ponderada_anos_estudo": media_escolaridade,
        "n_upas_desenho": len(desenho_base.grupos), "n_estratos_desenho": len(desenho_base.estratos),
        "graus_liberdade_desenho": gl, "nivel_confianca": args.nivel_confianca,
        "amostra": "Amostra comum do M5 histórico: pessoas ocupadas de 25 a 65 anos, salário e horas positivos, escolaridade, sexo, cor/raça e covariáveis válidos.",
        "desfecho": "Log do salário-hora real, em reais de janeiro/2026.",
        "ponderacao": "Peso final V1028.",
        "contraste_bruto": "Diferença de médias ponderadas de log(salário-hora) entre grupos; percentual = 100 × (exp(diferença) − 1).",
        "contraste_ajustado": "Contraste padronizado do M5 na distribuição V1028 comum; para sexo, interação com anos de estudo avaliada na escolaridade média ponderada da amostra; para cor/raça, contraste contra branca.",
        "ajuste": "M5 log-linear com idade e idade², escolaridade, formalidade, setor público, sexo, setores, ocupações, cor/raça, UF, ano, trimestre e interações de escolaridade com formalidade, setor público e sexo.",
        "variancia": "Linearização por UPA dentro de estrato, aproximação com reposição, sem FPC ou recalibração de pesos. UPA recorrente permanece agrupada por estrato; UPA única ajustada.",
        "interpretacao": "Diferenças brutas e ajustadas são comparações associativas de pessoas ocupadas na amostra salarial; não são efeitos causais nem medidas automáticas de discriminação.",
        "arquivo_resultados": output_csv.name,
        "linhas": linhas,
    }
    (args.saida_dir / "resumo_diferencas_grupos.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Diferenças salariais brutas e ajustadas por grupo", "",
        f"A amostra comum do M5 soma {modelo.n:,} observações pessoa-trimestre em {len(periodos)} trimestres. O peso final é V1028 e os valores são reais de janeiro/2026.", "",
        "As diferenças brutas comparam médias ponderadas do log do salário-hora. As diferenças ajustadas são contrastes padronizados do M5 na distribuição comum; a diferença por sexo é avaliada na escolaridade média ponderada. Percentuais são 100 × (exp(diferença em log) − 1).", "",
        "| Comparação | Bruta (%) | IC95% bruto | Ajustada (%) | IC95% ajustado | n grupo / referência |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in linhas:
        report.append(f"| {row['rotulo']} | {row['bruta_pct']:.2f} | {row['bruta_ic95_inferior_pct']:.2f} a {row['bruta_ic95_superior_pct']:.2f} | {row['ajustada_pct']:.2f} | {row['ajustada_ic95_inferior_pct']:.2f} a {row['ajustada_ic95_superior_pct']:.2f} | {row['n_grupo']:,} / {row['n_referencia']:,} |")
    report += ["", resumo["variancia"], "", resumo["interpretacao"], ""]
    (args.saida_dir / "relatorio_diferencas_grupos.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Concluído: {len(linhas)} contrastes; saída em {args.saida_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
