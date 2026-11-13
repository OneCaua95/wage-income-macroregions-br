#!/usr/bin/env python3
"""Compara o M5 original com winsorização ponderada do salário-hora em P1/P99."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pnad_pipeline.analise.amostragem import pesos_amostrais, quantis_ponderados
from pnad_pipeline.analise.regressao.robustez import contrastes_m5, tabela_comparacao

SPEC = importlib.util.spec_from_file_location("_pnad_m5_robustez", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = PROJECT_ROOT / "data/processed/recoleta_ponderada"
VERSAO = "robustez-m5-1.0"


def executar(entrada, saida, batch_size=100_000):
    entrada, saida = Path(entrada), Path(saida)
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    modelo = M5.ResultadoModelo("geral", M5.UFS)
    desenho = modelo.desenho
    desenho.politica_upa_unica = "ajustar"
    desenho.politica_upa_multiestrato = "separar"
    nomes = ["Intercept", *M5.nomes_features(M5.UFS)]
    periodos = set()
    lotes_quantis = []

    print("Passada 1/3: M5 original, cadastro do desenho completo e quantis ponderados", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        desenho.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        y = df["y"].to_numpy(dtype=float)
        w = pesos_amostrais(df)
        modelo.acumular_normal(M5.matriz_m5(df, M5.UFS), y, w)
        lotes_quantis.append(pd.DataFrame({"y": y, "V1028": w}))
        if lote % 40 == 0:
            print(f"  {lote} lotes · {modelo.n:,} registros elegíveis", flush=True)

    if not modelo.n:
        raise ValueError("A amostra elegível do M5 ficou vazia.")
    modelo.resolver()
    quantis = pd.concat(lotes_quantis, ignore_index=True)
    lotes_quantis.clear()
    p1, p99 = quantis_ponderados(quantis, "y", [.01, .99])
    del quantis
    if not np.isfinite([p1, p99]).all() or p1 >= p99:
        raise ValueError("Os limites ponderados do salário-hora precisam ser finitos e distintos.")

    print(f"Passada 2/3: limitando o salário-hora a R$ {np.exp(p1):.2f}–{np.exp(p99):.2f}", flush=True)
    delta_xty = np.zeros_like(modelo.xty)
    n_inferior = n_superior = n_comum = 0
    peso_limitado = 0.
    for batch in M5.iterar_lotes(entrada, batch_size):
        df = M5.normalizar_lote(batch)
        y = df["y"].to_numpy(dtype=float)
        abaixo, acima = y < p1, y > p99
        n_inferior += int(abaixo.sum())
        n_superior += int(acima.sum())
        n_comum += len(df)
        mudou = abaixo | acima
        if mudou.any():
            parte = df.loc[mudou]
            w = pesos_amostrais(parte)
            delta = np.clip(y[mudou], p1, p99) - y[mudou]
            delta_xty += M5.matriz_m5(parte, M5.UFS).T @ (w * delta)
            peso_limitado += float(w.sum())

    # A matriz de regressoras e os pesos são comuns às duas versões.
    beta_limitado = modelo.beta + modelo.bread @ delta_xty
    rotulos, contrastes, anos = contrastes_m5(nomes, modelo.xtx)
    ativos = set(modelo.indices_ativos)
    if any(index not in ativos for index in np.flatnonzero(np.any(contrastes != 0, axis=0))):
        raise ValueError("A base não identifica todos os contrastes do M5.")
    projecao = modelo.bread @ contrastes.T
    dimensao = 2 * len(rotulos)
    desenho.iniciar_scores(dimensao)

    print("Passada 3/3: intervalos pelo desenho amostral e diferença pareada", flush=True)
    n_scores = 0
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        n_scores += len(df)
        x = M5.matriz_m5(df, M5.UFS)
        w = pesos_amostrais(df)
        y = df["y"].to_numpy(dtype=float)
        z = x @ projecao
        residuo_original = w * (y - x @ modelo.beta)
        residuo_limitado = w * (np.clip(y, p1, p99) - x @ beta_limitado)
        scores = np.column_stack((z * residuo_original[:, None], z * residuo_limitado[:, None]))
        desenho.acumular(df, scores)
        if lote % 40 == 0:
            print(f"  {lote} lotes · {n_scores:,} registros com scores", flush=True)

    fim = entrada.stat()
    if n_comum != modelo.n or n_scores != modelo.n or (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise na mesma amostra.")
    cov = desenho.covariancia(np.eye(dimensao))
    rows = tabela_comparacao(rotulos, contrastes @ modelo.beta, contrastes @ beta_limitado, cov,
                            t.ppf(.975, desenho.graus_liberdade))
    inicio, ultimo = min(periodos), max(periodos)
    periodo = f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}"
    relatorio = {
        "status": "ok", "versao": VERSAO, "periodo": periodo,
        "processado_em": datetime.now(timezone.utc).isoformat(),
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "n_periodos": len(periodos), "n_observacoes": modelo.n,
        "soma_pesos": modelo.soma_pesos, "ponderacao": "V1028", "data_base": "Janeiro/2026",
        "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
        "graus_liberdade": desenho.graus_liberdade, "nivel_confianca": .95,
        "upa_unica": desenho.politica_upa_unica, "upa_multiestrato": desenho.politica_upa_multiestrato,
        "p1_salario_hora": float(np.exp(p1)), "p99_salario_hora": float(np.exp(p99)),
        "n_limitados_abaixo": n_inferior, "n_limitados_acima": n_superior,
        "fracao_ponderada_limitada": peso_limitado / modelo.soma_pesos,
        "media_ponderada_anos_estudo": anos, "rows": rows,
        "notas": [
            "Os mesmos registros elegíveis, pesos V1028 e controles do M5 entram nas duas versões. A winsorização limita o salário-hora real a P1 e P99 ponderados antes do ajuste em log.",
            "Os contrastes usam a distribuição ponderada comum de covariáveis. Formalidade e sexo são avaliados na escolaridade média. Escolaridade compara os códigos 12 e 8.",
            "IC95% pontuais por linearização de UPA dentro de estrato. Os limites de winsorização e as médias usadas nos contrastes são fixos no cálculo da incerteza. A mudança em pontos percentuais usa o método delta com covariância pareada.",
            "A amostra empilhada conta registros pessoa-trimestre. Os contrastes descrevem associações condicionais entre ocupados de 25 a 65 anos.",
        ],
    }
    saida.mkdir(parents=True, exist_ok=True)
    csv = saida / "robustez_m5.csv"
    resumo = saida / "resumo_robustez_m5.json"
    temporario_csv = csv.with_suffix(f".{os.getpid()}.tmp")
    temporario_json = resumo.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(temporario_csv, index=False)
    temporario_json.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporario_csv, csv)
    os.replace(temporario_json, resumo)
    print(f"Comparação concluída em {saida}", flush=True)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/robustez_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
