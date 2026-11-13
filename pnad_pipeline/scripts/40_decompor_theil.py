#!/usr/bin/env python3
"""Decompõe o Theil T do salário-hora em parcelas dentro e entre macroregiões."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import pesos_amostrais
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao
from pnad_pipeline.analise.theil import decompor_theil

SPEC = importlib.util.spec_from_file_location("_m5_theil", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)
HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida, batch_size=100_000):
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    parciais = []
    for i, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        w = pesos_amostrais(df)
        salario = np.exp(df.y.to_numpy(dtype=float))
        df = df.assign(peso_v1028=w, soma_salario_ponderado=w * salario,
                       soma_salario_log_ponderado=w * salario * df.y.to_numpy())
        agregado = df.groupby(["Ano", "Trimestre", "Regiao"], observed=True).agg(
            n_salarios=("y", "size"), peso_v1028=("peso_v1028", "sum"),
            soma_salario_ponderado=("soma_salario_ponderado", "sum"),
            soma_salario_log_ponderado=("soma_salario_log_ponderado", "sum"))
        parciais.append(agregado)
        if i % 100 == 0:
            print(f"{i} lotes agregados para a decomposição", flush=True)
    if not parciais:
        raise ValueError("A amostra elegível ficou vazia.")
    totais = pd.concat(parciais).groupby(level=[0, 1, 2], observed=True).sum().reset_index().rename(columns={"Regiao": "regiao"})
    rows, series = [], []
    for (ano, tri), sub in totais.groupby(["Ano", "Trimestre"]):
        resumo, regionais = decompor_theil(sub.reset_index(drop=True))
        trimestre = f"{ano}T{tri}"
        series.append({"trimestre": trimestre, **resumo})
        rows.extend({"trimestre": trimestre, **resumo, **row} for row in regionais)
    fim = entrada.stat()
    if (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento.")
    relatorio = {
        "status": "ok", "versao": "decomposicao-theil-1.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{series[0]['trimestre']} a {series[-1]['trimestre']}", "ponderacao": "V1028",
        "n_observacoes": int(totais.n_salarios.sum()), "n_periodos": len(series),
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size}, "rows": rows, "series": series,
        "notas": [
            "A amostra e os pesos V1028 seguem a distribuição salarial histórica do M5. Os valores descrevem o salário-hora de ocupados de 25 a 65 anos, por trimestre.",
            "Theil total = parcela dentro das macroregiões + parcela entre macroregiões. A parcela dentro é a soma dos Theils regionais ponderados pela participação de cada região na soma ponderada dos salários-hora.",
            "A parcela entre compara as médias regionais à média nacional. A contribuição individual de uma região à parcela entre pode ser negativa. A parcela entre agregada é não negativa.",
            "Esta é uma decomposição descritiva. A participação na soma de salários-hora não representa a participação na renda domiciliar. As parcelas são aditivas e a tabela não apresenta intervalos de confiança.",
        ],
        "referencia": "https://pysal.org/inequality/user-guide/measure/theil.html",
    }
    exportar_comparacao(saida, "decomposicao_theil", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/decomposicao_theil")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    executar(args.entrada, args.saida_dir, args.batch_size)
