#!/usr/bin/env python3
"""Diagnostica resíduos do M5 por decil estimado e macrorregião."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import pesos_amostrais, quantis_ponderados
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao
from pnad_pipeline.analise.regressao.residuos import resumir_residuos

SPEC = importlib.util.spec_from_file_location("_pnad_m5_residuos", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida, batch_size=100_000):
    entrada = Path(entrada)
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    modelo = M5.ResultadoModelo("geral", M5.UFS)
    periodos = set()
    print("Passada 1/2: ajustando o M5 completo", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        modelo.acumular_normal(M5.matriz_m5(df, M5.UFS), df["y"].to_numpy(), pesos_amostrais(df))
        if lote % 80 == 0:
            print(f"  {lote} lotes · {modelo.n:,} registros elegíveis", flush=True)
    if not modelo.n:
        raise ValueError("A amostra elegível do M5 ficou vazia.")
    modelo.resolver()

    # Guarda somente as colunas necessárias para os decis exatos e os resumos.
    ajustes, residuos, pesos = np.empty((3, modelo.n))
    regioes = np.empty(modelo.n, dtype=np.int8)
    posicao = 0
    print("Passada 2/2: calculando valores ajustados e resíduos", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        proxima = posicao + len(df)
        if proxima > modelo.n:
            raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")
        trecho = slice(posicao, proxima)
        ajustes[trecho] = M5.matriz_m5(df, M5.UFS) @ modelo.beta
        residuos[trecho] = df["y"].to_numpy() - ajustes[trecho]
        pesos[trecho] = pesos_amostrais(df)
        regioes[trecho] = pd.Categorical(df["Regiao"], categories=M5.MACROREGIOES).codes
        posicao = proxima
        if lote % 80 == 0:
            print(f"  {lote} lotes · {posicao:,} resíduos calculados", flush=True)

    fim = entrada.stat()
    if posicao != modelo.n or (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")

    cortes = quantis_ponderados(pd.DataFrame({"ajuste": ajustes, "V1028": pesos}, copy=False), "ajuste", np.arange(1, 10) / 10)
    decis = np.searchsorted(cortes, ajustes, side="left")
    limites = [None, *map(float, np.exp(cortes)), None]
    por_decil = resumir_residuos(decis, residuos, pesos, 10)
    por_regiao = resumir_residuos(regioes, residuos, pesos, len(M5.MACROREGIOES))
    linhas = [
        {"dimensao": "decil", "grupo": f"D{i + 1}",
         "limite_inferior_rs_h": limites[i], "limite_superior_rs_h": limites[i + 1], **linha}
        for i, linha in enumerate(por_decil)
    ] + [
        {"dimensao": "macrorregiao", "grupo": nome,
         "limite_inferior_rs_h": None, "limite_superior_rs_h": None, **linha}
        for nome, linha in zip(M5.MACROREGIOES, por_regiao)
    ]
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "residuos-m5-1.0", "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}",
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "n_periodos": len(periodos), "n_observacoes": modelo.n, "soma_pesos": modelo.soma_pesos,
        "rows": linhas,
        "notas": [
            "Resíduo é o log do salário-hora observado menos o log ajustado. Valores positivos indicam que o M5 estimou abaixo do observado em média. A média e o desvio-padrão usam os pesos V1028 na própria amostra de ajuste.",
            "Os decis nacionais usam o salário-hora estimado pelo M5. Os limites exibem a transformação exponencial do ajuste em reais de janeiro de 2026. Empates nos cortes ficam no mesmo grupo.",
            "Os efeitos fixos de UF fazem a média ponderada do resíduo ficar próxima de zero em cada macrorregião por construção. A dispersão pode variar entre regiões e faixas.",
        ],
    }
    exportar_comparacao(saida, "residuos_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/residuos_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
