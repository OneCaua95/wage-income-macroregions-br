#!/usr/bin/env python3
"""Compara a dispersão dos resíduos nas pontas e no centro dos ajustes do M5."""

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
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais, quantis_ponderados
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao
from pnad_pipeline.analise.regressao.dispersao import contrastar_desvios, linearizar_desvios

SPEC = importlib.util.spec_from_file_location("_pnad_m5_dispersao", SCRIPT_DIR / "08_inferencia_m5.py")
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
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
    periodos = set()
    print("Passada 1/3: ajustando o M5 e cadastrando o desenho amostral", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        desenho.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        modelo.acumular_normal(M5.matriz_m5(df, M5.UFS), df["y"].to_numpy(), pesos_amostrais(df))
        if lote % 100 == 0:
            print(f"  {lote} lotes · {modelo.n:,} registros elegíveis", flush=True)
    if not modelo.n:
        raise ValueError("A amostra elegível do M5 ficou vazia.")
    modelo.resolver()

    ajustes, pesos = np.empty((2, modelo.n))
    n_ajustes = 0
    print("Passada 2/3: calculando os cortes ponderados do salário-hora ajustado", flush=True)
    for batch in M5.iterar_lotes(entrada, batch_size):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        fim = n_ajustes + len(df)
        if fim > modelo.n:
            raise ValueError("A base mudou durante o processamento.")
        ajustes[n_ajustes:fim] = M5.matriz_m5(df, M5.UFS) @ modelo.beta
        pesos[n_ajustes:fim] = pesos_amostrais(df)
        n_ajustes = fim
    if n_ajustes != modelo.n:
        raise ValueError("A base mudou durante o processamento.")
    p10, p40, p60, p90 = quantis_ponderados(
        pd.DataFrame({"ajuste": ajustes, "V1028": pesos}, copy=False), "ajuste", [.1, .4, .6, .9],
    )
    del ajustes, pesos

    k = modelo.k
    desenho.iniciar_scores(k + 9)
    soma_x, soma_rx = np.zeros((2, 3, k))
    contagens = np.zeros(3, dtype=np.int64)
    n_scores = 0
    print("Passada 3/3: linearizando as diferenças por UPA e estrato", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        x = M5.matriz_m5(df, M5.UFS)
        pesos = pesos_amostrais(df)
        ajuste = x @ modelo.beta
        residuo = df["y"].to_numpy() - ajuste
        faixas = np.column_stack((ajuste <= p10, (ajuste > p40) & (ajuste <= p60), ajuste > p90))
        contagens += faixas.sum(axis=0)
        pesos_grupo = pesos[:, None] * faixas
        residuos_grupo = pesos_grupo * residuo[:, None]
        soma_x += pesos_grupo.T @ x
        soma_rx += residuos_grupo.T @ x
        momentos = np.stack((pesos_grupo, residuos_grupo, residuos_grupo * residuo[:, None]), axis=2)
        score_beta = (pesos * residuo)[:, None] * x
        desenho.acumular(df, np.column_stack((score_beta, momentos.reshape(len(df), 9))))
        n_scores += len(df)
        if lote % 100 == 0:
            print(f"  {lote} lotes · {n_scores:,} contribuições", flush=True)

    fim = entrada.stat()
    if n_scores != modelo.n or (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")
    desvios, scores_dp = linearizar_desvios(
        desenho.scores[:, k:].reshape(-1, 3, 3), soma_x, soma_rx, desenho.scores[:, :k], modelo.bread,
    )
    desenho.scores = scores_dp
    cov = desenho.covariancia(np.eye(3))
    linhas = contrastar_desvios(desvios, cov, desenho.graus_liberdade)
    for linha, i in zip(linhas, (0, 2)):
        linha["n_registros_grupo"] = int(contagens[i])
        linha["n_registros_referencia"] = int(contagens[1])
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "dispersao-m5-1.0", "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}",
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "n_periodos": len(periodos), "n_observacoes": modelo.n, "soma_pesos": modelo.soma_pesos,
        "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
        "graus_liberdade": desenho.graus_liberdade, "nivel_confianca": .95,
        "cortes_rs_h": dict(zip(("p10", "p40", "p60", "p90"), map(float, np.exp([p10, p40, p60, p90])))),
        "rows": linhas,
        "notas": [
            "O desvio-padrão usa resíduos na escala logarítmica. A referência D5 e D6 é calculada com os registros dos dois decis reunidos e seus pesos V1028.",
            "Os IC95% pontuais usam a covariância conjunta por UPA dentro de estrato e incluem a incerteza dos coeficientes do M5. A atribuição dos registros às faixas é tratada como fixa na linearização.",
            "As duas comparações descrevem a dispersão na amostra de ajuste. As faixas seguem os cortes nacionais do salário-hora estimado.",
        ],
    }
    exportar_comparacao(saida, "dispersao_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/dispersao_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
