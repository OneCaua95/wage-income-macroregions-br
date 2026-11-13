#!/usr/bin/env python3
"""Compara o M5 em nível e em log numa avaliação temporal ponderada."""

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
from pnad_pipeline.analise.regressao.previsao_m5 import metricas_ponderadas

SPEC = importlib.util.spec_from_file_location("_pnad_m5_escala", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"
PRIMEIRO_ANO, ULTIMO_TREINO, CALIBRACAO, TESTE = 2013, 2023, 2024, 2025
MANTER = [not nome.startswith("Ano_") for nome in ["Intercept", *M5.nomes_features(M5.UFS)]]


def matriz_temporal(df):
    tendencia = df["Ano"].to_numpy(dtype=float) - PRIMEIRO_ANO
    return np.column_stack((M5.matriz_m5(df, M5.UFS)[:, MANTER], tendencia))


def lotes_temporais(entrada, batch_size, inicio, fim):
    for batch in M5.iterar_lotes(entrada, batch_size):
        ano = pd.to_numeric(batch.column(batch.schema.get_field_index("Ano")).to_pandas(), errors="coerce")
        incluir = ano.between(inicio, fim).to_numpy(dtype=bool, na_value=False)
        if incluir.any():
            df = M5.normalizar_lote(batch.filter(incluir))
            if not df.empty:
                yield df


def conferir_periodos(periodos, inicio, fim):
    esperados = {(ano, tri) for ano in range(inicio, fim + 1) for tri in range(1, 5)}
    if periodos != esperados:
        raise ValueError(f"A janela {inicio}–{fim} deve conter os quatro trimestres elegíveis de cada ano.")


def executar(entrada, saida, batch_size=100_000):
    entrada = Path(entrada)
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    modelo = M5.ResultadoModelo("M5 temporal", M5.UFS)
    xty_nivel = np.zeros(sum(MANTER) + 1)
    periodos_treino = set()
    print("Passada 1/2: ajustando nível e log em 2013–2023", flush=True)
    for lote, df in enumerate(lotes_temporais(entrada, batch_size, PRIMEIRO_ANO, ULTIMO_TREINO), 1):
        periodos_treino.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        x = matriz_temporal(df)
        y, w = df["y"].to_numpy(), pesos_amostrais(df)
        modelo.acumular_normal(x, y, w)
        xty_nivel += x.T @ (w * np.exp(y))
        if lote % 100 == 0:
            print(f"  {lote} lotes · {modelo.n:,} registros de treino", flush=True)
    conferir_periodos(periodos_treino, PRIMEIRO_ANO, ULTIMO_TREINO)
    modelo.resolver()
    if modelo.k - 1 not in modelo.indices_ativos:
        raise ValueError("O treino não identifica a tendência linear de ano.")
    beta_nivel = modelo.bread @ xty_nivel

    n_calibracao = 0
    soma_pesos_calibracao = soma_fatores = 0.
    periodos_calibracao, periodos_teste = set(), set()
    lotes_teste = []
    print("Passada 2/2: calibrando Duan em 2024 e prevendo 2025", flush=True)
    for df in lotes_temporais(entrada, batch_size, CALIBRACAO, TESTE):
        for ano, parte in df.groupby("Ano", sort=False):
            x = matriz_temporal(parte)
            y, w = parte["y"].to_numpy(), pesos_amostrais(parte)
            previsto_log = x @ modelo.beta
            if ano == CALIBRACAO:
                periodos_calibracao.update((ano, tri) for tri in parte["Trimestre"].unique())
                n_calibracao += len(parte)
                soma_pesos_calibracao += float(w.sum())
                soma_fatores += float(w @ np.exp(y - previsto_log))
            else:
                periodos_teste.update((ano, tri) for tri in parte["Trimestre"].unique())
                lotes_teste.append(pd.DataFrame({
                    "observado": np.exp(y), "V1028": w,
                    "nivel": x @ beta_nivel, "log_exp": np.exp(previsto_log),
                }))
    conferir_periodos(periodos_calibracao, CALIBRACAO, CALIBRACAO)
    conferir_periodos(periodos_teste, TESTE, TESTE)
    fim = entrada.stat()
    if (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")

    fator_duan = soma_fatores / soma_pesos_calibracao
    teste = pd.concat(lotes_teste, ignore_index=True)
    teste["log_duan"] = teste["log_exp"] * fator_duan
    if not np.isfinite(teste.to_numpy()).all() or fator_duan <= 0:
        raise ValueError("As previsões e o fator de Duan devem ser finitos, com fator positivo.")
    cortes = quantis_ponderados(teste, "observado", np.arange(1, 10) / 10)
    decis = np.searchsorted(cortes, teste["observado"], side="left")
    limites = [None, *map(float, cortes), None]
    grupos = [("Brasil", teste, None, None)] + [
        (f"D{i + 1}", teste.loc[decis == i], limites[i], limites[i + 1]) for i in range(10)
    ]
    versoes = {"nivel": "Ajuste direto em reais", "log_exp": "Log convertido pela exponencial", "log_duan": "Log com correção de Duan"}
    linhas = []
    for grupo, parte, inferior, superior in grupos:
        observado, pesos = parte["observado"].to_numpy(), parte["V1028"].to_numpy()
        for codigo, versao in versoes.items():
            linhas.append({
                "grupo": grupo, "versao_modelo": versao, "codigo": codigo,
                "limite_inferior_rs_h": inferior, "limite_superior_rs_h": superior,
                **metricas_ponderadas(observado, parte[codigo].to_numpy(), pesos),
            })
    relatorio = {
        "status": "ok", "versao": "escala-m5-1.0", "periodo": "2025T1 a 2025T4",
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size}, "n_observacoes": len(teste),
        "treino": "2013–2023", "calibracao": "2024", "teste": "2025",
        "n_treino": modelo.n, "n_calibracao": n_calibracao, "fator_duan": fator_duan,
        "tendencia_ano": "Linear, centrada em 2013", "data_base": "Janeiro de 2026",
        "rows": linhas,
        "notas": [
            "A comparação retrospectiva usa os mesmos controles e registros por etapa. A tendência linear de ano extrapola para 2024 e 2025. O fator global de Duan é a média ponderada de exp(resíduo em log) em 2024.",
            "MAE, RMSE e viés usam V1028 e reais por hora a preços de janeiro de 2026. Viés é previsto menos observado. Valores não positivos do ajuste em nível entram nas métricas como foram estimados.",
            "Os decis ponderados usam o salário-hora observado em 2025, com empates agrupados. Os três modelos são avaliados nos mesmos registros de cada faixa.",
        ],
    }
    exportar_comparacao(saida, "escala_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "previsao_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
