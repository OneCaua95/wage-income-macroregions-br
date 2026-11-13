#!/usr/bin/env python3
"""Compara os contrastes do M5 em log do salário-hora real e nominal."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao
from pnad_pipeline.analise.regressao.robustez import contrastes_m5, tabela_comparacao
from pnad_pipeline.extracao.constants import IPCA_SGS_URL_PADRAO
from pnad_pipeline.extracao.deflacao import calcular_data_referencia_trimestre
from pnad_pipeline.extracao.ipca import calcular_fator_para_base, get_ipca_bc

SPEC = importlib.util.spec_from_file_location("_pnad_m5_nominal_real", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"
PERIODOS = [(ano, trimestre) for ano in range(2013, 2026) for trimestre in range(1, 5)]


def fatores_ipca(ipca, data_base):
    referencias = [calcular_data_referencia_trimestre(*periodo) for periodo in PERIODOS]
    limites = pd.to_datetime([referencias[0], referencias[-1], data_base], format="%d/%m/%Y")
    meses = pd.date_range(limites.min(), limites.max(), freq="MS")
    taxas = ipca.set_index("data")["valor"].reindex(meses).to_numpy(dtype=float)
    if not np.isfinite(taxas).all() or np.any(taxas <= -100):
        raise ValueError("IPCA mensal incompleto ou inválido entre os trimestres e a data-base.")
    return pd.Series(
        [calcular_fator_para_base(ipca, referencia, data_base) for referencia in referencias],
        index=pd.MultiIndex.from_tuples(PERIODOS, names=["Ano", "Trimestre"]),
    )


def ajuste_log(df, fatores_log):
    chaves = pd.MultiIndex.from_frame(df[["Ano", "Trimestre"]])
    ajuste = fatores_log.reindex(chaves).to_numpy()
    if not np.isfinite(ajuste).all():
        raise ValueError("Há registros do M5 sem fator de IPCA para o trimestre.")
    return ajuste


def executar(entrada, saida, batch_size=100_000):
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    entrada = Path(entrada)
    origem = entrada.stat()
    data_base = (pq.read_metadata(entrada).metadata or {}).get(b"deflacao_data_base", b"").decode()
    if not data_base:
        raise ValueError("A base deve registrar a data-base da deflação no Parquet.")
    fatores = fatores_ipca(get_ipca_bc(), data_base)
    fatores_log = -np.log(fatores)

    nomes = ["Intercept", *M5.nomes_features(M5.UFS)]
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
    modelo = M5.ResultadoModelo("real", M5.UFS)
    delta_xty = np.zeros(len(nomes))
    periodos = set()

    print("Passada 1/2: ajustando as duas respostas na amostra comum", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        desenho.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        x, w = M5.matriz_m5(df, M5.UFS), pesos_amostrais(df)
        modelo.acumular_normal(x, df["y"].to_numpy(), w)
        delta_xty += x.T @ (w * ajuste_log(df, fatores_log))
        if lote % 80 == 0:
            print(f"  {lote} lotes · {modelo.n:,} registros elegíveis", flush=True)

    if not modelo.n:
        raise ValueError("A amostra elegível do M5 ficou vazia.")
    modelo.resolver()
    # Os ajustes compartilham X'WX. Só o produto com a resposta muda.
    delta_beta = modelo.bread @ delta_xty
    rotulos, contrastes, escolaridade_media = contrastes_m5(nomes, modelo.xtx)
    termos = np.flatnonzero(np.any(contrastes != 0, axis=0))
    if not np.isin(termos, modelo.indices_ativos).all():
        raise ValueError("A amostra não identifica todos os contrastes do M5.")
    projecao = modelo.bread @ contrastes.T
    dimensao = 2 * len(rotulos)
    desenho.iniciar_scores(dimensao)

    n_scores = 0
    print("Passada 2/2: estimando a covariância entre os ajustes", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        n_scores += len(df)
        x, w = M5.matriz_m5(df, M5.UFS), pesos_amostrais(df)
        residuo_real = df["y"].to_numpy() - x @ modelo.beta
        residuo_nominal = residuo_real + ajuste_log(df, fatores_log) - x @ delta_beta
        z = (x @ projecao) * w[:, None]
        desenho.acumular(df, np.column_stack((z * residuo_real[:, None], z * residuo_nominal[:, None])))
        if lote % 80 == 0:
            print(f"  {lote} lotes · {n_scores:,} registros com contribuições", flush=True)

    fim = entrada.stat()
    if n_scores != modelo.n or (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")
    linhas = tabela_comparacao(
        rotulos, contrastes @ modelo.beta, contrastes @ (modelo.beta + delta_beta),
        desenho.covariancia(np.eye(dimensao)), t.ppf(.975, desenho.graus_liberdade),
        prefixos=("real", "nominal"),
    )
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "nominal-real-m5-1.0",
        "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}",
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "data_base_real": data_base, "fonte_ipca": IPCA_SGS_URL_PADRAO,
        "fatores_ipca": {f"{a}T{q}": float(fatores.loc[a, q]) for a, q in sorted(periodos)},
        "n_periodos": len(periodos), "n_observacoes": modelo.n, "soma_pesos": modelo.soma_pesos,
        "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
        "graus_liberdade": desenho.graus_liberdade, "nivel_confianca": .95,
        "media_ponderada_anos_estudo": escolaridade_media, "rows": linhas,
        "notas": [
            f"O salário real está a preços de {data_base}. O nominal foi reconstruído dividindo o salário real arquivado pelo fator trimestral de IPCA da coleta (SGS 433).",
            "Os dois ajustes usam o log do salário-hora, a mesma amostra, os pesos V1028 e os controles do M5, incluindo indicadores separados de ano e trimestre.",
            "A mudança é nominal menos real, em pontos percentuais. Os IC95% pareados consideram a covariância dos dois ajustes por UPA dentro de estrato. O perfil ponderado de referência e os fatores de IPCA são tratados como fixos.",
        ],
    }
    exportar_comparacao(saida, "nominal_real_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/nominal_real_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
