#!/usr/bin/env python3
"""Resume a influência de exclusões individuais sobre três contrastes do M5."""

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
from pnad_pipeline.analise.regressao.influencia import mudancas_por_exclusao, selecionar_influentes
from pnad_pipeline.analise.regressao.robustez import contrastes_m5

SPEC = importlib.util.spec_from_file_location("_pnad_m5_influencia", SCRIPT_DIR / "08_inferencia_m5.py")
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

    rotulos, contrastes, _ = contrastes_m5(["Intercept", *M5.nomes_features(M5.UFS)], modelo.xtx)
    if not np.isin(np.flatnonzero(np.any(contrastes != 0, axis=0)), modelo.indices_ativos).all():
        raise ValueError("A amostra não identifica todos os contrastes do M5.")
    estimativas_base = 100 * np.expm1(contrastes @ modelo.beta)
    ajustes, pesos = np.empty((2, modelo.n))
    mudancas = np.empty((modelo.n, len(rotulos)))
    regioes = np.empty(modelo.n, dtype=np.int8)
    posicao = 0
    print("Passada 2/2: calculando as mudanças por exclusão individual", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        proxima = posicao + len(df)
        if proxima > modelo.n:
            raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")
        trecho = slice(posicao, proxima)
        pesos[trecho] = pesos_amostrais(df)
        ajustes[trecho], mudancas[trecho] = mudancas_por_exclusao(
            M5.matriz_m5(df, M5.UFS), df["y"].to_numpy(), pesos[trecho], modelo.beta, modelo.bread, contrastes,
        )
        regioes[trecho] = pd.Categorical(df["Regiao"], categories=M5.MACROREGIOES).codes
        posicao = proxima
        if lote % 80 == 0:
            print(f"  {lote} lotes · {posicao:,} exclusões calculadas", flush=True)

    fim = entrada.stat()
    if posicao != modelo.n or (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")

    cortes = quantis_ponderados(pd.DataFrame({"ajuste": ajustes, "V1028": pesos}, copy=False), "ajuste", np.arange(1, 10) / 10)
    decis = np.searchsorted(cortes, ajustes, side="left")
    limites = [None, *map(float, np.exp(cortes)), None]
    grupos = (
        ("decil", decis, [f"D{i}" for i in range(1, 11)]),
        ("macrorregiao", regioes, M5.MACROREGIOES),
    )
    validos = np.isfinite(mudancas).all(axis=1)
    n_avaliados = int(validos.sum())
    if not n_avaliados:
        raise ValueError("Nenhuma exclusão individual preservou a identificação dos contrastes.")

    linhas = []
    for rotulo, estimativa, coluna in zip(rotulos, estimativas_base, mudancas.T):
        corte, influentes = selecionar_influentes(coluna, validos)
        n_influentes = int(influentes.sum())
        mudancas_validas = coluna[validos]
        menor, maior = float(mudancas_validas.min()), float(mudancas_validas.max())
        linhas.append({
            "dimensao": "resumo", "grupo": "Brasil", "contraste": rotulo,
            "estimativa_base_pct": float(estimativa), "mudanca_minima_pp": menor, "mudanca_maxima_pp": maior,
            "maior_mudanca_absoluta_pp": max(abs(menor), abs(maior)), "limiar_absoluto_pp": corte,
            "n_avaliados": n_avaliados, "n_mais_influentes": n_influentes,
        })
        for dimensao, codigos, nomes in grupos:
            totais = np.bincount(codigos[validos], minlength=len(nomes))
            top = np.bincount(codigos[influentes], minlength=len(nomes))
            for i, (nome, total, n_top) in enumerate(zip(nomes, totais, top)):
                linhas.append({
                    "dimensao": dimensao, "grupo": nome, "contraste": rotulo,
                    "limite_inferior_rs_h": limites[i] if dimensao == "decil" else None,
                    "limite_superior_rs_h": limites[i + 1] if dimensao == "decil" else None,
                    "n_avaliados": int(total), "n_mais_influentes": int(n_top),
                    "parcela_dos_avaliados": float(total / n_avaliados),
                    "parcela_dos_mais_influentes": float(n_top / n_influentes) if n_influentes else None,
                    "parcela_influente_no_grupo": float(n_top / total) if total else None,
                })

    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "influencia-m5-1.0", "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}",
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "n_periodos": len(periodos), "n_observacoes": modelo.n, "soma_pesos": modelo.soma_pesos,
        "n_avaliados": n_avaliados, "n_exclusoes_indisponiveis": modelo.n - n_avaliados,
        "rows": linhas,
        "notas": [
            "A mudança é a estimativa após retirar um registro menos a estimativa na amostra completa, em pontos percentuais. O cálculo mantém o perfil de referência e os pesos V1028 das demais observações.",
            "Para cada contraste, os casos mais influentes são o 1% com maiores mudanças absolutas, incluindo empates no corte e excluindo mudanças nulas. As parcelas descrevem contagens de registros. Os decis usam o salário-hora estimado pelo M5 completo e os mesmos cortes do diagnóstico de resíduos.",
            "O painel publica máximos nacionais e contagens agrupadas. O diagnóstico avalia exclusões individuais, com um contraste de cada vez.",
        ],
    }
    exportar_comparacao(saida, "influencia_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/influencia_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
