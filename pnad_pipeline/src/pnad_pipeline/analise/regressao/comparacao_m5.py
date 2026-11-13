"""Comparação pareada entre especificações aninhadas do M5."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import t

from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.regressao.robustez import contrastes_m5, tabela_comparacao


def controles_emprego(m5):
    return {
        *m5.SETORES.values(), *m5.OCUPACOES.values(),
        "is_formal", "is_public", "Estudo_x_is_formal", "Estudo_x_is_public",
    }


def comparar_especificacoes(m5, entrada, batch_size, *, excluir, indices_contrastes, prefixos):
    """Reajusta o M5 retirando colunas da regressão na mesma amostra."""
    entrada = Path(entrada)
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
    completo = m5.ResultadoModelo("completo", m5.UFS)
    nomes = ["Intercept", *m5.nomes_features(m5.UFS)]
    desconhecidos = excluir.difference(nomes)
    if desconhecidos:
        raise ValueError(f"Termos ausentes na matriz do M5: {', '.join(sorted(desconhecidos))}.")
    manter = np.array([nome not in excluir for nome in nomes])
    periodos = set()

    print("Passada 1/2: ajustando as duas especificações na amostra comum", flush=True)
    for lote, batch in enumerate(m5.iterar_lotes(entrada, batch_size), 1):
        desenho.registrar(batch.to_pandas())
        df = m5.normalizar_lote(batch)
        if df.empty:
            continue
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        completo.acumular_normal(m5.matriz_m5(df, m5.UFS), df["y"].to_numpy(), pesos_amostrais(df))
        if lote % 80 == 0:
            print(f"  {lote} lotes · {completo.n:,} registros elegíveis", flush=True)

    if not completo.n:
        raise ValueError("A amostra elegível do M5 ficou vazia.")
    # As equações normais da versão reduzida usam as mesmas linhas e pesos.
    reduzido = m5.ResultadoModelo(
        "reduzido", m5.UFS, k=int(manter.sum()), n=completo.n,
        xtx=completo.xtx[np.ix_(manter, manter)], xty=completo.xty[manter],
    )
    rotulos, contrastes, escolaridade_media = contrastes_m5(nomes, completo.xtx)
    rotulos = [rotulos[i] for i in indices_contrastes]
    contrastes = contrastes[list(indices_contrastes)]
    contrastes_reduzido = contrastes[:, manter]
    for modelo, matriz in ((completo, contrastes), (reduzido, contrastes_reduzido)):
        modelo.resolver()
        termos = np.flatnonzero(np.any(matriz != 0, axis=0))
        if not np.isin(termos, modelo.indices_ativos).all():
            raise ValueError(f"A amostra não identifica todos os contrastes na versão {modelo.nome}.")

    proj_completo = completo.bread @ contrastes.T
    proj_reduzido = reduzido.bread @ contrastes_reduzido.T
    dimensao = 2 * len(rotulos)
    desenho.iniciar_scores(dimensao)
    n_scores = 0
    print("Passada 2/2: estimando a covariância entre os ajustes", flush=True)
    for lote, batch in enumerate(m5.iterar_lotes(entrada, batch_size), 1):
        df = m5.normalizar_lote(batch)
        if df.empty:
            continue
        n_scores += len(df)
        x = m5.matriz_m5(df, m5.UFS)
        x_reduzido = x[:, manter]
        y, w = df["y"].to_numpy(), pesos_amostrais(df)
        residuo_completo = w * (y - x @ completo.beta)
        residuo_reduzido = w * (y - x_reduzido @ reduzido.beta)
        scores = np.column_stack((
            (x @ proj_completo) * residuo_completo[:, None],
            (x_reduzido @ proj_reduzido) * residuo_reduzido[:, None],
        ))
        desenho.acumular(df, scores)
        if lote % 80 == 0:
            print(f"  {lote} lotes · {n_scores:,} registros com contribuições", flush=True)

    fim = entrada.stat()
    if n_scores != completo.n or (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")
    cov = desenho.covariancia(np.eye(dimensao))
    linhas = tabela_comparacao(
        rotulos, contrastes @ completo.beta, contrastes_reduzido @ reduzido.beta,
        cov, t.ppf(.975, desenho.graus_liberdade), prefixos=prefixos,
    )
    inicio, ultimo = min(periodos), max(periodos)
    periodo = f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}"
    return {
        "status": "ok", "periodo": periodo,
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "n_periodos": len(periodos), "n_observacoes": completo.n, "soma_pesos": completo.soma_pesos,
        "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
        "graus_liberdade": desenho.graus_liberdade, "nivel_confianca": .95,
        "media_ponderada_anos_estudo": escolaridade_media, "rows": linhas,
    }


def exportar_comparacao(saida, nome, relatorio):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    csv, resumo = f"{nome}.csv", f"resumo_{nome}.json"
    contexto = {campo: relatorio[campo] for campo in ("periodo", "n_observacoes", "ponderacao")}
    exportacao = [{**linha, **contexto} for linha in relatorio["rows"]]
    with TemporaryDirectory(dir=saida) as pasta:
        temporario = Path(pasta)
        pd.DataFrame(exportacao).to_csv(temporario / csv, index=False, encoding="utf-8-sig")
        (temporario / resumo).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporario / csv, saida / csv)
        os.replace(temporario / resumo, saida / resumo)
    print(f"Comparação concluída em {saida}", flush=True)
