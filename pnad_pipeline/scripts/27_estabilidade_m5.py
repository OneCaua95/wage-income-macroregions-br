#!/usr/bin/env python3
"""Estima a estabilidade dos contrastes do M5 entre anos completos da PNAD."""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.regressao.estabilidade import resumir_estabilidade
from pnad_pipeline.analise.regressao.robustez import contrastes_m5

SPEC = importlib.util.spec_from_file_location("_pnad_m5_estabilidade", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida, batch_size=100_000):
    entrada, saida = Path(entrada), Path(saida)
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
    modelos, trimestres = {}, {}
    nomes = ["Intercept", *M5.nomes_features(M5.UFS)]

    print("Passada 1/2: ajustando o M5 por ano", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        desenho.registrar(batch.to_pandas())
        for ano, df in M5.normalizar_lote(batch).groupby("Ano", sort=False):
            ano = int(ano)
            if ano not in modelos:
                modelos[ano] = M5.ResultadoModelo(str(ano), M5.UFS)
                trimestres[ano] = set()
            trimestres[ano].update(df["Trimestre"].unique())
            modelos[ano].acumular_normal(M5.matriz_m5(df, M5.UFS), df["y"].to_numpy(), pesos_amostrais(df))
        if lote % 80 == 0:
            print(f"  {lote} lotes · {sum(modelo.n for modelo in modelos.values()):,} registros elegíveis", flush=True)

    anos = sorted(modelos)
    if len(anos) < 2 or any(trimestres[ano] != {1, 2, 3, 4} for ano in anos):
        raise ValueError("A análise requer pelo menos dois anos com os quatro trimestres elegíveis.")
    rotulos, contrastes, escolaridade_media = contrastes_m5(nomes, sum(modelos[ano].xtx for ano in anos))
    termos = np.flatnonzero(np.any(contrastes != 0, axis=0))
    projecoes, fatias = {}, {}
    for i, ano in enumerate(anos):
        modelo = modelos[ano]
        modelo.resolver()
        if not np.isin(termos, modelo.indices_ativos).all():
            raise ValueError(f"A amostra de {ano} não identifica todos os contrastes do M5.")
        projecoes[ano] = modelo.bread @ contrastes.T
        fatias[ano] = slice(i * len(rotulos), (i + 1) * len(rotulos))

    dimensao = len(anos) * len(rotulos)
    desenho.iniciar_scores(dimensao)
    contagens = dict.fromkeys(anos, 0)
    print("Passada 2/2: acumulando contribuições das UPAs para a covariância entre anos", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        for ano, df in M5.normalizar_lote(batch).groupby("Ano", sort=False):
            contagens[ano] += len(df)
            x = M5.matriz_m5(df, M5.UFS)
            residuo_ponderado = pesos_amostrais(df) * (df["y"].to_numpy() - x @ modelos[ano].beta)
            scores = np.zeros((len(df), dimensao))
            scores[:, fatias[ano]] = (x @ projecoes[ano]) * residuo_ponderado[:, None]
            desenho.acumular(df, scores)
        if lote % 80 == 0:
            print(f"  {lote} lotes · {sum(contagens.values()):,} registros com contribuições", flush=True)

    fim = entrada.stat()
    if (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns) or any(
        contagens[ano] != modelos[ano].n for ano in anos
    ):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")

    cov = desenho.covariancia(np.eye(dimensao))
    estimativas = np.array([contrastes @ modelos[ano].beta for ano in anos])
    linhas, testes = resumir_estabilidade(anos, rotulos, estimativas, cov, desenho.graus_liberdade, contagens)
    periodo = f"{anos[0]}–{anos[-1]}"
    relatorio = {
        "status": "ok", "versao": "estabilidade-m5-1.0", "periodo": periodo, "anos": anos,
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "n_observacoes": sum(contagens.values()), "n_upas_desenho": len(desenho.grupos),
        "n_estratos_desenho": len(desenho.estratos), "graus_liberdade": desenho.graus_liberdade,
        "media_ponderada_anos_estudo": escolaridade_media, "nivel_confianca": .95,
        "rows": linhas, "testes": testes,
        "notas": [
            "O M5 é ajustado por ano com os critérios de elegibilidade, pesos V1028 e controles da análise histórica. Os contrastes usam a mesma distribuição ponderada de referência em todos os anos.",
            "IC95% pontuais por linearização de UPA dentro de estrato, com o perfil de referência tratado como fixo. O teste F de Wald verifica se o contraste em log é igual em todos os anos. Os três valores de p são ajustados por Holm.",
            "A covariância preserva UPAs recorrentes dentro do mesmo estrato. A variância segue a aproximação do M5 histórico, com as reclassificações de estrato tratadas como versões distintas do conglomerado.",
        ],
    }
    saida.mkdir(parents=True, exist_ok=True)
    csv, resumo = "estabilidade_m5.csv", "resumo_estabilidade_m5.json"
    testes_por_contraste = {teste["contraste"]: teste for teste in testes}
    exportacao = [{**linha, **testes_por_contraste[linha["contraste"]], "periodo": periodo, "ponderacao": "V1028"} for linha in linhas]
    with TemporaryDirectory(dir=saida) as pasta:
        temporario = Path(pasta)
        pd.DataFrame(exportacao).to_csv(temporario / csv, index=False, encoding="utf-8-sig")
        (temporario / resumo).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporario / csv, saida / csv)
        os.replace(temporario / resumo, saida / resumo)
    print(f"Análise concluída em {saida}", flush=True)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/estabilidade_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
