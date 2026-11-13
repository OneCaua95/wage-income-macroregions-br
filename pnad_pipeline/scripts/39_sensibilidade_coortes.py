#!/usr/bin/env python3
"""Incerteza das médias das coortes e sensibilidade às faixas de nascimento."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import t
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao

SPEC = importlib.util.spec_from_file_location("_sensibilidade_pseudopainel", SCRIPT_DIR / "18_pseudopainel_coortes.py")
PP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PP
SPEC.loader.exec_module(PP)
HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"
ESQUEMAS = {
    "historica": ("Faixas históricas", PP.M5.BIRTH_BINS),
    "quinquenal_deslocada": ("Faixas de 5 anos deslocadas", [1946, 1949, 1954, 1959, 1964, 1969, 1974, 1979, 1984, 1989, 1994, 1999, 2003]),
    "decenal": ("Faixas de 10 anos", [1946, 1949, 1959, 1969, 1979, 1989, 1999, 2003]),
}


def incerteza_celulas(df, cells, edges, desenho):
    nascimento = df.Ano_Nascimento.where(df.Ano_Nascimento.between(1947, 2003), df.Ano - df.Idade)
    faixa = pd.cut(nascimento, bins=edges, labels=False, include_lowest=True).astype(int)
    coortes = df.Sexo.astype(int) * 100 + faixa
    codigos = pd.Categorical(coortes, categories=cells.coorte_id).codes
    if np.any(codigos < 0):
        raise ValueError("Há trabalhadores sem célula na definição de coortes.")
    w = pesos_amostrais(df)
    fatores = w / cells.w.to_numpy()[codigos]
    k = len(cells)
    scores = np.zeros((len(df), 2 * k))
    linhas = np.arange(len(df))
    scores[linhas, codigos] = fatores * (df.y.to_numpy() - cells.media_log_salario_hora.to_numpy()[codigos])
    scores[linhas, codigos + k] = fatores * (df.Anos_de_Estudo.to_numpy() - cells.media_anos_estudo.to_numpy()[codigos])
    desenho.iniciar_scores(2 * k)
    desenho.acumular(df, scores)
    erros = np.sqrt(np.maximum(np.diag(desenho.covariancia(np.eye(2 * k))), 0))
    critico = t.ppf(.975, desenho.graus_liberdade)
    cells = cells.copy()
    cells["erro_padrao_media_log"] = erros[:k]
    cells["erro_padrao_media_anos_estudo"] = erros[k:]
    cells["ic95_inferior_media_log"] = cells.media_log_salario_hora - critico * erros[:k]
    cells["ic95_superior_media_log"] = cells.media_log_salario_hora + critico * erros[:k]
    cells["ic95_inferior_media_geometrica"] = np.exp(cells.ic95_inferior_media_log)
    cells["ic95_superior_media_geometrica"] = np.exp(cells.ic95_superior_media_log)
    cells["graus_liberdade_desenho"] = desenho.graus_liberdade
    return cells


def executar(entrada, saida):
    origem = entrada.stat()
    parquet = PP.M5.pq.ParquetFile(entrada)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") != b"true":
        raise ValueError("Use a base completa para registrar o desenho antes do domínio.")
    periodos = set()
    for batch in parquet.iter_batches(batch_size=100_000, columns=["Ano", "Trimestre"]):
        periodos.update(batch.to_pandas().drop_duplicates().itertuples(index=False, name=None))
    dataset = ds.dataset(entrada)
    partes = []
    n_elegiveis = 0
    for ano, trimestre in sorted(periodos):
        filtro = (ds.field("Ano") == ano) & (ds.field("Trimestre") == trimestre)
        batch = dataset.to_table(columns=PP.M5.RAW_COLUMNS, filter=filtro)
        desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
        desenho.registrar(batch.to_pandas())
        df = PP.M5.normalizar_lote(batch)
        n_elegiveis += len(df)
        if df.empty:
            continue
        for codigo, (rotulo, edges) in ESQUEMAS.items():
            cells = PP.montar_cells([PP.agregar_lote(batch, birth_bins=edges)], birth_bins=edges)
            cells = incerteza_celulas(df, cells, edges, desenho)
            cells["esquema"] = codigo
            cells["definicao_coorte"] = rotulo
            partes.append(cells)
        print(f"{ano}T{trimestre}: três definições de coorte com IC95% das médias", flush=True)
    if not partes:
        raise ValueError("A amostra elegível ficou vazia.")
    todas = pd.concat(partes, ignore_index=True)
    modelos, coeficientes, indicadores = [], [], []
    for codigo, (rotulo, edges) in ESQUEMAS.items():
        cells = todas.loc[todas.esquema.eq(codigo)]
        if int(cells.n.sum()) != n_elegiveis:
            raise ValueError("As definições de coortes alteraram a amostra salarial.")
        indicadores.append({"esquema": codigo, "rotulo": rotulo, "limites_nascimento": edges,
                            "n_celulas": len(cells), "n_coortes": int(cells.coorte_id.nunique()),
                            "mediana_ep_media_log": float(cells.erro_padrao_media_log.median()),
                            "p90_ep_media_log": float(cells.erro_padrao_media_log.quantile(.9))})
        for minimo_n, minimo_kish in PP.MINIMUMS:
            for especificacao in ("efeitos_fixos", "composicao", "tendencias"):
                contexto = {"esquema": codigo, "definicao_coorte": rotulo, "limiar_n": minimo_n,
                            "limiar_kish": minimo_kish, "especificacao": especificacao}
                try:
                    resumo, coef = PP.fit_model(cells, especificacao, minimo_n, minimo_kish)
                except ValueError as exc:
                    modelos.append({**contexto, "status": "indisponivel", "motivo": str(exc)})
                    continue
                modelos.append({**contexto, **resumo, "status": "ok"})
                coeficientes.extend({**contexto, **row} for row in coef
                                    if row["termo"] in {"media_anos_estudo", "proporcao_formal", "proporcao_setor_publico"})
    fim = entrada.stat()
    if (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento.")
    campos = ["esquema", "definicao_coorte", "periodo", "coorte_id", "coorte", "sexo", "faixa_nascimento", "n",
              "n_efetivo_kish", "media_log_salario_hora", "media_geometrica_salario_hora", "media_anos_estudo",
              "erro_padrao_media_log", "erro_padrao_media_anos_estudo", "ic95_inferior_media_log", "ic95_superior_media_log",
              "ic95_inferior_media_geometrica", "ic95_superior_media_geometrica", "graus_liberdade_desenho"]
    todas["trimestre"] = todas.periodo
    campos = ["trimestre", *[c for c in campos if c != "periodo"]]
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "sensibilidade-coortes-1.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}", "ponderacao": "V1028",
        "n_observacoes": n_elegiveis, "n_periodos": len(periodos), "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "rows": todas[campos].to_dict("records"), "esquemas": indicadores, "modelos": modelos,
        "coeficientes": pd.DataFrame(coeficientes).replace([np.nan, np.inf, -np.inf], None).to_dict("records"),
        "notas": [
            "As três definições usam os mesmos trabalhadores elegíveis do M5. As faixas alternativas de 5 e 10 anos têm bordas parciais para preservar o intervalo de nascimento disponível, de 1947 a 2003.",
            "Os IC95% pontuais das médias de log salário e os erros-padrão da escolaridade média usam V1028 e linearização por UPA dentro de estrato em cada trimestre, com cadastro completo antes do domínio.",
            "As regressões são refeitas para três definições de coorte, três limiares de n/Kish e três especificações. As médias têm erro amostral, inclusive nas covariáveis agregadas. Seus erros-padrão são diagnósticos e não corrigem o erro de mensuração nos coeficientes.",
            "Os ICs dos coeficientes continuam exploratórios, clusterizados por coorte e trimestre. Rotação amostral e poucos grupos limitam a inferência temporal. Diferenças de ajuste também refletem mudanças na agregação e nas células incluídas.",
        ],
    }
    exportar_comparacao(saida, "sensibilidade_coortes", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/sensibilidade_coortes")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir)
