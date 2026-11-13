#!/usr/bin/env python3
"""Crescimento real da média salarial desde 2013T1 e frente ao mesmo trimestre do ano anterior."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.crescimento import comparar_medias
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"
GRUPOS = ["Brasil", "Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
CHAVES = ["UF", "Estrato", "UPA"]


def executar(entrada, entrada_analise, saida):
    fontes = [entrada, entrada_analise]
    origens = [(p.stat().st_size, p.stat().st_mtime_ns) for p in fontes]
    parquet = pq.ParquetFile(entrada)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") != b"true":
        raise ValueError("Use a base completa para registrar o desenho.")
    periodos = set()
    for batch in parquet.iter_batches(batch_size=100_000, columns=["Ano", "Trimestre"]):
        periodos.update(batch.to_pandas().drop_duplicates().itertuples(index=False, name=None))
    if (2013, 1) not in periodos:
        raise ValueError("O crescimento acumulado exige 2013T1 na entrada.")
    bruto, analise = ds.dataset(entrada), ds.dataset(entrada_analise)
    historico = {}
    n = 0
    for ano, tri in sorted(periodos):
        filtro = (ds.field("Ano") == ano) & (ds.field("Trimestre") == tri)
        raw = bruto.to_table(columns=[*CHAVES, "V1028"], filter=filtro).to_pandas()
        desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
        desenho.registrar(raw)
        df = analise.to_table(columns=[*CHAVES, "V1028", "Regiao", "Idade", "Condicao_Ocupacao", "Salario_Hora",
                                      "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas"], filter=filtro).to_pandas()
        df = df.loc[df.Idade.between(25, 65) & df.Condicao_Ocupacao.eq(1) & df.Rendimento_hab_Trab_princ.gt(0)
                    & df.Horas_hab_trabalhadas.gt(0) & df.Salario_Hora.gt(0) & np.isfinite(df.Salario_Hora)].reset_index(drop=True)
        n += len(df)
        scores = np.zeros((len(df), len(GRUPOS)))
        estatisticas = []
        for i, grupo in enumerate(GRUPOS):
            mask = np.ones(len(df), dtype=bool) if grupo == "Brasil" else df.Regiao.eq(grupo).to_numpy()
            sub = df.loc[mask]
            if len(sub) < 30:
                raise ValueError(f"Menos de 30 salários em {ano}T{tri}, {grupo}.")
            w, y = pesos_amostrais(sub), sub.Salario_Hora.to_numpy(dtype=float)
            media = float(w @ y / w.sum())
            estatisticas.append({"media": media, "n_salarios": len(sub)})
            scores[mask, i] = w * (y - media) / w.sum()
        desenho.iniciar_scores(len(GRUPOS))
        desenho.acumular(df, scores)
        historico[ano, tri] = {"desenho": desenho, "cadastro": raw.drop_duplicates(CHAVES), "estatisticas": estatisticas}
        print(f"{ano}T{tri}: contribuições das médias por UPA", flush=True)
    rows = []
    for periodo, atual in sorted(historico.items()):
        ano, tri = periodo
        referencias = [("desde_2013", (2013, 1))]
        if (ano - 1, tri) in historico:
            referencias.append(("interanual", (ano - 1, tri)))
        for tipo, referencia in referencias:
            base = historico[referencia]
            conjunto = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
            conjunto.registrar(pd.concat([base["cadastro"], atual["cadastro"]], ignore_index=True).drop_duplicates(CHAVES))
            k = len(GRUPOS)
            conjunto.iniciar_scores(2 * k)
            for offset, dados in ((0, base), (k, atual)):
                ids = [conjunto.grupos[chave] for chave in dados["desenho"].grupos]
                conjunto.scores[np.asarray(ids), offset:offset+k] = dados["desenho"].scores
            cov = conjunto.covariancia(np.eye(2 * k))
            comuns = len(base["desenho"].grupos.keys() & atual["desenho"].grupos.keys())
            for i, grupo in enumerate(GRUPOS):
                media_base, media_atual = base["estatisticas"][i]["media"], atual["estatisticas"][i]["media"]
                rows.append({"trimestre": f"{ano}T{tri}", "referencia": f"{referencia[0]}T{referencia[1]}",
                             "tipo": tipo, "grupo": grupo, "media_base": media_base, "media_atual": media_atual,
                             "n_salarios_base": base["estatisticas"][i]["n_salarios"],
                             "n_salarios_atual": atual["estatisticas"][i]["n_salarios"],
                             "n_upas_compartilhadas": comuns, "n_upas_multiestrato": len(conjunto.upas_multiestrato),
                             "graus_liberdade": conjunto.graus_liberdade,
                             **comparar_medias(media_base, media_atual, cov[np.ix_([i, i+k], [i, i+k])], conjunto.graus_liberdade)})
    if origens != [(p.stat().st_size, p.stat().st_mtime_ns) for p in fontes]:
        raise ValueError("Uma das bases mudou durante o processamento.")
    relatorio = {
        "status": "ok", "versao": "crescimento-salarial-1.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{rows[0]['trimestre']} a {rows[-1]['trimestre']}", "ponderacao": "V1028",
        "n_observacoes": n, "n_periodos": len(periodos), "rows": rows,
        "entradas": [{"arquivo": p.name, "bytes": v[0]} for p, v in zip(fontes, origens)],
        "notas": [
            "Crescimento da média aritmética do salário-hora real, em preços de janeiro/2026, no domínio descritivo de ocupados de 25 a 65 anos. Acumulado usa 2013T1 como referência fixa; interanual compara o mesmo trimestre do ano anterior.",
            "A covariância conjunta usa os scores das duas médias nas UPAs compartilhadas, dentro de estrato, e o cadastro completo da união dos dois períodos. O IC95% pontual aplica o método delta ao log da razão e a distribuição t do desenho conjunto.",
            "A variância temporal é uma aproximação com reposição. UPAs reclassificadas são separadas por UF/estrato/UPA. A rotação e a mudança da Amostra Mestra limitam a covariância identificável; o cálculo não reproduz a variância temporal oficial.",
            "A série acumulada tem uma referência fixa e inclui diferenças sazonais entre trimestres. Os seletores locais escolhem grupo e comparação. Os filtros demográficos globais não são aplicados.",
        ],
    }
    exportar_comparacao(saida, "crescimento_salarial", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--entrada-analise", type=Path, default=HISTORICO / "pnad_analise_25_65.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/crescimento_salarial")
    args = parser.parse_args()
    executar(args.entrada, args.entrada_analise, args.saida_dir)
