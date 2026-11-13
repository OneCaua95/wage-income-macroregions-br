#!/usr/bin/env python3
"""Mediana anual ponderada e seu crescimento desde 2013, por macroregião."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import quantis_ponderados, validar_desenho
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"
GRUPOS = ["Brasil", "Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]


def resumir_ano(df, ano):
    if set(df.Trimestre.unique()) != {1, 2, 3, 4}:
        raise ValueError(f"O ano {ano} não contém os quatro trimestres.")
    validar_desenho(df)
    valido = (df.Idade.between(25, 65) & df.Condicao_Ocupacao.eq(1) &
              df.Rendimento_hab_Trab_princ.gt(0) & df.Horas_hab_trabalhadas.gt(0) &
              np.isfinite(df.Salario_Hora) & df.Salario_Hora.gt(0))
    wage = df.loc[valido]
    rows = []
    for grupo in GRUPOS:
        sub = wage if grupo == "Brasil" else wage.loc[wage.Regiao.eq(grupo)]
        trimestres = set(sub.Trimestre.unique())
        suficiente = len(sub) >= 30 and trimestres == {1, 2, 3, 4}
        mediana = float(quantis_ponderados(sub, "Salario_Hora", [.5])[0]) if suficiente else None
        rows.append({"ano": int(ano), "grupo": grupo, "mediana": mediana, "n_salarios": len(sub),
                     "n_trimestres": len(trimestres), "populacao_media_salarios": float(sub.V1028.sum() / 4),
                     "status": "ok" if suficiente else "insufficient"})
    return rows


def executar(entrada, saida):
    origem = (entrada.stat().st_size, entrada.stat().st_mtime_ns)
    parquet = pq.ParquetFile(entrada)
    anos = set()
    for lote in parquet.iter_batches(columns=["Ano"]):
        anos.update(lote.column(0).to_pylist())
    dataset = ds.dataset(entrada)
    colunas = ["Ano", "Trimestre", "UF", "Estrato", "UPA", "V1028", "Idade", "Condicao_Ocupacao",
               "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas", "Salario_Hora", "Regiao"]
    rows = []
    for ano in sorted(anos):
        df = dataset.to_table(columns=colunas, filter=ds.field("Ano") == ano).to_pandas()
        rows.extend(resumir_ano(df, ano))
        print(f"{ano}: {len(df):,} registros · seis medianas anuais", flush=True)
    bases = {r["grupo"]: r for r in rows if r["ano"] == 2013}
    if not bases:
        raise ValueError("A série requer o ano-base 2013.")
    for r in rows:
        base = bases[r["grupo"]]
        r.update({"ano_base": 2013, "mediana_base": base["mediana"], "n_salarios_base": base["n_salarios"],
                  "crescimento_mediana_pct": 100 * (r["mediana"] / base["mediana"] - 1)
                  if r["mediana"] is not None and base["mediana"] is not None and base["mediana"] > 0 else None})
    if origem != (entrada.stat().st_size, entrada.stat().st_mtime_ns):
        raise ValueError("A base mudou durante o processamento.")
    relatorio = {"status": "ok", "versao": "evolucao-regional-anual-1.0",
                 "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
                 "periodo": f"{min(anos)} a {max(anos)}", "entrada": {"arquivo": entrada.name, "bytes": origem[0]},
                 "n_observacoes": sum(r["n_salarios"] for r in rows if r["grupo"] == "Brasil"),
                 "rows": rows, "notas": [
                     "Pessoas ocupadas de 25 a 65 anos com rendimento habitual positivo do trabalho principal e horas habituais positivas em todos os trabalhos. A proxy salarial histórica é VD4016 / (VD4031 × 4,33). Valores reais a preços de janeiro/2026.",
                     "Mediana da distribuição ponderada de todos os registros dos quatro trimestres de cada ano; não é média das medianas trimestrais. Cada trimestre contribui com seus pesos V1028. A divisão comum por quatro não muda a mediana.",
                     "Os registros são pessoas-trimestres, não pessoas únicas. População média de salários = soma dos pesos válidos / quatro; a rotação não identifica trajetórias individuais.",
                     "Crescimento descritivo da mediana anual contra a mediana de 2013 da mesma região. Não equivale ao crescimento da média desde 2013T1. Sem IC neste painel; mínimo de 30 salários e quatro trimestres válidos por célula.",
                     "Resultados fixos para Brasil e macroregiões; os filtros demográficos e o trimestre global não recalculam as medianas anuais.",
                 ]}
    exportar_comparacao(saida, "evolucao_regional_anual", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_analise_25_65.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/evolucao_regional_anual")
    args = parser.parse_args()
    executar(args.entrada, args.saida_dir)
