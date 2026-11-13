#!/usr/bin/env python3
"""IC95% trimestrais da média, P10, mediana e P90 para Brasil e macroregiões."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA
from pnad_pipeline.analise.incerteza import intervalos_salario
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, entrada_analise, saida):
    fontes = [entrada, entrada_analise]
    origens = [(p.stat().st_size, p.stat().st_mtime_ns) for p in fontes]
    parquet = pq.ParquetFile(entrada)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") != b"true":
        raise ValueError("Use a base completa para registrar o desenho anterior ao domínio.")
    periodos = set()
    for batch in parquet.iter_batches(batch_size=100_000, columns=["Ano", "Trimestre"]):
        periodos.update(batch.to_pandas().drop_duplicates().itertuples(index=False, name=None))
    bruto, analise = ds.dataset(entrada), ds.dataset(entrada_analise)
    rows, n_salarios = [], 0
    for ano, trimestre in sorted(periodos):
        filtro = (ds.field("Ano") == ano) & (ds.field("Trimestre") == trimestre)
        raw = bruto.to_table(columns=["UF", "Estrato", "UPA", "V1028"], filter=filtro).to_pandas()
        desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
        desenho.registrar(raw)
        df = analise.to_table(columns=["UF", "Estrato", "UPA", "V1028", "Regiao", "Idade",
                                      "Condicao_Ocupacao", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas", "Salario_Hora"], filter=filtro).to_pandas()
        valido = (df.Idade.between(25, 65) & df.Condicao_Ocupacao.eq(1)
                  & df.Rendimento_hab_Trab_princ.gt(0) & df.Horas_hab_trabalhadas.gt(0)
                  & df.Salario_Hora.gt(0) & np.isfinite(df.Salario_Hora))
        df = df.loc[valido]
        n_salarios += len(df)
        grupos = [("Brasil", df), *df.groupby("Regiao", observed=True)]
        for grupo, sub in grupos:
            for resultado in intervalos_salario(sub, desenho):
                rows.append({"trimestre": f"{ano}T{trimestre}", "grupo": grupo,
                             "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
                             **resultado})
        print(f"{ano}T{trimestre}: {len(df):,} salários na amostra descritiva", flush=True)
    if origens != [(p.stat().st_size, p.stat().st_mtime_ns) for p in fontes]:
        raise ValueError("Uma das bases mudou durante o processamento.")
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "incerteza-descritiva-1.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}", "ponderacao": "V1028",
        "n_observacoes": n_salarios, "n_periodos": len(periodos), "nivel_confianca": .95,
        "entradas": [{"arquivo": p.name, "bytes": info[0]} for p, info in zip(fontes, origens)],
        "rows": rows,
        "notas": [
            "A amostra é a mesma das descritivas: ocupados de 25 a 65 anos com rendimento e horas positivos e salário-hora finito. Os valores são reais de janeiro/2026.",
            "Cada trimestre usa V1028 e cadastro completo de UPAs antes do domínio, com linearização dentro de estrato, aproximação com reposição e política de UPA única ajustar.",
            "O IC95% da média usa a linearização da razão. Os quantis usam Woodruff com intervalo t da proporção de salários menores ou iguais ao quantil, transformado pela inversa empírica ponderada. Empates são preservados.",
            "Os intervalos são pontuais. Limites do quantil sem suporte na escala de probabilidade são apresentados como indisponíveis. A faixa representa incerteza da estimativa populacional naquele trimestre.",
            "Brasil e macroregiões são grupos pré-calculados. O período escolhe o trimestre da tabela e os filtros demográficos globais não são aplicados a esta série.",
        ],
        "referencia": "https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyquantile.html",
    }
    exportar_comparacao(saida, "incerteza_descritiva", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--entrada-analise", type=Path, default=HISTORICO / "pnad_analise_25_65.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/incerteza_descritiva")
    args = parser.parse_args()
    executar(args.entrada, args.entrada_analise, args.saida_dir)
