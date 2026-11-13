#!/usr/bin/env python3
"""IC95% de Gini, Theil, P90/P10 e participação dos 10% maiores salários."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA
from pnad_pipeline.analise.distribuicao_inferencia import intervalos_desigualdade
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao

SPEC = importlib.util.spec_from_file_location("_m5_ic_desigualdade", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)
HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida):
    origem = entrada.stat()
    parquet = M5.pq.ParquetFile(entrada)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") != b"true":
        raise ValueError("Use a base completa para registrar o desenho antes do domínio.")
    periodos = set()
    for batch in parquet.iter_batches(batch_size=100_000, columns=["Ano", "Trimestre"]):
        periodos.update(batch.to_pandas().drop_duplicates().itertuples(index=False, name=None))
    dataset = ds.dataset(entrada)
    rows, n = [], 0
    for ano, trimestre in sorted(periodos):
        batch = dataset.to_table(columns=M5.RAW_COLUMNS, filter=(ds.field("Ano") == ano) & (ds.field("Trimestre") == trimestre))
        desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
        desenho.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        df["Salario_Hora"] = np.exp(df.y.to_numpy(dtype=float))
        n += len(df)
        for grupo, sub in [("Brasil", df), *df.groupby("Regiao", observed=True)]:
            for row in intervalos_desigualdade(sub, desenho):
                rows.append({"trimestre": f"{ano}T{trimestre}", "grupo": grupo, "n_salarios": len(sub),
                             "graus_liberdade": desenho.graus_liberdade, **row})
        print(f"{ano}T{trimestre}: quatro indicadores com IC95%", flush=True)
    fim = entrada.stat()
    if (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento.")
    relatorio = {
        "status": "ok", "versao": "incerteza-desigualdade-1.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{rows[0]['trimestre']} a {rows[-1]['trimestre']}", "n_observacoes": n,
        "ponderacao": "V1028", "n_periodos": len(periodos), "rows": rows,
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "notas": [
            "A amostra segue os indicadores históricos de desigualdade do M5. Cada trimestre usa o cadastro completo antes do domínio e V1028, com linearização por UPA dentro de estrato e aproximação com reposição.",
            "Gini, Theil e participação do topo usam funções de influência e intervalos t pontuais. Os limites são restritos ao suporte de cada indicador. A participação do topo usa exatamente 10% da massa ponderada e preserva empates na fronteira.",
            "O IC de P90/P10 combina limites simultâneos de Woodruff a 97,5% para P10 e P90, com Bonferroni. É um intervalo conservador e não pressupõe independência entre os dois quantis. Limites sem suporte são indisponíveis.",
            "Os intervalos se referem a cada indicador e trimestre. Comparações entre grupos exigem a covariância correspondente.",
        ],
    }
    exportar_comparacao(saida, "incerteza_desigualdade", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/incerteza_desigualdade")
    args = parser.parse_args()
    executar(args.entrada, args.saida_dir)
