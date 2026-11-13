#!/usr/bin/env python3
"""Compara os contrastes do M5 na mesma amostra com e sem efeitos fixos de UF."""

import argparse
import importlib.util
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.regressao.comparacao_m5 import comparar_especificacoes, exportar_comparacao

SPEC = importlib.util.spec_from_file_location("_pnad_m5_sensibilidade_uf", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida, batch_size=100_000):
    relatorio = comparar_especificacoes(
        M5, entrada, batch_size, excluir={f"UF_{uf}" for uf in M5.UFS[1:]},
        indices_contrastes=(0, 1, 2), prefixos=("com_uf", "sem_uf"),
    )
    relatorio["versao"] = "sensibilidade-uf-m5-1.0"
    relatorio["notas"] = [
        "As duas versões usam os mesmos registros, pesos V1028 e salário-hora original. A segunda retira apenas os indicadores de UF da regressão. A UF permanece no desenho amostral para identificar estratos e UPAs.",
        "Os contrastes usam o mesmo perfil ponderado de referência, tratado como fixo nos IC95% pontuais. A mudança corresponde a sem UF menos com UF e usa a covariância entre os ajustes na linearização do desenho.",
    ]
    exportar_comparacao(saida, "sensibilidade_uf_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/sensibilidade_uf_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
