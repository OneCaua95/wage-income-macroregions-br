#!/usr/bin/env python3
"""Compara a associação da escolaridade no M5 com e sem controles do emprego."""

import argparse
import importlib.util
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.regressao.comparacao_m5 import comparar_especificacoes, controles_emprego, exportar_comparacao

SPEC = importlib.util.spec_from_file_location("_pnad_m5_sensibilidade_emprego", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"
CONTROLES_EMPREGO = controles_emprego(M5)


def executar(entrada, saida, batch_size=100_000):
    relatorio = comparar_especificacoes(
        M5, entrada, batch_size, excluir=CONTROLES_EMPREGO,
        indices_contrastes=(0,), prefixos=("completo", "sem_controles_emprego"),
    )
    relatorio["versao"] = "sensibilidade-emprego-m5-1.0"
    relatorio["controles_retirados"] = sorted(CONTROLES_EMPREGO)
    relatorio["notas"] = [
        "O ajuste reduzido retira ocupação, setor de atividade, setor público e formalidade, incluindo as interações da escolaridade com setor público e formalidade. As duas versões usam os mesmos registros, pesos V1028 e efeitos fixos de UF, ano e trimestre.",
        "A mudança compara a estimativa sem os controles do emprego com a do M5 completo. Os IC95% pontuais tratam o perfil de referência como fixo e consideram a covariância entre os ajustes. A comparação descreve a sensibilidade da associação entre escolaridade e salário-hora à escolha dos controles.",
    ]
    exportar_comparacao(saida, "sensibilidade_emprego_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/sensibilidade_emprego_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
