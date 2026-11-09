#!/usr/bin/env python3
"""Ajusta nova execução temporal e avalia intervalos com meta de 90%."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnad_pipeline.dashboard.data import HISTORICO_LEGADO
from pnad_pipeline.dashboard.prediction_intervals import ARTIFACT_DIR, refresh_evaluation, train_intervals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO_LEGADO)
    parser.add_argument("--artefatos-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--max-treino", type=int, default=150_000)
    parser.add_argument("--max-avaliacao", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reavaliar", action="store_true", help="Atualiza o relatório da execução existente com a base, o modelo e a calibração registrados.")
    args = parser.parse_args()
    if args.reavaliar:
        result = refresh_evaluation(args.artefatos_dir)
        error = result["diagnostico"]["summary"]["mdape_salario_hora"]
        print(f"Relatório atualizado: {result['run_id']} · erro percentual absoluto mediano {error:.2%}")
    else:
        train_intervals(args.entrada, args.artefatos_dir, args.max_treino, args.max_avaliacao, args.seed)
