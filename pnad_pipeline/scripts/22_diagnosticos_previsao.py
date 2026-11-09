#!/usr/bin/env python3
"""Gera diagnósticos agregados de teste para o modelo operacional já exportado."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnad_pipeline.dashboard.prediction import ARTIFACT_DIR
from pnad_pipeline.dashboard.prediction_diagnostics import generate_existing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artefatos-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    generate_existing(args.artefatos_dir)
