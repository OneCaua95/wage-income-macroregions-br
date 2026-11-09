#!/usr/bin/env python3
"""Calibra Duan em 2023, decide em 2024 e relata a comparação em 2025."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnad_pipeline.dashboard.prediction import ARTIFACT_DIR
from pnad_pipeline.dashboard.prediction_smearing import evaluate_existing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artefatos-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    evaluate_existing(args.artefatos_dir)
