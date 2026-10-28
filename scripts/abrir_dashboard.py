#!/usr/bin/env python3
"""Inicializador do dashboard; pode ser executado a partir de qualquer pasta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pnad_pipeline/src"))
from pnad_pipeline.dashboard.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
