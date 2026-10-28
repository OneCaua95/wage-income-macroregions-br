#!/usr/bin/env python3
"""Seleciona na validação temporal e exporta modelo + pré-processamento.

Os resultados vão para previsao_perfis/, preservando o benchmark histórico.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pnad_pipeline.dashboard.prediction import train, ARTIFACT_DIR, PROCESSED

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada",type=Path,default=PROCESSED/"pnad_analise_25_65.parquet")
    p.add_argument("--saida-dir",type=Path,default=ARTIFACT_DIR)
    p.add_argument("--max-treino",type=int,default=150000)
    p.add_argument("--max-avaliacao",type=int,default=50000)
    p.add_argument("--seed",type=int,default=42)
    a = p.parse_args()
    train(a.entrada,a.saida_dir,a.max_treino,a.max_avaliacao,a.seed)
