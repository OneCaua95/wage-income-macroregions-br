#!/usr/bin/env python3
"""Incorpora resultados observados, proveniência e JUnit em um HTML autônomo."""

from __future__ import annotations

import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pnad_pipeline.analise.amostragem import COLUNAS_DESENHO, exigir_colunas_desenho

MODELS = {"geral": "geral", "Norte": "norte", "Nordeste": "nordeste", "Sudeste": "sudeste", "Sul": "sul", "Centro-Oeste": "centro_oeste"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def result_set(folder):
    return {
        "summary": read_json(folder / "resumo_inferencia_m5.json"),
        "datasets": {k: list(csv.DictReader((folder / f"inferencial_m5_{v}.csv").open(encoding="utf-8-sig", newline=""))) for k, v in MODELS.items()},
        "recorded_at": datetime.fromtimestamp((folder / "resumo_inferencia_m5.json").stat().st_mtime, timezone(timedelta(hours=-3))).strftime("%d/%m/%Y %H:%M · Brasília"),
    }


def junit(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    cases = []
    for case in root.iter("testcase"):
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        item = failure if failure is not None else error
        cases.append({"name": case.attrib["name"], "module": case.attrib.get("classname", ""),
                      "seconds": float(case.attrib.get("time", 0)),
                      "status": "reprovado" if item is not None else "ignorado" if skipped is not None else "aprovado",
                      "message": item.attrib.get("message", "") if item is not None else ""})
    return {"total": len(cases), "passed": sum(c["status"] == "aprovado" for c in cases),
            "failed": sum(c["status"] == "reprovado" for c in cases), "skipped": sum(c["status"] == "ignorado" for c in cases),
            "seconds": sum(float(s.attrib.get("time", 0)) for s in suites), "cases": cases}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    real = ROOT / "data" / "processed" / "validacao_real" / "2025T4"
    historico = ROOT / "data" / "processed" / "modelos_salario_hora"
    anterior = pq.ParquetFile(ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet")
    colunas = anterior.schema_arrow.names
    try:
        exigir_colunas_desenho(colunas)
        guard = "A base contém as colunas do desenho amostral."
    except ValueError as exc:
        guard = "Proteção observada na base real: " + str(exc)
    dados = {
        "validation": read_json(real / "validacao_observada.json"),
        "real": result_set(real / "inferencia"), "historico": result_set(historico),
        "tests": junit(ROOT / "data" / "processed" / "testes_codigo.xml"),
        "audit": {"historical_rows": anterior.metadata.num_rows,
                  "real_rows": pq.ParquetFile(real / "pnad_completa_2025T4.parquet").metadata.num_rows,
                  "present": [c for c in COLUNAS_DESENHO if c in colunas],
                  "missing": [c for c in COLUNAS_DESENHO if c not in colunas], "guard_message": guard},
    }
    dados = clean(dados)
    literal = json.dumps(dados, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "templates" / "relatorio_testes_reais.html").read_text(encoding="utf-8")
    destino = ROOT.parent / "frontend" / "resultados_testes_reais.html"
    destino.write_text(template.replace("__REPORT_DATA__", literal), encoding="utf-8")
    (real / "relatorio_dados.json").write_text(json.dumps(dados, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"HTML gerado: {destino}")
    print(f"Tamanho: {destino.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
