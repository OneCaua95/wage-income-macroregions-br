#!/usr/bin/env python3
"""Compara intervalos conformais nos microdados da PNAD de 2025 e 2026."""
from __future__ import annotations

import argparse
import runpy
import sys
import zipfile
from calendar import monthrange
from functools import partial
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnad_pipeline.analise import preparar_variaveis_analise
from pnad_pipeline.dashboard.data import ROOT
from pnad_pipeline.dashboard.prediction import sample_frame
from pnad_pipeline.dashboard.prediction_cqr import (
    ARTIFACT_DIR, HOLDOUT_SEED, TAIL_HOLDOUT_SEED, compare_intervals, compare_separate_tails, sha256_file,
)
from pnad_pipeline.extracao.coleta import processar_periodo
from pnad_pipeline.extracao.ipca import get_ipca_bc

IBGE = ("https://ftp.ibge.gov.br/Trabalho_e_Rendimento/"
        "Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados")
LAYOUT_URL = f"{IBGE}/Documentacao/Dicionario_e_input_20221031.zip"
AVAILABILITY_URL = f"{IBGE}/Documentacao/Variaveis_PNADC_Trimestral.xls"
REQUIRED_FIELDS = {"Ano", "Trimestre", "UF", "UPA", "Estrato", "V1028", "V2007", "V2009", "V2010", "VD3005",
                   "VD4002", "VD4009", "VD4010", "VD4012", "VD4016", "VD4031"}


def load_holdout(output, n, *, quarter=1):
    seed = {1: HOLDOUT_SEED, 2: TAIL_HOLDOUT_SEED}[quarter]
    period = f"2026T{quarter}"
    archive_name = f"PNADC_{quarter:02d}2026.zip"
    zip_url = f"{IBGE}/2026/{archive_name}"
    ipca_month = quarter * 3 - 1
    ipca_last_day = monthrange(2026, ipca_month)[1]
    ipca_url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"
                f"&dataInicial=01/02/2026&dataFinal={ipca_last_day:02d}/{ipca_month:02d}/2026")
    print(f"Abrindo a amostra reservada de {period}", flush=True)
    official = runpy.run_path(str(ROOT / "pnad_pipeline/scripts/10_validar_dados_reais.py"))
    raw = ROOT / "pnad_pipeline/data/raw"
    archive = raw / archive_name
    layout = raw / "Dicionario_e_input_20221031.zip"
    availability = raw / "Variaveis_PNADC_Trimestral_2026.xls"
    official["baixar"](zip_url, archive)
    official["baixar"](LAYOUT_URL, layout)
    if not availability.is_file():
        temp = availability.with_suffix(".part")
        with urlopen(AVAILABILITY_URL, timeout=45) as response:
            temp.write_bytes(response.read())
        temp.replace(availability)
    fields = pd.read_excel(availability)
    available = set(fields.loc[fields[f"{quarter}º tri/2026"].eq("X"), "Variável"])
    if missing := REQUIRED_FIELDS - available:
        raise ValueError(f"Variáveis do modelo indisponíveis em {period}: {sorted(missing)}")
    dictionary = output / "dicionario_posicoes.xls"
    with zipfile.ZipFile(layout) as package:
        dictionary.write_bytes(package.read("dicionario_PNADC_microdados_trimestral.xls"))
    ipca = get_ipca_bc(ipca_url)
    base, _, factor = processar_periodo(2026, quarter, ipca, extractor=official["ExtratorZipOficial"](archive, dictionary))
    if not base.Ano.eq(2026).all() or not base.Trimestre.eq(quarter).all() or base.UF.nunique() != 27:
        raise ValueError("Microdados incompatíveis com o período e a cobertura de UFs esperados.")
    analytic = preparar_variaveis_analise(base)
    path = output / f"pnad_analise_{period}.parquet"
    temp = path.with_suffix(".parquet.part")
    analytic.to_parquet(temp, index=False, engine="pyarrow")
    temp.replace(path)
    eligible = int(analytic.Salario_Hora.gt(0).sum())
    del base, analytic
    sample = sample_frame(path, (ds.field("Ano") == 2026) & (ds.field("Trimestre") == quarter), n, seed)
    return sample, {"url_fonte": zip_url, "sha256_zip": sha256_file(archive), "url_dicionario": LAYOUT_URL,
                    "sha256_dicionario": sha256_file(dictionary), "url_disponibilidade": AVAILABILITY_URL,
                    "sha256_disponibilidade": sha256_file(availability), "url_ipca": ipca_url, "fator_ipca": factor,
                    "entrada": {"arquivo": path.name, "bytes": path.stat().st_size},
                    "n_elegiveis": eligible, "seed": seed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artefatos-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--referencia-dir", type=Path)
    parser.add_argument("--por-cauda", action="store_true", help="Reutiliza o CQR existente e testa ajustes separados em 2026T2.")
    args = parser.parse_args()
    if args.por_cauda:
        report = compare_separate_tails(partial(load_holdout, quarter=2), args.artefatos_dir, args.referencia_dir)
    else:
        report = compare_intervals(load_holdout, args.artefatos_dir, args.referencia_dir)
    for period, result in report["periodos"].items():
        for method, metrics in result["geral"].items():
            print(f"{period} · {method}: cobertura {metrics['cobertura']:.2%}, largura média R$ {metrics['largura_media']:.2f}/h")
