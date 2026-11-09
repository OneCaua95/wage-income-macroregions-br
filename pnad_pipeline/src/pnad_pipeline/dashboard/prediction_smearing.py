"""Correção global de Duan: calibração 2023, decisão 2024, relato 2025."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from .data import ROOT, Unavailable, clean, read_json
from .prediction_diagnostics import build_diagnostics, metrics, sample_fingerprint

CRITERION = "Recomendar Duan se o RMSE em reais diminuir e o viés absoluto não aumentar em 2024; MAE e demais métricas são diagnósticos."
REFERENCE = "https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017"


def fit_factor(y_log, pred_log):
    """Média de exp(resíduo), sem truncar resíduos ou ajustar pelo teste."""
    y_log, pred_log = np.asarray(y_log, dtype=float), np.asarray(pred_log, dtype=float)
    if y_log.ndim != 1 or y_log.shape != pred_log.shape or len(y_log) < 100:
        raise ValueError("A calibração requer ao menos 100 observações e previsões correspondentes.")
    if not np.isfinite(y_log).all() or not np.isfinite(pred_log).all():
        raise ValueError("A calibração requer valores finitos.")
    # O estimador deve corresponder à conversão base efetivamente servida.
    factor = float(np.exp(logsumexp(y_log - np.clip(pred_log, -20, 20)) - np.log(len(y_log))))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("O fator de Duan não é positivo e finito.")
    return factor


def recommend(original, corrected):
    return bool(corrected["rmse_salario_hora"] < original["rmse_salario_hora"] - 1e-9
                and abs(corrected["vies_salario_hora"]) <= abs(original["vies_salario_hora"]) + 1e-9)


def freeze_validation(calibration, calibration_predictions, validation, validation_predictions, manifest):
    for frame, year in [(calibration, "2023"), (validation, "2024")]:
        if len(frame) < 100 or not frame.Ano.astype(str).eq(year).all():
            raise ValueError(f"O protocolo exige ao menos 100 observações exclusivamente de {year}.")
    factor = fit_factor(calibration.log_salario_hora, calibration_predictions)
    original = metrics(validation.log_salario_hora, validation_predictions)
    corrected = metrics(validation.log_salario_hora, validation_predictions, smearing_factor=factor)
    accepted = recommend(original, corrected)
    return clean({
        "version": "duan-1.0", "run_id": manifest["run_id"], "modelo_sha256": manifest["sha256"],
        "congelado_em": datetime.now(timezone.utc).isoformat(), "criterio": CRITERION,
        "calibracao": {"periodo": "2023", "n": len(calibration), "fator": factor, "amostra_sha256": sample_fingerprint(calibration)},
        "validacao": {"periodo": "2024", "n": len(validation), "original": original, "duan": corrected,
                      "amostra_sha256": sample_fingerprint(validation)},
        "recomendada": "duan" if accepted else "original", "adotar_duan": accepted,
        "formula": "S = média[exp(log observado − log da conversão base)] em 2023; previsto corrigido = exp(log previsto) × S.",
        "teste_usado_na_decisao": False,
    })


def save_artifact(payload, folder, prefix):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    content = json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    name = f"{prefix}_{payload['run_id']}_{digest[:12]}.json"
    path = folder / name
    temp = path.with_suffix(".tmp")
    temp.write_bytes(content)
    os.replace(temp, path)
    return {"arquivo": name, "sha256": digest, "version": "duan-1.0"}


def verify_original_validation(frame, predictions, manifest):
    actual = metrics(frame.log_salario_hora, predictions)
    expected = manifest["metricas_validacao"]
    keys = [f"{metric}_{scale}" for scale in ("log", "salario_hora") for metric in ("mae", "rmse", "r2")]
    if actual["n"] != manifest["amostras"]["validacao"] or any(
        not np.isclose(actual[key], expected[key], rtol=1e-9, atol=1e-10) for key in keys
    ):
        raise ValueError("A amostra reconstruída não reproduz as métricas da validação original.")


def evaluate_existing(folder=None):
    from . import prediction
    folder = Path(folder) if folder is not None else prediction.ARTIFACT_DIR
    manifest_path = folder / "manifesto.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    source = (ROOT / manifest["entrada"]["arquivo"]).resolve()
    if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or source.stat().st_size != manifest["entrada"]["bytes"]:
        raise ValueError("Restaure a base da execução original indicada no manifesto.")
    model = prediction.validated_pipeline(manifest, folder)
    print("Etapa 1: reconstruindo a validação original, sem ler o teste de 2025", flush=True)
    _, frames = prediction.sample_temporal(source, max_eval=manifest["amostras"]["validacao"], seed=manifest["seed"], splits={"validacao"})
    validation_all = frames["validacao"]
    with threadpool_limits(limits=4):
        predicted = model.predict(validation_all)
    verify_original_validation(validation_all, predicted, manifest)
    calibration_mask = validation_all.Ano.astype(str).eq("2023").to_numpy()
    validation_mask = validation_all.Ano.astype(str).eq("2024").to_numpy()
    if not (calibration_mask | validation_mask).all():
        raise ValueError("A validação original contém anos fora de 2023–2024.")
    validation = validation_all.loc[validation_mask]
    validation_predictions = predicted[validation_mask]
    decision = freeze_validation(validation_all.loc[calibration_mask], predicted[calibration_mask], validation, validation_predictions, manifest)
    frozen_reference = save_artifact(decision, folder, "decisao_duan")
    print(f"Decisão persistida ANTES do teste: fator {decision['calibracao']['fator']:.8f}; versão {decision['recomendada']}; SHA {frozen_reference['sha256']}", flush=True)

    # Somente após persistir a decisão, reconstrói e avalia o teste.
    print("Etapa 2: relatando as duas conversões no teste de 2025, com fator e decisão fixos", flush=True)
    _, test_frames = prediction.sample_temporal(source, max_eval=manifest["amostras"]["teste"], seed=manifest["seed"], splits={"teste"})
    test = test_frames["teste"]
    with threadpool_limits(limits=4):
        test_predictions = model.predict(test)
    factor = decision["calibracao"]["fator"]
    validation_results = {
        key: build_diagnostics(validation, validation_predictions, manifest, smearing_factor=f,
                               period="2024", verify_original=False)
        for key, f in [("original", 1.0), ("duan", factor)]
    }
    test_results = {key: build_diagnostics(test, test_predictions, manifest, smearing_factor=f)
                    for key, f in [("original", 1.0), ("duan", factor)]}
    comparison = []
    for period, results in [("2024", validation_results), ("2025", test_results)]:
        for variant, result in results.items():
            comparison.append({"periodo": period, "uso": "Decisão da conversão" if period == "2024" else "Relato posterior",
                               "conversao": variant, **result["summary"]})
    payload = clean({
        "status": "ok", "version": "duan-1.0", "run_id": manifest["run_id"], "modelo": manifest["modelo"], "modelo_sha256": manifest["sha256"],
        "processado_em": datetime.now(timezone.utc).isoformat(), "decisao": decision, "decisao_arquivo": frozen_reference,
        "ponderacao": "Não ponderado", "data_base": manifest["data_base"], "comparacao": comparison,
        "validation": validation_results, "test": test_results, "referencia_metodologica": REFERENCE,
        "limites": ["O fator global pressupõe que a distribuição dos resíduos de calibração seja transportável; heterogeneidade pode manter viés por grupo.",
                    "O pipeline base já foi selecionado usando 2023–2024; esta é uma comparação retrospectiva condicionada ao modelo fixo.",
                    "O teste de 2025 já teve diagnósticos inspecionados; a comparação não constitui uma nova avaliação independente.",
                    "O fator é definido exclusivamente em 2023, a escolha em 2024; não há ajuste ou escolha com dados de 2025.",
                    "RMSE, MAE e viés podem melhorar em direções diferentes: a escolha prioriza RMSE e viés absoluto em reais.",
                    "Sem ponderação, interpretação causal ou intervalos de previsão; valores reais a preços de janeiro/2026."],
    })
    reference = save_artifact(payload, folder, "comparacao_duan")
    if manifest_path.read_bytes() != original_manifest:
        raise ValueError("O manifesto mudou durante a avaliação. Execute novamente para a versão atual.")
    manifest["duan"] = reference
    temp = folder / "manifesto_duan.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, manifest_path)
    print(f"Concluído: {len(comparison)} avaliações; recomendada {decision['recomendada']}", flush=True)
    return payload


@lru_cache(maxsize=4)
def read_artifact(path, stamp, expected_sha):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise Unavailable("O arquivo de Duan não corresponde ao hash registrado. Execute novamente o script 23.")
    return json.loads(content)


def artifact_path(folder, reference):
    path = folder / reference["arquivo"]
    if path.parent.resolve() != folder.resolve() or path.suffix != ".json":
        raise Unavailable("Caminho do artefato de Duan inválido.")
    return path


def get_smearing(manifest=None, folder=None):
    from . import prediction
    folder = Path(folder) if folder is not None else prediction.ARTIFACT_DIR
    manifest = read_json(folder / "manifesto.json") if manifest is None else manifest
    reference = manifest.get("duan")
    unavailable = {"status": "unavailable", "message": "Comparação de Duan indisponível para este modelo. Execute python pnad_pipeline/scripts/23_avaliar_duan.py."}
    if not reference:
        return unavailable
    path = artifact_path(folder, reference)
    if not path.is_file():
        return unavailable
    payload = read_artifact(path, path.stat().st_mtime_ns, reference["sha256"])
    if payload.get("run_id") != manifest.get("run_id") or payload.get("modelo_sha256") != manifest.get("sha256"):
        return unavailable
    frozen_reference = payload["decisao_arquivo"]
    frozen_path = artifact_path(folder, frozen_reference)
    if not frozen_path.is_file():
        return unavailable
    frozen = read_artifact(frozen_path, frozen_path.stat().st_mtime_ns, frozen_reference["sha256"])
    if payload["decisao"] != frozen or frozen.get("teste_usado_na_decisao") is not False:
        raise Unavailable("A decisão de Duan difere do registro congelado.")
    return clean({**payload, "provenance": {
        "conjunto": "Comparação Duan da execução " + manifest["run_id"], "ponderacao": "Não ponderado", "periodo": "2024 / 2025",
        "versao": payload["version"], "run_id": manifest["run_id"], "modelo_sha256": manifest["sha256"],
        "decisao_sha256": frozen_reference["sha256"], "fator_duan": frozen["calibracao"]["fator"], "filtros": {},
    }})
