"""Compara intervalos conformais na mesma amostra de cada período."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pyarrow.dataset as ds
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from . import prediction, prediction_intervals
from .data import PROCESSED, ROOT, Unavailable, clean, read_json
from .prediction_diagnostics import sample_fingerprint
from .prediction_intervals import TARGET, bounds_metrics, checked_read, conformal_order, evaluate_bounds, get_intervals, now, save

ARTIFACT_DIR = PROCESSED / "previsao_adaptativa"
VERSION = "comparacao-intervalos-1.0"
HOLDOUT = "2026T1"
HOLDOUT_SEED = 202601
TAIL_HOLDOUT = "2026T2"
TAIL_HOLDOUT_SEED = 202602
TAIL_ERROR = .05
TAIL_VERSION = "comparacao-intervalos-por-cauda-1.0"
REFERENCE = "https://arxiv.org/abs/1905.03222"
QUANTILES = (.05, .95)
MODEL_PARAMS = {"max_iter": 250, "learning_rate": .08, "max_leaf_nodes": 31,
                "min_samples_leaf": 20, "l2_regularization": 1., "early_stopping": False}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile_bounds(models, frame):
    with threadpool_limits(limits=4):
        first, second = (np.clip(model.predict(frame), -20, 20) for model in models)
    return np.minimum(first, second), np.maximum(first, second), int(np.sum(first > second))


def calibrated_bounds(lower, upper, correction, upper_correction=None):
    if upper_correction is None:
        upper_correction = correction
    adjusted_lower = lower - correction
    adjusted_upper = upper + upper_correction
    midpoint = (adjusted_lower + adjusted_upper) / 2
    return np.minimum(adjusted_lower, midpoint), np.maximum(adjusted_upper, midpoint)


def compare_bounds(frame, bounds_by_method, period):
    evaluations = {method: evaluate_bounds(frame, *bounds, period) for method, bounds in bounds_by_method.items()}
    bands = {method: [row for row in result["rows"] if row["dimensao"] == "faixa_salarial"]
             for method, result in evaluations.items()}
    rows = []
    for index, row in enumerate(bands["atual"]):
        labels = {key: row[key] for key in ("grupo", "ordem", "limite_inferior", "limite_superior", "n", "status")}
        rows.append({**labels, **{method: group_rows[index] for method, group_rows in bands.items()}})
    return {"periodo": period, "amostra_sha256": evaluations["atual"]["amostra_sha256"],
            "geral": {method: result["summary"] for method, result in evaluations.items()}, "faixas": rows}


def historical_sample(path, manifest, stage, expression, offset):
    frame = prediction.sample_frame(path, expression, manifest["amostras"][stage], manifest["seed"] + offset)
    if sample_fingerprint(frame) != manifest["amostras_sha256"][stage]:
        raise ValueError(f"A amostra histórica de {stage} difere da execução registrada.")
    return frame


def load_baseline(baseline_dir):
    if baseline_dir is None:
        baseline_dir = prediction_intervals.ARTIFACT_DIR
    baseline = get_intervals(baseline_dir)
    if baseline["status"] != "ok":
        raise Unavailable("Gere o intervalo atual com o script 24 antes da comparação.")
    manifest = baseline["manifesto"]
    source = (ROOT / manifest["entrada"]["arquivo"]).resolve()
    if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
        raise ValueError("Restaure a base histórica indicada pelo intervalo atual.")
    base_model = prediction.validated_pipeline(manifest, baseline_dir)
    identity = {"run_id": baseline["run_id"], "modelo_sha256": baseline["modelo_sha256"],
                "calibracao_sha256": baseline["calibracao_arquivo"]["sha256"],
                "fator": baseline["decisao"]["fator_aplicado"], "raio_log": baseline["raio_log"]}
    return baseline, source, base_model, identity


def baseline_bounds(model, identity, frame):
    with threadpool_limits(limits=4):
        center = np.clip(model.predict(frame), -20, 20) + np.log(identity["fator"])
    radius = identity["raio_log"]
    return center - radius, center + radius


def compare_intervals(load_holdout, output=ARTIFACT_DIR, baseline_dir=None):
    baseline, source, base_model, baseline_identity = load_baseline(baseline_dir)
    manifest = baseline["manifesto"]
    output = Path(output)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    protocol = {"version": VERSION, "run_id": run, "fixado_em": now(), "atual": baseline_identity,
                "meta": TARGET, "quantis": QUANTILES, "modelo": "HistGradientBoosting em log",
                "parametros": MODEL_PARAMS, "sklearn": sklearn.__version__, "seed": manifest["seed"],
                "treino": "2013–2021", "calibracao": "2024", "referencia": "2025", "teste_final": HOLDOUT,
                "seed_teste_final": HOLDOUT_SEED, "n_teste_final": manifest["amostras"]["teste"],
                "escore": "max(P5 − observado, observado − P95) em log após ordenar os limites",
                "correcao": "Escore de ordem teto((n+1)×0,90). Se a contração inverter os limites, usar o ponto médio.",
                "comparacao": "Cobertura e largura média e mediana no geral e nos decis de salário observado de cada período.",
                "referencia_metodologica": REFERENCE}
    protocol_ref = save(protocol, output, "protocolo")
    train = historical_sample(source, manifest, "treino", (ds.field("Ano") >= 2013) & (ds.field("Ano") <= 2021), 0)
    legacy = prediction.legacy_module()
    models = []
    with threadpool_limits(limits=4):
        for quantile in QUANTILES:
            print(f"Ajustando quantil {quantile:.0%} em 2013–2021", flush=True)
            model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile,
                                                  random_state=manifest["seed"], **MODEL_PARAMS)
            pipeline = legacy.build_pipeline(model, legacy.BASE_CATEGORICAL + legacy.FIXED_EFFECTS, dense=True)
            pipeline.fit(train, train[legacy.TARGET])
            models.append(pipeline)
    del train
    model_path = output / f"quantis_{run}.joblib"
    joblib.dump(models, model_path)
    calibration = historical_sample(source, manifest, "calibracao", ds.field("Ano") == 2024, 5)
    lower, upper, crossings = quantile_bounds(models, calibration)
    y = calibration.log_salario_hora.to_numpy(dtype=float)
    correction, rank = conformal_order(np.maximum(lower - y, y - upper))
    frozen = {"version": VERSION, "run_id": run, "fixado_em": now(), "protocolo": protocol_ref,
              "modelo": {"arquivo": model_path.name, "sha256": sha256_file(model_path)},
              "correcao_log": correction, "ordem": rank, "n": len(y), "cruzamentos_ordenados": crossings,
              "amostra_sha256": manifest["amostras_sha256"]["calibracao"]}
    calibration_ref = save(frozen, output, "calibracao")
    del calibration
    print(f"Ajuste congelado antes dos dois testes: {calibration_ref['sha256']}", flush=True)

    def evaluate(frame, period):
        lower, upper, crossings = quantile_bounds(models, frame)
        bounds = {"atual": baseline_bounds(base_model, baseline_identity, frame),
                  "adaptativa": calibrated_bounds(lower, upper, correction)}
        result = compare_bounds(frame, bounds, period)
        result["cruzamentos_ordenados"] = crossings
        return result

    test = historical_sample(source, manifest, "teste", ds.field("Ano") == 2025, 2)
    retrospective = evaluate(test, "2025")
    for metric, expected in baseline["test"]["summary"].items():
        if not np.isclose(retrospective["geral"]["atual"][metric], expected, rtol=1e-9, atol=1e-10):
            raise ValueError("A comparação não reproduziu o intervalo atual de 2025.")
    del test
    holdout, provenance = load_holdout(output, protocol["n_teste_final"])
    if (len(holdout) != protocol["n_teste_final"] or not holdout.Ano.astype(str).eq("2026").all()
            or not holdout.Trimestre.astype(str).eq("1").all()):
        raise ValueError("A amostra final deve conter o tamanho registrado e somente 2026T1.")
    final = evaluate(holdout, HOLDOUT)
    final["fonte"] = provenance
    report = {"status": "ok", "version": VERSION, "run_id": run, "meta": TARGET, "atual": baseline_identity,
              "protocolo": protocol, "protocolo_arquivo": protocol_ref, "calibracao": frozen,
              "calibracao_arquivo": calibration_ref, "processado_em": now(), "data_base": manifest["data_base"],
              "ponderacao": manifest["ponderacao"], "periodos": {"2025": retrospective, HOLDOUT: final}}
    reference = save(report, output, "comparacao")
    write_pointer(output / "manifesto.json", run, reference)
    return clean(report)


def write_pointer(path, run, reference):
    temp = path.with_suffix(f".{run}.tmp")
    temp.write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def read_comparison(folder, reference):
    report = checked_read(folder, reference)
    for key in ("protocolo", "calibracao"):
        artifact = checked_read(folder, report[f"{key}_arquivo"])
        if artifact != report[key] or artifact["run_id"] != report["run_id"]:
            raise Unavailable("A comparação difere do protocolo ou da calibração registrada.")
    if (report["calibracao"]["protocolo"] != report["protocolo_arquivo"]
            or report["atual"] != report["protocolo"]["atual"]):
        raise Unavailable("A comparação difere da execução de referência registrada.")
    return report


def compare_separate_tails(load_holdout, output=ARTIFACT_DIR, baseline_dir=None):
    output = Path(output)
    reference = read_json(output / "manifesto.json")
    if not reference:
        raise Unavailable("Gere o CQR com o script 25 antes de calibrar por cauda.")
    common = read_comparison(output, reference)
    baseline, source, base_model, identity = load_baseline(baseline_dir)
    manifest = baseline["manifesto"]
    if common["atual"] != identity or common["calibracao"]["amostra_sha256"] != manifest["amostras_sha256"]["calibracao"]:
        raise ValueError("O CQR registrado usa outra referência ou amostra de calibração.")
    model_ref = common["calibracao"]["modelo"]
    models = prediction.validated_pipeline({**model_ref, "sklearn": common["protocolo"]["sklearn"]}, output)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    protocol = {"version": TAIL_VERSION, "run_id": run, "fixado_em": now(), "atual": identity,
                "comparacao_base": reference, "execucao_cqr": common["run_id"], "meta": TARGET,
                "meta_erro_por_lado": TAIL_ERROR, "calibracao": "2024", "teste_final": TAIL_HOLDOUT,
                "seed_teste_final": TAIL_HOLDOUT_SEED, "n_teste_final": manifest["amostras"]["teste"],
                "metodos": ["atual", "adaptativa", "por_cauda"], "referencia_metodologica": REFERENCE,
                "comparacao": "Cobertura, largura e erros por lado no geral e nos decis da mesma amostra."}
    protocol_ref = save(protocol, output, "protocolo_caudas")
    calibration = historical_sample(source, manifest, "calibracao", ds.field("Ano") == 2024, 5)
    lower, upper, _ = quantile_bounds(models, calibration)
    y = calibration.log_salario_hora.to_numpy(dtype=float)
    common_correction = common["calibracao"]["correcao_log"]
    reproduced, rank = conformal_order(np.maximum(lower - y, y - upper))
    if rank != common["calibracao"]["ordem"] or not np.isclose(reproduced, common_correction, rtol=1e-9, atol=1e-12):
        raise ValueError("A amostra não reproduziu o ajuste único do CQR registrado.")
    lower_correction, lower_rank = conformal_order(lower - y, 1 - TAIL_ERROR)
    upper_correction, upper_rank = conformal_order(y - upper, 1 - TAIL_ERROR)
    frozen = {"version": TAIL_VERSION, "run_id": run, "fixado_em": now(), "protocolo": protocol_ref,
              "modelo": model_ref, "amostra_sha256": common["calibracao"]["amostra_sha256"], "n": len(y),
              "correcao_inferior_log": lower_correction, "correcao_superior_log": upper_correction,
              "ordem_inferior": lower_rank, "ordem_superior": upper_rank,
              "metricas_calibracao": bounds_metrics(y, *calibrated_bounds(lower, upper, lower_correction, upper_correction))}
    calibration_ref = save(frozen, output, "calibracao_caudas")
    del calibration
    print(f"Ajustes por cauda congelados antes de 2026T2: {calibration_ref['sha256']}", flush=True)

    holdout, provenance = load_holdout(output, protocol["n_teste_final"])
    if (len(holdout) != protocol["n_teste_final"] or not holdout.Ano.astype(str).eq("2026").all()
            or not holdout.Trimestre.astype(str).eq("2").all()):
        raise ValueError("A amostra final deve conter o tamanho registrado e somente 2026T2.")
    lower, upper, crossings = quantile_bounds(models, holdout)
    bounds = {"atual": baseline_bounds(base_model, identity, holdout),
              "adaptativa": calibrated_bounds(lower, upper, common_correction),
              "por_cauda": calibrated_bounds(lower, upper, lower_correction, upper_correction)}
    result = compare_bounds(holdout, bounds, TAIL_HOLDOUT)
    result.update(fonte=provenance, cruzamentos_ordenados=crossings)
    report = {"status": "ok", "version": TAIL_VERSION, "run_id": run, "atual": identity, "meta": TARGET,
              "protocolo": protocol, "protocolo_arquivo": protocol_ref, "calibracao": frozen,
              "calibracao_arquivo": calibration_ref, "processado_em": now(), "periodos": {TAIL_HOLDOUT: result}}
    report_ref = save(report, output, "comparacao_caudas")
    if read_json(output / "manifesto.json") != reference:
        raise ValueError("A execução CQR mudou durante a comparação. Execute novamente com a referência atual.")
    write_pointer(output / "manifesto_caudas.json", run, {"referencia": reference, "relatorio": report_ref})
    return clean(report)


def get_comparison(folder=None):
    folder = ARTIFACT_DIR if folder is None else Path(folder)
    reference = read_json(folder / "manifesto.json")
    if not reference:
        return {"status": "unavailable", "message": "Execute o script 25 para gerar a comparação de intervalos."}
    report = read_comparison(folder, reference)
    extension = read_json(folder / "manifesto_caudas.json")
    if extension.get("referencia") != reference:
        return report
    tails = read_comparison(folder, extension["relatorio"])
    if (tails["protocolo"]["comparacao_base"] != reference or tails["atual"] != report["atual"]
            or tails["calibracao"]["modelo"] != report["calibracao"]["modelo"]):
        raise Unavailable("Os ajustes por cauda não correspondem ao CQR registrado.")
    metadata = {key: tails[key] for key in ("run_id", "protocolo", "protocolo_arquivo", "calibracao", "calibracao_arquivo")}
    return {**report, "periodos": {**report["periodos"], **tails["periodos"]}, "por_cauda": metadata}
