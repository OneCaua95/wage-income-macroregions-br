"""Intervalos split conformal em log, com etapas temporais separadas.

Avaliação retrospectiva: a rotação da PNAD e a mudança temporal não asseguram
permutabilidade. 90% é a meta nominal, não uma garantia futura ou por grupo.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from . import prediction
from .data import PROCESSED, ROOT, Unavailable, clean, file_info, read_json
from .prediction_diagnostics import (
    DIMENSIONS, MIN_GROUP, build_diagnostics, check_test_metrics, metrics, observed_wage_bands, sample_fingerprint,
)
from .prediction_smearing import fit_factor, recommend

ARTIFACT_DIR = PROCESSED / "previsao_intervalos"
VERSION = "intervalos-previsao-1.0"
TARGET = .90
REFERENCE = "https://arxiv.org/abs/2107.07511"
PERIODS = {"treino": "2013–2021", "validacao": "2022", "fator_duan": "2023T1–T2",
           "conversao": "2023T3–T4", "calibracao": "2024", "teste": "2025"}
LIMITS = [
    "Avaliação retrospectiva: o teste de 2025 já foi inspecionado em trabalhos anteriores; não é validação prospectiva independente.",
    "90% é a meta nominal. A garantia marginal split conformal requer permutabilidade entre calibração e novas observações, não assegurada pela separação temporal e pela rotação de domicílios da PNAD.",
    "Não há garantia de cobertura futura, populacional, individual ou por grupo. Resultados não ponderados para ocupados de 25 a 65 anos com renda e horas positivas.",
    "Raio constante em log produz faixas proporcionais ao valor previsto; pode gerar intervalos amplos e cobertura desigual entre grupos.",
    "Anos de 2022 a 2025 são categorias não vistas no treino e são ignorados pelo codificador; não se trata de projeção temporal.",
    "Faixa de previsão de salário individual, distinta de intervalo de confiança de coeficientes ou de médias. Valores a preços de janeiro/2026.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(payload, folder, prefix):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    content = json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    name = f"{prefix}_{payload['run_id']}_{digest[:12]}.json"
    path = folder / name
    temp = path.with_suffix(".tmp")
    temp.write_bytes(content)
    os.replace(temp, path)
    return {"arquivo": name, "sha256": digest, "version": payload["version"]}


def conformal_radius(observed_log, center_log, target=TARGET):
    observed, center = np.asarray(observed_log, dtype=float), np.asarray(center_log, dtype=float)
    if observed.ndim != 1 or center.shape != observed.shape or not len(observed):
        raise ValueError("Calibração exige vetores não vazios de igual tamanho.")
    if not np.isfinite(observed).all() or not np.isfinite(center).all() or not 0 < target < 1:
        raise ValueError("Calibração exige valores finitos e meta entre zero e um.")
    return conformal_order(np.abs(observed - center), target)


def conformal_order(scores, target=TARGET):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 < target < 1:
        raise ValueError("O quantil conformal exige escores finitos e uma meta entre zero e um.")
    rank = math.ceil((len(scores) + 1) * target)
    if rank > len(scores):
        raise ValueError("Amostra insuficiente para um raio finito com esta meta.")
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


def interval_metrics(observed_log, center_log, radius):
    y, center = np.asarray(observed_log, dtype=float), np.asarray(center_log, dtype=float)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Valores de avaliação ou raio inválidos.")
    return bounds_metrics(y, center - radius, center + radius)


def bounds_metrics(observed_log, lower_log, upper_log):
    y, lower_log, upper_log = (np.asarray(values, dtype=float) for values in (observed_log, lower_log, upper_log))
    if (y.ndim != 1 or lower_log.shape != y.shape or upper_log.shape != y.shape or not len(y)
            or not all(np.isfinite(values).all() for values in (y, lower_log, upper_log))
            or np.any(lower_log > upper_log)):
        raise ValueError("Observações e limites ordenados precisam ser finitos e ter o mesmo tamanho positivo.")
    below, above = y < lower_log, y > upper_log
    widths = np.exp(upper_log) - np.exp(lower_log)
    if not np.isfinite(widths).all():
        raise ValueError("Limites de previsão não finitos.")
    hits = int((~(below | above)).sum())
    return {"n": len(y), "cobertos": hits, "abaixo": int(below.sum()), "acima": int(above.sum()),
            "cobertura": hits / len(y), "largura_media": float(widths.mean()),
            "largura_mediana": float(np.median(widths)), "largura_p90": float(np.quantile(widths, .9))}


def evaluate_intervals(frame, center, radius, period):
    if period not in {"2024", "2025"} or not frame.Ano.astype(str).eq(period).all():
        raise ValueError("Ano da amostra incompatível com a etapa de avaliação.")
    center = np.asarray(center, dtype=float)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Raio de avaliação inválido.")
    return evaluate_bounds(frame, center - radius, center + radius, period)


def evaluate_bounds(frame, lower, upper, period):
    y = frame.log_salario_hora.to_numpy(dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    summary = bounds_metrics(y, lower, upper)
    education = pd.cut(frame.Anos_de_Estudo, [-.001, 4, 8, 11, 16],
                       labels=["0–4 anos", "5–8 anos", "9–11 anos", "12 anos ou mais"]).astype("string").fillna("Sem informação")
    rows = []
    bins, bands = observed_wage_bands(np.exp(y))
    groups = {"regiao": frame.Regiao, "sexo": frame.Sexo_Desc, "escolaridade": education}
    labels_by_dimension = {name: values.astype("string").fillna("Sem informação").to_numpy()
                           for name, values in groups.items()}
    labels_by_dimension["faixa_salarial"] = bins
    for dimension, labels in labels_by_dimension.items():
        for group in sorted(pd.unique(labels)):
            mask = labels == group
            result = bounds_metrics(y[mask], lower[mask], upper[mask])
            enough = result["n"] >= MIN_GROUP
            row = {"dimensao": dimension, "grupo": str(group), "status": "ok" if enough else "insufficient",
                   **{k: v if enough or k == "n" else None for k, v in result.items()}}
            if dimension == "faixa_salarial":
                row.update(bands[group])
                if not enough:
                    row.update(limite_inferior=None, limite_superior=None)
            rows.append(row)
    return {"periodo": period, "summary": summary, "rows": rows,
            "amostra_sha256": sample_fingerprint(frame)}


def candidates(legacy, seed):
    categories = legacy.BASE_CATEGORICAL + legacy.FIXED_EFFECTS
    return {
        **{f"GradientBoosting_folhas{leaves}": legacy.build_pipeline(HistGradientBoostingRegressor(
            max_iter=250, learning_rate=.08, max_leaf_nodes=leaves, l2_regularization=1., random_state=seed), categories, dense=True) for leaves in [31, 15]},
        "RandomForest": legacy.build_pipeline(RandomForestRegressor(n_estimators=80, min_samples_leaf=20,
            max_features=.7, n_jobs=4, random_state=seed), categories, dense=True),
        "OLS_efeitos_fixos": legacy.build_pipeline(LinearRegression(), categories),
    }


def train_intervals(path, output=ARTIFACT_DIR, max_train=150_000, max_eval=50_000, seed=42):
    path, output = Path(path), Path(output)
    if min(max_train, max_eval) < 100:
        raise ValueError("Use ao menos 100 registros por etapa.")
    output.mkdir(parents=True, exist_ok=True)
    legacy = prediction.legacy_module()
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    protocol = {"version": VERSION, "run_id": run, "meta": TARGET, "periodos": PERIODS,
                "score": "abs(log observado − log da estimativa escolhida)",
                "quantil": "k = ceil((n + 1) × 0,90); raio = k-ésimo escore ordenado, sem interpolação",
                "criterio_modelo": "Menor RMSE log em 2022, sem reestimar depois da seleção",
                "criterio_conversao": "Duan se RMSE em reais cair e viés absoluto não aumentar em 2023T3–T4",
                "seed": seed, "limites_amostrais": {"treino": max_train, "demais_etapas": max_eval},
                "entrada": file_info(path), "fixado_em": now(), "teste_usado_no_ajuste": False}
    protocol_ref = save(protocol, output, "protocolo")
    def sample(expr, limit, offset):
        return prediction.sample_frame(path, expr, limit, seed + offset)
    print("Etapas 1–2/6: ajuste 2013–2021 e seleção exclusivamente em 2022", flush=True)
    train = sample((ds.field("Ano") >= 2013) & (ds.field("Ano") <= 2021), max_train, 0)
    validation = sample(ds.field("Ano") == 2022, max_eval, 1)
    models = candidates(legacy, seed)
    validation_results = []
    with threadpool_limits(limits=4):
        for name, model in models.items():
            print(f"Treinando {name}", flush=True)
            validation_results.extend(legacy.avaliar_sklearn(name, model, train, {"validacao": validation}))
        leader = min(validation_results, key=lambda item: item["rmse_log"])
        model = models[leader["modelo"]]
    artifact = output / f"pipeline_{run}.joblib"
    joblib.dump(model, artifact)
    manifest = {"version": VERSION, "run_id": run, "modelo": leader["modelo"],
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "arquivo": artifact.name,
                "schema": prediction.input_schema(train), "sklearn": sklearn.__version__, "python": platform.python_version(),
                "entrada": file_info(path), "seed": seed, "data_base": "Janeiro/2026", "ponderacao": "Não ponderado",
                "divisao_temporal": PERIODS, "validacao": validation_results, "criterio": protocol["criterio_modelo"],
                "metricas_validacao": leader, "treinado_em": now(), "limitacoes": LIMITS,
                "amostras": {"treino": len(train), "validacao": len(validation)},
                "amostras_sha256": {"treino": sample_fingerprint(train), "validacao": sample_fingerprint(validation)}}
    del train, validation, models
    print("Etapas 3–4/6: fator em 2023T1–T2 e escolha da conversão em 2023T3–T4", flush=True)
    half1 = sample((ds.field("Ano") == 2023) & (ds.field("Trimestre") <= 2), max_eval, 3)
    half2 = sample((ds.field("Ano") == 2023) & (ds.field("Trimestre") >= 3), max_eval, 4)
    with threadpool_limits(limits=4):
        factor = fit_factor(half1.log_salario_hora, model.predict(half1))
        p2 = model.predict(half2)
    original = metrics(half2.log_salario_hora, p2)
    corrected = metrics(half2.log_salario_hora, p2, smearing_factor=factor)
    accepted = recommend(original, corrected)
    selected_factor = factor if accepted else 1.
    decision = {"version": VERSION, "run_id": run, "modelo_sha256": manifest["sha256"], "protocolo": protocol_ref,
                "fator_duan": factor, "fator_aplicado": selected_factor, "conversao": "duan" if accepted else "original",
                "original": original, "duan": corrected, "periodo_decisao": PERIODS["conversao"],
                "amostras": {"fator_duan": len(half1), "conversao": len(half2)},
                "amostras_sha256": {"fator_duan": sample_fingerprint(half1), "conversao": sample_fingerprint(half2)},
                "congelado_em": now(), "calibracao_2024_usada_na_decisao": False, "teste_usado_na_decisao": False}
    decision_ref = save(decision, output, "decisao")
    manifest["amostras"].update(decision["amostras"])
    manifest["amostras_sha256"].update(decision["amostras_sha256"])
    del half1, half2
    print(f"Decisão gravada antes da calibração: {decision['conversao']} · {decision_ref['sha256']}", flush=True)
    print("Etapa 5/6: calibração do raio exclusivamente em 2024", flush=True)
    calibration = sample(ds.field("Ano") == 2024, max_eval, 5)
    with threadpool_limits(limits=4):
        calibration_center = np.clip(model.predict(calibration), -20, 20) + np.log(selected_factor)
    radius, rank = conformal_radius(calibration.log_salario_hora, calibration_center)
    calibration_result = evaluate_intervals(calibration, calibration_center, radius, "2024")
    frozen = {"version": VERSION, "run_id": run, "modelo_sha256": manifest["sha256"], "decisao": decision_ref,
              "protocolo": protocol_ref, "meta": TARGET, "raio_log": radius, "ordem": rank,
              "n": len(calibration), "amostra_sha256": calibration_result["amostra_sha256"],
              "congelado_em": now(), "teste_usado_na_calibracao": False}
    calibration_ref = save(frozen, output, "calibracao")
    del calibration
    print(f"Raio {radius:.8f} gravado ANTES do teste · {calibration_ref['sha256']}", flush=True)
    print("Etapa 6/6: avaliação retrospectiva em 2025; sem ajustar raio ou conversão", flush=True)
    # Offset 2 preserva exatamente a amostra reservada da execução anterior.
    test = sample(ds.field("Ano") == 2025, max_eval, 2)
    with threadpool_limits(limits=4):
        predicted = model.predict(test)
    center = np.clip(predicted, -20, 20) + np.log(selected_factor)
    test_result = evaluate_intervals(test, center, radius, "2025")
    diagnostic = build_diagnostics(test, predicted, manifest, smearing_factor=selected_factor, verify_original=False)
    diagnostic["limitacoes"] = LIMITS
    diagnostic["conversao"] = decision["conversao"]
    manifest["amostras"].update(calibracao=frozen["n"], teste=len(test))
    manifest["amostras_sha256"].update(calibracao=frozen["amostra_sha256"], teste=test_result["amostra_sha256"])
    manifest["metricas_teste"] = diagnostic["summary"]
    manifest["interpretacao"] = ("Conversão com fator global de Duan, escolhido em 2023T3–T4, para aproximar a média em reais."
                                 if accepted else "exp(log previsto): aproximação da média geométrica condicional.")
    manifest["intervalo"] = "Faixa com meta de 90%, calibrada em 2024; cobertura avaliada retrospectivamente em 2025."
    payload = {"status": "ok", "version": VERSION, "run_id": run, "modelo": manifest["modelo"],
               "modelo_sha256": manifest["sha256"], "meta": TARGET, "raio_log": radius,
               "protocolo": protocol, "protocolo_arquivo": protocol_ref, "decisao": decision, "decisao_arquivo": decision_ref,
               "calibracao": frozen, "calibracao_arquivo": calibration_ref, "calibration": calibration_result, "test": test_result,
               "diagnostico": diagnostic, "manifesto": manifest, "dimensoes": DIMENSIONS,
               "minimo_grupo": MIN_GROUP, "limites": LIMITS, "referencia_metodologica": REFERENCE, "processado_em": now()}
    report_ref = save(payload, output, "relatorio")
    pointer = {"version": VERSION, "run_id": run, "sha256": manifest["sha256"], "relatorio": report_ref}
    temp = output / f"manifesto_{run}.tmp"
    temp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, output / "manifesto.json")
    print(f"Concluído: cobertura {test_result['summary']['cobertura']:.2%}; largura média R$ {test_result['summary']['largura_media']:.2f}/h", flush=True)
    return payload


@lru_cache(maxsize=12)
def read_artifact(path, stamp, expected):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise Unavailable("Artefato de intervalos difere do hash registrado. Execute novamente o script 24.")
    return json.loads(content)


def checked_read(folder, reference):
    folder = Path(folder)
    path = folder / reference["arquivo"]
    if path.parent.resolve() != folder.resolve() or path.suffix != ".json" or not path.is_file():
        raise Unavailable("Artefato de intervalos ausente ou caminho inválido.")
    return read_artifact(path, path.stat().st_mtime_ns, reference["sha256"])


def get_intervals(folder=None):
    folder = ARTIFACT_DIR if folder is None else Path(folder)
    pointer = read_json(folder / "manifesto.json")
    if not pointer:
        return {"status": "unavailable", "message": "Intervalos ainda não gerados. Execute python pnad_pipeline/scripts/24_intervalos_previsao.py."}
    payload = checked_read(folder, pointer["relatorio"])
    if payload["run_id"] != pointer["run_id"] or payload["modelo_sha256"] != pointer["sha256"] or payload["manifesto"]["sha256"] != pointer["sha256"] or payload["manifesto"]["run_id"] != pointer["run_id"]:
        raise Unavailable("Relatório de intervalos incompatível com o modelo atual.")
    for key in ["protocolo", "decisao", "calibracao"]:
        frozen = checked_read(folder, payload[key + "_arquivo"])
        if frozen != payload[key] or frozen["run_id"] != pointer["run_id"] or (key != "protocolo" and frozen.get("modelo_sha256") != pointer["sha256"]):
            raise Unavailable("Etapa congelada de intervalos incompatível com o relatório.")
    if (payload["calibracao"]["decisao"] != payload["decisao_arquivo"]
            or payload["calibracao"]["protocolo"] != payload["protocolo_arquivo"]
            or payload["decisao"]["protocolo"] != payload["protocolo_arquivo"]
            or payload["calibracao"]["teste_usado_na_calibracao"]
            or payload["decisao"]["teste_usado_na_decisao"]
            or payload["decisao"]["calibracao_2024_usada_na_decisao"]
            or payload["meta"] != TARGET or payload["calibracao"]["meta"] != TARGET
            or payload["protocolo"]["meta"] != TARGET
            or payload["raio_log"] != payload["calibracao"]["raio_log"]
            or payload["calibracao"]["amostra_sha256"] != payload["calibration"]["amostra_sha256"]):
        raise Unavailable("Protocolo de calibração incompatível com o relatório.")
    provenance = {"conjunto": "Intervalos da execução " + pointer["run_id"], "periodo": "2025", "ponderacao": "Não ponderado",
                  "versao": VERSION, "run_id": pointer["run_id"], "modelo_sha256": pointer["sha256"],
                  "amostra_sha256": payload["test"]["amostra_sha256"], "filtros": {},
                  "data_base": payload["manifesto"]["data_base"], "processado_em": payload["processado_em"],
                  "conversao": payload["decisao"]["conversao"]}
    return clean({**payload, "provenance": provenance})


@lru_cache(maxsize=2)
def load_model(folder, model_json, stamp):
    return prediction.validated_pipeline(json.loads(model_json), folder)


def refresh_evaluation(folder=None):
    """Acrescenta métricas às amostras registradas usando o pipeline congelado."""
    folder = ARTIFACT_DIR if folder is None else Path(folder)
    pointer_path = folder / "manifesto.json"
    if not pointer_path.is_file():
        raise Unavailable("Execute o script 24 sem --reavaliar para gerar a primeira execução.")
    pointer_bytes = pointer_path.read_bytes()
    report = get_intervals(folder)
    report.pop("provenance")
    manifest = report["manifesto"]
    source = (ROOT / manifest["entrada"]["arquivo"]).resolve()
    if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
        raise ValueError("Restaure a base da execução registrada no manifesto.")
    model = prediction.validated_pipeline(manifest, folder)
    factor = report["decisao"]["fator_aplicado"]
    for key, year, offset in [("calibration", 2024, 5), ("test", 2025, 2)]:
        previous = report[key]
        frame = prediction.sample_frame(source, ds.field("Ano") == year, previous["summary"]["n"], manifest["seed"] + offset)
        with threadpool_limits(limits=4):
            raw = model.predict(frame)
        center = np.clip(raw, -20, 20) + np.log(factor)
        current = evaluate_intervals(frame, center, report["raio_log"], str(year))
        if current["amostra_sha256"] != previous["amostra_sha256"] or any(
            not np.isclose(current["summary"][name], value, rtol=1e-9, atol=1e-10)
            for name, value in previous["summary"].items()
        ):
            raise ValueError(f"A reavaliação não reproduz a amostra ou os resultados registrados de {year}.")
        report[key] = current
        if key == "test":
            diagnostic = build_diagnostics(frame, raw, manifest, smearing_factor=factor, verify_original=False)
            check_test_metrics(diagnostic["summary"], manifest)
            diagnostic.update(limitacoes=report["diagnostico"]["limitacoes"], conversao=report["decisao"]["conversao"])
            report["diagnostico"] = diagnostic
            manifest["metricas_teste"] = diagnostic["summary"]
    report.update(dimensoes=DIMENSIONS, processado_em=now())
    pointer = json.loads(pointer_bytes)
    pointer["relatorio"] = save(report, folder, "relatorio")
    temp = folder / f"manifesto_{pointer['relatorio']['sha256']}.tmp"
    temp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if pointer_path.read_bytes() != pointer_bytes:
            raise ValueError("A execução mudou durante a reavaliação. Execute novamente.")
        os.replace(temp, pointer_path)
    finally:
        temp.unlink(missing_ok=True)
    return report


def predict_interval(profile):
    report = get_intervals()
    if report["status"] != "ok":
        raise Unavailable(report["message"])
    manifest = report["manifesto"]
    values = prediction.prepare_profile(profile, manifest)
    path = ARTIFACT_DIR / manifest["arquivo"]
    if not path.is_file():
        raise Unavailable("Pipeline de intervalos ausente. Execute novamente o script 24.")
    model = load_model(ARTIFACT_DIR, json.dumps(manifest, sort_keys=True), path.stat().st_mtime_ns)
    with threadpool_limits(limits=4):
        raw = float(model.predict(pd.DataFrame([values]))[0])
    log_center = float(np.clip(raw, -20, 20) + np.log(report["decisao"]["fator_aplicado"]))
    radius = report["raio_log"]
    lower, point, upper = np.exp([log_center-radius, log_center, log_center+radius])
    if not np.isfinite([lower, point, upper]).all():
        raise ValueError("Previsão ou limites não finitos para este perfil.")
    return clean({"salario_hora": point, "previsao_log": raw, "centro_log_intervalo": log_center,
                  "intervalo_previsao": {"meta": TARGET, "inferior": lower, "superior": upper, "largura": upper-lower,
                      "raio_log": radius, "calibrado_em": "2024", "avaliado_em": "2025", "teste": report["test"]["summary"],
                      "calibracao_sha256": report["calibracao_arquivo"]["sha256"], "protocolo_sha256": report["protocolo_arquivo"]["sha256"]},
                  "run_id": manifest["run_id"], "modelo": manifest["modelo"], "modelo_sha256": manifest["sha256"],
                  "perfil": values, "retransformacao": report["decisao"]["conversao"], "fator_duan": report["decisao"]["fator_duan"],
                  "decisao_sha256": report["decisao_arquivo"]["sha256"], "interpretacao": manifest["interpretacao"],
                  "ponderacao": manifest["ponderacao"], "data_base": manifest["data_base"], "limitacoes": LIMITS,
                  "aviso_ano": LIMITS[4] if int(values["Ano"]) > 2021 else None})
