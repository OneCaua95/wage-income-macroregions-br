"""Avaliação agregada do modelo operacional na mesma amostra reservada de 2025."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import ROOT, Unavailable, clean, read_json

MIN_GROUP = 30
DIMENSIONS = {"regiao": "Macroregião", "sexo": "Sexo", "escolaridade": "Anos de estudo", "faixa_salarial": "Faixa salarial observada"}


def metrics(y_log, pred_log, *, smearing_factor=1.0):
    y_log, pred_log = np.asarray(y_log, dtype=float), np.asarray(pred_log, dtype=float)
    if y_log.ndim != 1 or pred_log.shape != y_log.shape or not len(y_log):
        raise ValueError("Observado e previsto precisam ter o mesmo número positivo de valores.")
    if not np.isfinite(y_log).all() or not np.isfinite(pred_log).all():
        raise ValueError("Observado e previsto precisam conter valores finitos.")
    if not np.isfinite(smearing_factor) or smearing_factor <= 0:
        raise ValueError("O fator de retransformação precisa ser positivo e finito.")
    observed, predicted = np.exp(y_log), np.exp(np.clip(pred_log, -20, 20)) * smearing_factor
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("A retransformação produziu valores não finitos.")
    if smearing_factor != 1.0:
        pred_log = np.clip(pred_log, -20, 20) + np.log(smearing_factor)
    result = {"n": len(y_log), "media_observada": float(observed.mean()), "media_prevista": float(predicted.mean())}
    for suffix, actual, estimate in [("log", y_log, pred_log), ("salario_hora", observed, predicted)]:
        error = estimate - actual
        sse, sst = float(error @ error), float(np.sum((actual - actual.mean())**2))
        result.update({f"mae_{suffix}": float(np.abs(error).mean()), f"rmse_{suffix}": float(np.sqrt(sse / len(error))),
                       f"vies_{suffix}": float(error.mean()), f"r2_{suffix}": float(1 - sse / sst) if sst > 0 else float(sse == 0)})
    return result


def quantile_bins(values, q):
    """Faixas fechadas à direita: salários empatados nunca são separados."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, min(q, len(values)) + 1)))
    if len(edges) == 1:
        return np.zeros(len(values), dtype=int), np.repeat(edges, 2)
    return np.searchsorted(edges[1:-1], values, side="left"), edges


def sample_fingerprint(frame):
    # Ordem e valores, sem exportar registros individuais na API.
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def check_test_metrics(actual, manifest):
    expected = manifest.get("metricas_teste", {})
    keys = [f"{metric}_{scale}" for scale in ("log", "salario_hora") for metric in ("mae", "rmse", "r2")]
    if actual["n"] != manifest.get("amostras", {}).get("teste") or any(
        key not in expected or not np.isclose(actual[key], expected[key], rtol=1e-9, atol=1e-10) for key in keys
    ):
        raise ValueError("A avaliação não reproduz as métricas do manifesto. Confira a base, a semente e o ambiente originais antes de gerar o diagnóstico.")


def build_diagnostics(frame, predictions, manifest, *, smearing_factor=1.0, period="2025", verify_original=True):
    if period not in {"2024", "2025"} or not frame.Ano.astype(str).eq(period).all():
        raise ValueError(f"O diagnóstico aceita somente a amostra do período {period}.")
    y_log = frame.log_salario_hora.to_numpy(dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    original = metrics(y_log, predictions)
    if verify_original:
        if period != "2025":
            raise ValueError("A conferência do manifesto original requer o teste de 2025.")
        check_test_metrics(original, manifest)
    overall = metrics(y_log, predictions, smearing_factor=smearing_factor)
    observed = np.exp(y_log)
    education = pd.cut(pd.to_numeric(frame.Anos_de_Estudo, errors="coerce"), [-.001, 4, 8, 11, 16],
                       labels=["0–4 anos", "5–8 anos", "9–11 anos", "12 anos ou mais"]).astype("string").fillna("Sem informação").to_numpy()
    deciles, edges = quantile_bins(observed, 10)
    labels = {"regiao": frame.Regiao.astype("string").fillna("Sem informação").to_numpy(),
              "sexo": frame.Sexo_Desc.astype("string").fillna("Sem informação").to_numpy(),
              "escolaridade": education, "faixa_salarial": deciles}
    rows = []
    for dimension, groups in labels.items():
        order = list(pd.unique(groups))
        if dimension == "escolaridade":
            order = [g for g in ["0–4 anos", "5–8 anos", "9–11 anos", "12 anos ou mais", "Sem informação"] if g in order]
        else:
            order = sorted(order)
        for value in order:
            mask = groups == value
            result = metrics(y_log[mask], predictions[mask], smearing_factor=smearing_factor)
            enough = result["n"] >= MIN_GROUP
            row = {"dimensao": dimension, "grupo": str(value), "status": "ok" if enough else "insufficient",
                   **{k: v if enough or k == "n" else None for k, v in result.items()}}
            if dimension == "faixa_salarial":
                label = f"D{value+1}" if len(edges) == 11 else f"Faixa {value+1} de {len(edges)-1}"
                row.update({"grupo": label, "ordem": int(value+1), "limite_inferior": float(edges[value]), "limite_superior": float(edges[value+1])})
            rows.append(row)
    # Médias de faixas quantis, evitando entregar pontos individuais ao navegador.
    bins, _ = quantile_bins(observed, min(100, max(1, len(frame) // MIN_GROUP)))
    points = []
    suppressed = 0
    predicted = np.exp(np.clip(predictions, -20, 20)) * smearing_factor
    for value in np.unique(bins):
        mask = bins == value
        n = int(mask.sum())
        if n < MIN_GROUP:
            suppressed += n
            continue
        points.append({"faixa": int(value+1), "n": n, "media_observada": float(observed[mask].mean()),
                       "media_prevista": float(predicted[mask].mean()), "observado_min": float(observed[mask].min()),
                       "observado_max": float(observed[mask].max())})
    return clean({
        "status": "ok", "version": "diagnostico-previsao-1.0", "run_id": manifest["run_id"], "modelo": manifest["modelo"],
        "modelo_sha256": manifest["sha256"], "periodo": period, "ponderacao": "Não ponderado", "data_base": manifest["data_base"],
        "fator_retransformacao": smearing_factor,
        "processado_em": datetime.now(timezone.utc).isoformat(), "entrada": manifest["entrada"], "seed": manifest["seed"],
        "amostra_sha256": sample_fingerprint(frame), "summary": overall, "rows": rows, "observado_previsto": points,
        "dimensoes": DIMENSIONS, "minimo_grupo": MIN_GROUP, "n_nao_exibidos_grafico": suppressed,
        "definicao_vies": "Previsto menos observado: negativo indica subestimação e positivo indica superestimação.",
        "faixas": "Decis do salário-hora observado na amostra de teste, fechados à direita. Empates permanecem juntos; limites duplicados são unidos, podendo resultar em menos de dez faixas.",
        "grafico": "Médias aritméticas de observado e previsto em até 100 faixas quantis do salário observado; não são observações individuais. Faixas com menos de 30 registros não são exibidas.",
        "limitacoes": [f"Avaliação descritiva não ponderada da amostra de {period}; não é estimativa populacional ou causal.",
                       "Modelo base selecionado em 2023–2024; estes resultados de 2024 participam da decisão da conversão." if period == "2024" else "Modelo e hiperparâmetros selecionados na validação 2023–2024; estes diagnósticos de teste não participam da seleção da conversão.",
                       "A previsão é exp(log previsto), sem correção de retransformação; o viés em reais pode refletir essa escolha." if smearing_factor == 1.0 else "A conversão usa um fator global de Duan calibrado em 2023; não assegura média correta em todos os perfis.",
                       "Métricas por grupo com menos de 30 registros são suprimidas; contagens são apresentadas.",
                       "Os erros de 2025 não validam projeções futuras nem constituem intervalos de previsão."],
    })


def save_diagnostics(payload, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    name = f"diagnostico_{payload['run_id']}_{digest[:12]}.json"
    path = folder / name
    temp = path.with_suffix(".tmp")
    temp.write_bytes(content)
    os.replace(temp, path)
    return {"arquivo": name, "sha256": digest, "version": payload["version"]}


def generate_existing(folder=None):
    from . import prediction
    folder = Path(folder) if folder is not None else prediction.ARTIFACT_DIR
    manifest_path = folder / "manifesto.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    source = (ROOT / manifest["entrada"]["arquivo"]).resolve()
    if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
        raise ValueError("Base de origem ausente ou fora do projeto. Restaure a base indicada no manifesto.")
    if source.stat().st_size != manifest["entrada"]["bytes"]:
        raise ValueError("O tamanho da base difere do manifesto. Restaure a base da execução original.")
    model = prediction.validated_pipeline(manifest, folder)
    print("Reconstruindo apenas a amostra reservada de 2025, com a semente original", flush=True)
    _, frames = prediction.sample_temporal(source, max_eval=manifest["amostras"]["teste"], seed=manifest["seed"], splits={"teste"})
    with threadpool_limits(limits=4):
        predicted = model.predict(frames["teste"])
    payload = build_diagnostics(frames["teste"], predicted, manifest)
    reference = save_diagnostics(payload, folder)
    if manifest_path.read_bytes() != original:
        raise ValueError("O manifesto mudou durante o processamento. Execute novamente para a nova versão do modelo.")
    manifest["diagnostico"] = reference
    temp = folder / "manifesto_diagnostico.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, manifest_path)
    print(f"Concluído: {payload['summary']['n']:,} registros, {len(payload['rows'])} grupos e {len(payload['observado_previsto'])} pontos agregados", flush=True)
    return payload


@lru_cache(maxsize=4)
def _read_diagnostics(path, stamp, expected_sha):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise Unavailable("O diagnóstico não corresponde ao SHA-256 registrado. Gere novamente o diagnóstico do modelo.")
    return json.loads(content)


def get_diagnostics():
    from . import prediction
    manifest = read_json(prediction.ARTIFACT_DIR / "manifesto.json")
    reference = manifest.get("diagnostico")
    unavailable = {"status": "unavailable", "message": "Diagnóstico ainda não gerado para o modelo atual. Execute python pnad_pipeline/scripts/22_diagnosticos_previsao.py."}
    if not reference:
        return unavailable
    path = prediction.ARTIFACT_DIR / reference["arquivo"]
    if path.parent.resolve() != prediction.ARTIFACT_DIR.resolve() or path.suffix != ".json":
        raise Unavailable("Caminho de diagnóstico inválido no manifesto.")
    if not path.is_file():
        return unavailable
    payload = _read_diagnostics(path, path.stat().st_mtime_ns, reference["sha256"])
    if payload.get("run_id") != manifest.get("run_id") or payload.get("modelo_sha256") != manifest.get("sha256"):
        return unavailable
    return clean({**payload, "provenance": {
        "conjunto": "Diagnóstico da execução preditiva · " + manifest["run_id"], "periodo": "2025",
        "ponderacao": "Não ponderado", "versao": payload["version"], "run_id": manifest["run_id"],
        "modelo_sha256": manifest["sha256"], "amostra_sha256": payload["amostra_sha256"],
        "data_base": payload["data_base"], "processado_em": payload["processado_em"], "filtros": {},
        "definicao_vies": payload["definicao_vies"], "grafico": payload["grafico"],
    }})
