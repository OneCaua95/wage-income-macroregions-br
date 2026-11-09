"""Duan: fórmula independente, decisão anterior ao teste e serviço coerente."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pnad_pipeline.dashboard import prediction
from pnad_pipeline.dashboard import prediction_smearing as smearing
from pnad_pipeline.dashboard.prediction_diagnostics import metrics
from pnad_pipeline.dashboard.api import app


def frame(year, observed=20., n=120):
    return pd.DataFrame({"Ano": [str(year)]*n, "log_salario_hora": np.full(n, np.log(observed)),
                         "Regiao": ["Sudeste"]*n, "Sexo_Desc": ["Homem"]*n, "Anos_de_Estudo": [12.]*n})


def base_manifest():
    return {"run_id": "frozen-model", "modelo": "fixed", "sha256": "model-sha", "data_base": "Janeiro/2026",
            "entrada": {"arquivo": "base.parquet", "bytes": 1}, "seed": 42}


def write_comparison(folder, manifest, accepted=True):
    frozen = {"run_id": manifest["run_id"], "adotar_duan": accepted, "recomendada": "duan" if accepted else "original",
              "calibracao": {"fator": 1.5}, "teste_usado_na_decisao": False}
    frozen_reference = smearing.save_artifact(frozen, folder, "decisao_duan")
    payload = {"status": "ok", "version": "duan-1.0", "run_id": manifest["run_id"], "modelo_sha256": manifest["sha256"],
               "decisao": frozen, "decisao_arquivo": frozen_reference, "test": {"duan": {"summary": {"rmse_salario_hora": 2.}}}}
    manifest["duan"] = smearing.save_artifact(payload, folder, "comparacao_duan")
    (folder / "manifesto.json").write_text(json.dumps(manifest), encoding="utf-8")
    return payload


def test_factor_is_mean_exp_residual_and_has_no_test_input():
    ratios = np.tile([1., 2., 3., 4.], 30)
    assert smearing.fit_factor(np.log(ratios), np.zeros(120)) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="100"):
        smearing.fit_factor(np.zeros(99), np.zeros(99))
    with pytest.raises(ValueError, match="finitos"):
        smearing.fit_factor(np.r_[np.zeros(119), np.nan], np.zeros(120))


def test_policy_prioritizes_rmse_and_absolute_bias_while_reporting_mae():
    original = {"rmse_salario_hora": 10., "vies_salario_hora": -3., "mae_salario_hora": 5.}
    corrected = {"rmse_salario_hora": 9., "vies_salario_hora": 1., "mae_salario_hora": 6.}
    assert smearing.recommend(original, corrected)
    assert not smearing.recommend(original, {**corrected, "vies_salario_hora": 4.})
    assert not smearing.recommend(original, {**corrected, "rmse_salario_hora": 10.})


def test_corrected_metrics_use_factor_after_base_clipping():
    result = metrics(np.full(120, 20.), np.full(120, 30.), smearing_factor=2.)
    assert result["media_prevista"] == pytest.approx(2*np.exp(20))
    assert result["vies_salario_hora"] == pytest.approx(np.exp(20))
    assert result["vies_log"] == pytest.approx(np.log(2))
    with pytest.raises(ValueError, match="positivo"):
        metrics([1], [1], smearing_factor=0)


def test_freezing_rejects_future_years():
    calibration, validation = frame(2023), frame(2024)
    predictions = np.full(120, np.log(10.))
    decision = smearing.freeze_validation(calibration, predictions, validation, predictions, base_manifest())
    assert decision["calibracao"]["fator"] == pytest.approx(2.)
    assert decision["recomendada"] == "duan" and decision["teste_usado_na_decisao"] is False
    with pytest.raises(ValueError, match="2023"):
        smearing.freeze_validation(frame(2025), predictions, validation, predictions, base_manifest())
    with pytest.raises(ValueError, match="2024"):
        smearing.freeze_validation(calibration, predictions, frame(2025), predictions, base_manifest())


def test_evaluation_persists_decision_before_reading_test_even_when_test_worsens(tmp_path, monkeypatch):
    calibration, validation = frame(2023), frame(2024)
    past = pd.concat([calibration, validation], ignore_index=True)
    # A correção vai piorar propositalmente no teste: isso não pode mudar a decisão.
    future = frame(2025, observed=5.)
    manifest = base_manifest()
    manifest.update({"amostras": {"validacao": len(past), "teste": len(future)},
                     "metricas_validacao": metrics(past.log_salario_hora, np.full(len(past), np.log(10.))),
                     "metricas_teste": metrics(future.log_salario_hora, np.full(len(future), np.log(10.)))})
    (tmp_path / "base.parquet").write_bytes(b"x")
    (tmp_path / "manifesto.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(smearing, "ROOT", tmp_path)
    calls = []
    def temporal(source, **kwargs):
        calls.append(kwargs["splits"])
        if kwargs["splits"] == {"validacao"}:
            assert not list(tmp_path.glob("decisao_duan_*.json"))
            return None, {"validacao": past}
        frozen_path, = tmp_path.glob("decisao_duan_*.json")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        assert frozen["adotar_duan"] and frozen["teste_usado_na_decisao"] is False
        return None, {"teste": future}
    class FixedModel:
        def predict(self, values):
            return np.full(len(values), np.log(10.))
    monkeypatch.setattr(prediction, "sample_temporal", temporal)
    monkeypatch.setattr(prediction, "validated_pipeline", lambda *args: FixedModel())
    result = smearing.evaluate_existing(tmp_path)
    assert calls == [{"validacao"}, {"teste"}]
    assert result["decisao"]["adotar_duan"]
    assert result["test"]["duan"]["summary"]["rmse_salario_hora"] > result["test"]["original"]["summary"]["rmse_salario_hora"]
    updated = json.loads((tmp_path / "manifesto.json").read_text(encoding="utf-8"))
    assert updated.pop("duan") and updated == manifest


def test_api_verifies_model_and_frozen_decision_integrity(tmp_path, monkeypatch):
    manifest = base_manifest()
    monkeypatch.setattr(prediction, "ARTIFACT_DIR", tmp_path)
    client = TestClient(app)
    assert client.get("/api/duan").json()["status"] == "unavailable"
    payload = write_comparison(tmp_path, manifest)
    result = client.get("/api/duan").json()
    assert result["status"] == "ok" and result["decisao"]["adotar_duan"]
    assert result["provenance"]["filtros"] == {}
    current = {**manifest, "sha256": "another-model"}
    (tmp_path / "manifesto.json").write_text(json.dumps(current), encoding="utf-8")
    assert client.get("/api/duan").json()["status"] == "unavailable"
    (tmp_path / "manifesto.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / payload["decisao_arquivo"]["arquivo"]).write_text("{}", encoding="utf-8")
    smearing.read_artifact.cache_clear()
    assert client.get("/api/duan").status_code == 409


@pytest.mark.parametrize("accepted", [True, False])
def test_profile_returns_both_values_and_uses_only_frozen_recommendation(tmp_path, monkeypatch, accepted):
    profile = {"Idade": 40, "Categoria_emprego": "1", "Contribuicao_previdencia": "1", "UF": "35", "Ano": "2025"}
    manifest = base_manifest()
    manifest.update({"arquivo": "pipeline.joblib", "interpretacao": "original", "intervalo": "indisponível", "limitacoes": ["ano ausente"],
                     "metricas_teste": {"rmse_salario_hora": 3.}, "divisao_temporal": {"teste": "2025"}, "ponderacao": "Não ponderado",
                     "schema": [{"name": "Idade", "label": "Idade", "type": "number", "min": 25, "max": 65},
                                *[{"name": key, "label": key, "type": "select", "options": [{"value": value}]} for key, value in profile.items() if key != "Idade"]]})
    write_comparison(tmp_path, manifest, accepted)
    (tmp_path / "pipeline.joblib").write_bytes(b"local")
    monkeypatch.setattr(prediction, "ARTIFACT_DIR", tmp_path)
    class FixedModel:
        def predict(self, values):
            return np.array([np.log(10.)])
    monkeypatch.setattr(prediction, "load_pipeline", lambda *args: FixedModel())
    result = prediction.predict(profile)
    assert result["salario_hora_original"] == pytest.approx(10.)
    assert result["salario_hora_duan"] == pytest.approx(15.)
    assert result["salario_hora"] == pytest.approx(15. if accepted else 10.)
    assert result["retransformacao"] == ("duan" if accepted else "original")
    assert result["metricas_teste"]["rmse_salario_hora"] == (2. if accepted else 3.)
