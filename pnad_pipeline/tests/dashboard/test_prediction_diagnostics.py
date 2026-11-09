"""Diagnóstico: escala, direção do erro, amostra fixa e integridade do artefato."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pnad_pipeline.dashboard import prediction
from pnad_pipeline.dashboard import prediction_diagnostics as diagnostics
from pnad_pipeline.dashboard.api import app


def sample():
    n = 240
    observed = np.arange(1, n+1, dtype=float) / 3 + 1
    frame = pd.DataFrame({"Ano": ["2025"]*n, "log_salario_hora": np.log(observed),
                          "Regiao": np.tile(["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"], n//5),
                          "Sexo_Desc": np.tile(["Homem", "Mulher"], n//2),
                          "Anos_de_Estudo": np.tile([2, 6, 10, 14], n//4)})
    predicted = np.log(observed) + .1 * np.sin(np.arange(n))
    manifest = {"run_id": "test-run", "modelo": "modelo_teste", "sha256": "model-sha", "data_base": "Janeiro/2026",
                "entrada": {"arquivo": "base.parquet", "bytes": 1}, "seed": 42, "amostras": {"teste": n},
                "metricas_teste": prediction.legacy_module().metricas("modelo_teste", "teste", frame.log_salario_hora, predicted)}
    return frame, predicted, manifest


def test_metrics_in_both_scales_and_bias_direction():
    observed, predicted = np.array([1., 2., 4., 8.]), np.array([2., 2., 2., 4.])
    result = diagnostics.metrics(np.log(observed), np.log(predicted))
    assert result["mae_salario_hora"] == pytest.approx(1.75)
    assert result["rmse_salario_hora"] == pytest.approx(np.sqrt(5.25))
    assert result["vies_salario_hora"] == pytest.approx(-1.25)
    skewed = diagnostics.metrics(np.log([10., 10., 10.]), np.log([11., 12., 80.]), smearing_factor=1.5)
    assert skewed["mdape_salario_hora"] == pytest.approx(.8)
    assert result["mae_log"] == pytest.approx(.75*np.log(2))
    assert result["rmse_log"] == pytest.approx(np.sqrt(.75)*np.log(2))
    assert result["vies_log"] == pytest.approx(-.25*np.log(2))
    assert result["media_prevista"] - result["media_observada"] == pytest.approx(result["vies_salario_hora"])
    with pytest.raises(ValueError, match="finitos"):
        diagnostics.metrics([1, np.nan], [1, 1])


def test_group_partitions_reconstruct_overall_metrics_and_export_only_aggregates():
    frame, predicted, manifest = sample()
    result = diagnostics.build_diagnostics(frame, predicted, manifest)
    for dimension in diagnostics.DIMENSIONS:
        rows = [r for r in result["rows"] if r["dimensao"] == dimension]
        assert sum(r["n"] for r in rows) == len(frame)
        if all(r["status"] == "ok" for r in rows):
            for scale in ("log", "salario_hora"):
                assert sum(r["n"] * r["mae_"+scale] for r in rows) / len(frame) == pytest.approx(result["summary"]["mae_"+scale])
                assert sum(r["n"] * r["vies_"+scale] for r in rows) / len(frame) == pytest.approx(result["summary"]["vies_"+scale])
                assert np.sqrt(sum(r["n"] * r["rmse_"+scale]**2 for r in rows) / len(frame)) == pytest.approx(result["summary"]["rmse_"+scale])
    points = result["observado_previsto"]
    assert len(points) <= 100 and all(p["n"] >= 30 for p in points)
    assert sum(p["n"] for p in points) + result["n_nao_exibidos_grafico"] == len(frame)
    assert all(set(p) == {"faixa", "n", "media_observada", "media_prevista", "observado_min", "observado_max"} for p in points)


def test_ties_stay_together_and_small_group_metrics_are_suppressed():
    values = np.repeat([1., 2., 10.], [100, 50, 50])
    bins, edges = diagnostics.quantile_bins(values, 10)
    assert len(edges) < 11
    assert all(len(np.unique(bins[values == value])) == 1 for value in np.unique(values))
    single, single_edges = diagnostics.quantile_bins(np.full(50, 10.), 10)
    assert np.unique(single).tolist() == [0] and single_edges.tolist() == [10., 10.]
    frame, predicted, manifest = sample()
    frame.loc[:11, "Regiao"] = "Grupo pequeno"
    result = diagnostics.build_diagnostics(frame, predicted, manifest)
    row = next(r for r in result["rows"] if r["grupo"] == "Grupo pequeno")
    assert row["n"] == 12 and row["status"] == "insufficient"
    assert row["mae_salario_hora"] is None and row["vies_log"] is None
    assert row["mdape_salario_hora"] is None


def test_wrong_period_and_different_evaluation_are_rejected():
    frame, predicted, manifest = sample()
    with pytest.raises(ValueError, match="reproduz"):
        diagnostics.build_diagnostics(frame, predicted + .2, manifest)
    frame.loc[0, "Ano"] = "2024"
    with pytest.raises(ValueError, match="2025"):
        diagnostics.build_diagnostics(frame, predicted, manifest)


def test_api_artifact_identity_and_fixed_provenance(tmp_path, monkeypatch):
    frame, predicted, manifest = sample()
    payload = diagnostics.build_diagnostics(frame, predicted, manifest)
    manifest["diagnostico"] = diagnostics.save_diagnostics(payload, tmp_path)
    path = tmp_path / "manifesto.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(prediction, "ARTIFACT_DIR", tmp_path)
    diagnostics._read_diagnostics.cache_clear()
    client = TestClient(app)
    result = client.get("/api/diagnostico-previsao?fonte=real&periodo=2013T1&sexo=Mulher").json()
    assert result["status"] == "ok" and result["summary"]["n"] == len(frame)
    assert result["periodo"] == "2025" and result["ponderacao"] == "Não ponderado"
    assert result["provenance"]["filtros"] == {}
    manifest["run_id"] = "another-run"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert client.get("/api/diagnostico-previsao").json()["status"] == "unavailable"


def test_api_missing_and_corrupt_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "ARTIFACT_DIR", tmp_path)
    client = TestClient(app)
    assert client.get("/api/diagnostico-previsao").json()["status"] == "unavailable"
    frame, predicted, manifest = sample()
    payload = diagnostics.build_diagnostics(frame, predicted, manifest)
    manifest["diagnostico"] = diagnostics.save_diagnostics(payload, tmp_path)
    (tmp_path / "manifesto.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / manifest["diagnostico"]["arquivo"]).write_text("{}", encoding="utf-8")
    diagnostics._read_diagnostics.cache_clear()
    assert client.get("/api/diagnostico-previsao").status_code == 409


def test_existing_export_uses_only_test_split_and_preserves_the_model(tmp_path, monkeypatch):
    frame, predicted, manifest = sample()
    (tmp_path / "base.parquet").write_bytes(b"x")
    path = tmp_path / "manifesto.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    class FixedModel:
        def predict(self, values):
            pd.testing.assert_frame_equal(frame, values)
            return predicted
    monkeypatch.setattr(prediction, "validated_pipeline", lambda *args: FixedModel())
    def reserved_sample(source, **kwargs):
        assert kwargs["splits"] == {"teste"} and kwargs["seed"] == 42
        return None, {"teste": frame}
    monkeypatch.setattr(prediction, "sample_temporal", reserved_sample)
    diagnostics.generate_existing(tmp_path)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated.pop("diagnostico")
    assert updated == manifest
