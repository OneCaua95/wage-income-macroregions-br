"""Intervalos: ordem finita, cobertura, isolamento temporal e identidade da API."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pnad_pipeline.dashboard import prediction, prediction_diagnostics
from pnad_pipeline.dashboard import prediction_intervals as intervals
from pnad_pipeline.dashboard.api import app


class FixedModel:
    def predict(self, frame):
        return np.full(len(frame), np.log(10.))


def frame(year, value=20., n=120):
    return pd.DataFrame({"Ano": [str(year)] * n, "log_salario_hora": np.full(n, np.log(value)),
                         "Anos_de_Estudo": [12.] * n, "Sexo_Desc": ["Homem"] * n, "Regiao": ["Sudeste"] * n})


PROFILE = {"Idade": 40, "Categoria_emprego": "1", "Contribuicao_previdencia": "1", "UF": "35", "Ano": "2025"}
SCHEMA = [{"name": "Idade", "label": "Idade", "type": "number", "min": 25, "max": 65},
          *[{"name": k, "label": k, "type": "select", "options": [{"value": v}]} for k, v in PROFILE.items() if k != "Idade"]]


@pytest.fixture
def execution(tmp_path, monkeypatch):
    calls = []
    def sample(path, expression, limit, seed):
        offset = seed - 42
        calls.append((offset, str(expression)))
        protocol, = tmp_path.glob("protocolo_*.json")
        assert json.loads(protocol.read_text(encoding="utf-8"))["meta"] == .9
        if offset == 5:
            decision, = tmp_path.glob("decisao_*.json")
            assert json.loads(decision.read_text(encoding="utf-8"))["fator_aplicado"] == pytest.approx(2.)
            assert not list(tmp_path.glob("calibracao_*.json"))
        if offset == 2:
            calibration, = tmp_path.glob("calibracao_*.json")
            frozen = json.loads(calibration.read_text(encoding="utf-8"))
            assert frozen["teste_usado_na_calibracao"] is False
            assert frozen["raio_log"] == pytest.approx(0., abs=1e-12)
        # O teste deliberadamente falha: o raio e a conversão não podem mudar.
        return frame({0: 2021, 1: 2022, 3: 2023, 4: 2023, 5: 2024, 2: 2025}[offset], 5. if offset == 2 else 20.)
    def evaluate(name, model, train, validation):
        assert train.Ano.eq("2021").all() and validation["validacao"].Ano.eq("2022").all()
        return [{"modelo": name, "rmse_log": 1., "mae_log": .5, "r2_log": .1}]
    monkeypatch.setattr(prediction, "sample_frame", sample)
    monkeypatch.setattr(prediction, "legacy_module", lambda: SimpleNamespace(avaliar_sklearn=evaluate))
    monkeypatch.setattr(prediction, "input_schema", lambda train: SCHEMA)
    monkeypatch.setattr(intervals, "candidates", lambda *args: {"fixed": FixedModel()})
    monkeypatch.setattr(intervals, "file_info", lambda path: {"arquivo": "base.parquet", "bytes": 1})
    monkeypatch.setattr(intervals, "ARTIFACT_DIR", tmp_path)
    result = intervals.train_intervals(tmp_path / "base.parquet", tmp_path, 120, 120)
    return tmp_path, result, calls


def test_quantile_uses_exact_finite_sample_rank_and_keeps_ties():
    scores = np.arange(1., 21.)
    radius, rank = intervals.conformal_radius(scores, np.zeros(20))
    assert rank == 19 and radius == 19.  # np.quantile(scores, .9) produziria 18,1.
    radius, rank = intervals.conformal_radius(np.ones(20), np.zeros(20))
    assert radius == 1. and rank == 19
    with pytest.raises(ValueError, match="insuficiente"):
        intervals.conformal_radius([1.], [0.])
    with pytest.raises(ValueError, match="finitos"):
        intervals.conformal_radius([np.nan] * 20, np.zeros(20))


def test_coverage_includes_borders_and_width_is_in_reais():
    result = intervals.interval_metrics([-1., 1., -1.01, 1.01], np.zeros(4), 1.)
    assert result["cobertos"] == 2 and result["abaixo"] == 1 and result["acima"] == 1
    assert result["cobertura"] == .5
    assert result["largura_media"] == pytest.approx(np.e - 1 / np.e)
    assert result["largura_mediana"] == result["largura_p90"] == result["largura_media"]


def test_group_partitions_reconstruct_overall_and_small_groups_are_suppressed():
    data = frame(2025, n=60)
    data.loc[:19, "Regiao"] = "Norte"
    data.loc[:29, "Sexo_Desc"] = "Mulher"
    data.loc[:29, "Anos_de_Estudo"] = 3.
    center = data.log_salario_hora.to_numpy() + np.r_[np.zeros(30), np.ones(30)]
    result = intervals.evaluate_intervals(data, center, .5, "2025")
    for dimension in ["sexo", "escolaridade"]:
        rows = [r for r in result["rows"] if r["dimensao"] == dimension]
        assert sum(r["n"] for r in rows) == 60
        assert sum(r["cobertos"] for r in rows) == result["summary"]["cobertos"]
        assert sum(r["largura_media"] * r["n"] for r in rows) / 60 == pytest.approx(result["summary"]["largura_media"])
    small, = [r for r in result["rows"] if r["grupo"] == "Norte"]
    assert small["n"] == 20 and small["status"] == "insufficient"
    assert all(v is None for k, v in small.items() if k not in {"n", "grupo", "dimensao", "status"})
    with pytest.raises(ValueError, match="Ano"):
        intervals.evaluate_intervals(data, center, .5, "2024")


def test_coverage_uses_the_diagnostic_wage_bands_with_ties():
    data = frame(2025, n=120)
    data["log_salario_hora"] = np.log(np.repeat([1., 2., 20.], [32, 35, 53]))
    center = np.full(len(data), np.log(2.))
    report = intervals.evaluate_intervals(data, center, .5, "2025")
    manifest = {"run_id": "test", "modelo": "fixed", "sha256": "test", "data_base": "Janeiro/2026",
                "entrada": {}, "seed": 42}
    diagnostic = prediction_diagnostics.build_diagnostics(data, center, manifest, verify_original=False)
    coverage_rows = [r for r in report["rows"] if r["dimensao"] == "faixa_salarial"]
    diagnostic_rows = [r for r in diagnostic["rows"] if r["dimensao"] == "faixa_salarial"]
    assert len(coverage_rows) == len(diagnostic_rows) == 2
    keys = ("grupo", "n", "limite_inferior", "limite_superior")
    assert [[r[k] for k in keys] for r in coverage_rows] == [[r[k] for k in keys] for r in diagnostic_rows]
    assert [r["n"] for r in coverage_rows] == [67, 53]
    assert [r["cobertos"] for r in coverage_rows] == [35, 0]
    assert sum(r["n"] for r in coverage_rows) == len(data)


def test_all_stages_disjoint_and_frozen_before_later_samples(execution):
    folder, result, calls = execution
    assert [offset for offset, expression in calls] == [0, 1, 3, 4, 5, 2]
    assert "2021" in calls[0][1] and "2022" in calls[1][1]
    assert "Trimestre <= 2" in calls[2][1] and "Trimestre >= 3" in calls[3][1]
    assert "2024" in calls[4][1] and "2025" in calls[5][1]
    assert result["decisao"]["conversao"] == "duan"
    assert result["test"]["summary"]["cobertura"] == 0.
    assert result["raio_log"] == pytest.approx(0., abs=1e-12)
    assert intervals.get_intervals(folder)["run_id"] == result["run_id"]


def test_profile_and_interval_share_pipeline_conversion_and_provenance(execution):
    _, report, _ = execution
    client = TestClient(app)
    result = client.post("/api/previsao-intervalos", json={"perfil": PROFILE})
    assert result.status_code == 200
    data = result.json()
    assert data["salario_hora"] == pytest.approx(20.)
    assert data["intervalo_previsao"]["inferior"] == pytest.approx(20.)
    assert data["intervalo_previsao"]["superior"] == pytest.approx(20.)
    assert data["perfil"]["Idade_2"] == 1600.
    assert data["modelo_sha256"] == report["modelo_sha256"]
    assert data["intervalo_previsao"]["calibracao_sha256"] == report["calibracao_arquivo"]["sha256"]
    assert data["intervalo_previsao"]["teste"]["cobertura"] == 0.
    assert client.post("/api/previsao-intervalos", json={"perfil": {**PROFILE, "Idade": 40.5}}).status_code == 422


def test_api_reports_unavailable_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(intervals, "ARTIFACT_DIR", tmp_path)
    client = TestClient(app)
    assert client.get("/api/intervalos-previsao").json()["status"] == "unavailable"
    assert client.post("/api/previsao-intervalos", json={"perfil": PROFILE}).status_code == 409


@pytest.mark.parametrize("kind", ["calibracao", "decisao", "protocolo"])
def test_api_rejects_modified_frozen_artifacts(execution, kind):
    folder, report, _ = execution
    path = folder / report[kind + "_arquivo"]["arquivo"]
    path.write_bytes(path.read_bytes() + b" ")
    assert TestClient(app).get("/api/intervalos-previsao").status_code == 409


def test_api_rejects_different_model_in_report(execution):
    folder, report, _ = execution
    pointer = json.loads((folder / "manifesto.json").read_text(encoding="utf-8"))
    report["manifesto"]["sha256"] = "another-model"
    pointer["relatorio"] = intervals.save(report, folder, "relatorio")
    (folder / "manifesto.json").write_text(json.dumps(pointer), encoding="utf-8")
    assert TestClient(app).get("/api/intervalos-previsao").status_code == 409


def test_profile_rejects_corrupt_pipeline(execution):
    folder, report, _ = execution
    path = folder / report["manifesto"]["arquivo"]
    path.write_bytes(b"alterado")
    assert TestClient(app).post("/api/previsao-intervalos", json={"perfil": PROFILE}).status_code == 409


def test_api_rejects_radius_different_from_frozen_calibration(execution):
    folder, report, _ = execution
    pointer = json.loads((folder / "manifesto.json").read_text(encoding="utf-8"))
    report["raio_log"] = 2.
    pointer["relatorio"] = intervals.save(report, folder, "relatorio")
    (folder / "manifesto.json").write_text(json.dumps(pointer), encoding="utf-8")
    assert TestClient(app).get("/api/intervalos-previsao").status_code == 409


def test_refresh_upgrades_previous_report_and_rejects_a_different_sample(execution, monkeypatch):
    folder, old, _ = execution
    (folder / "base.parquet").touch()
    monkeypatch.setattr(intervals, "ROOT", folder)
    pointer_path = folder / "manifesto.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    old["dimensoes"] = {k: v for k, v in old["dimensoes"].items() if k != "faixa_salarial"}
    for section in ("calibration", "test"):
        old[section]["rows"] = [r for r in old[section]["rows"] if r["dimensao"] != "faixa_salarial"]
    old["diagnostico"]["summary"].pop("mdape_salario_hora")
    for row in old["diagnostico"]["rows"]:
        row.pop("mdape_salario_hora")
    pointer["relatorio"] = intervals.save(old, folder, "relatorio")
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    def sample(path, expression, limit, seed):
        return frame(2024 if seed == 47 else 2025, 20. if seed == 47 else 5.)

    monkeypatch.setattr(prediction, "sample_frame", sample)
    updated = intervals.refresh_evaluation(folder)
    assert updated["diagnostico"]["summary"]["mdape_salario_hora"] == pytest.approx(3.)
    assert "faixa_salarial" in updated["dimensoes"]
    for key in ("run_id", "modelo_sha256", "calibracao_arquivo", "decisao_arquivo", "protocolo_arquivo", "raio_log"):
        assert updated[key] == old[key]
    assert updated["test"]["summary"] == old["test"]["summary"]

    saved_pointer = pointer_path.read_bytes()
    monkeypatch.setattr(prediction, "sample_frame", lambda *args: sample(*args).assign(Sexo_Desc="Mulher"))
    with pytest.raises(ValueError, match="não reproduz a amostra"):
        intervals.refresh_evaluation(folder)
    assert pointer_path.read_bytes() == saved_pointer
