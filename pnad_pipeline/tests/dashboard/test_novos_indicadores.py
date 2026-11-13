"""Período e filtros das séries históricas pré-calculadas."""

import json

from fastapi.testclient import TestClient

from pnad_pipeline.dashboard import data
from pnad_pipeline.dashboard.api import app


def test_series_fixas_preservam_grupos_e_filtros_ignorados(tmp_path, monkeypatch):
    relatorio = {"status": "ok", "periodo": "2013T1 a 2025T4", "ponderacao": "V1028",
                 "versao": "teste", "processado_em": "2026-11-12T12:00:00Z", "notas": [],
                 "rows": [{"trimestre": "2013T1", "grupo": "Brasil"}, {"trimestre": "2025T4", "grupo": "Brasil"}]}
    (tmp_path / "resumo_incerteza_descritiva.json").write_text(json.dumps(relatorio), encoding="utf-8")
    monkeypatch.setattr(data, "HISTORICO_INCERTEZA", tmp_path)
    monkeypatch.setattr(data, "file_info", lambda path: {"arquivo": path.name})
    client = TestClient(app)
    result = client.get("/api/incerteza-descritiva?fonte=historico&periodo=2025T4&regiao=Sul").json()
    assert result["status"] == "ok" and len(result["rows"]) == 1 and len(result["all_rows"]) == 2
    assert result["provenance"]["filtros"] == {}
    assert result["provenance"]["filtros_globais_nao_aplicados"] == {"regiao": "Sul"}
    assert result["rows"][0]["trimestre"] == "2025T4"
    assert client.get("/api/incerteza-descritiva?fonte=real").json()["status"] == "unavailable"
    assert client.get("/api/decomposicao-theil?fonte=real").json()["status"] == "unavailable"
    assert client.get("/api/sensibilidade-coortes?fonte=real").json()["status"] == "unavailable"
    assert client.get("/api/incerteza-desigualdade?fonte=real").json()["status"] == "unavailable"
    assert client.get("/api/crescimento-salarial?fonte=real").json()["status"] == "unavailable"
    assert client.get("/api/rif-desigualdade?fonte=real").json()["status"] == "unavailable"
