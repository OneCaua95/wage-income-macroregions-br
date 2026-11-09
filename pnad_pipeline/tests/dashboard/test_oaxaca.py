"""Contrato da decomposição fixa, incluindo filtros e proveniência."""
import json

import pandas as pd
from fastapi.testclient import TestClient

from pnad_pipeline.dashboard import data
from pnad_pipeline.dashboard.api import app


def test_oaxaca_informa_fonte_ou_processamento_ausente(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "HISTORICO_OAXACA", tmp_path)
    client = TestClient(app)
    real = client.get("/api/oaxaca?fonte=real").json()
    assert real["status"] == "unavailable" and "histórica" in real["message"]
    missing = client.get("/api/oaxaca?fonte=historico").json()
    assert missing["status"] == "unavailable" and "21_decomposicao_oaxaca.py" in missing["message"]


def test_oaxaca_preserva_serie_completa_e_nao_atribui_filtros_ao_resultado(tmp_path, monkeypatch):
    summary = {"status": "ok", "periodo": "2013–2025 (empilhado)", "versao": "oaxaca-1.0", "comparacoes": [{"id": "sexo_2_1"}]}
    (tmp_path / "resumo_oaxaca.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [{"comparacao_id": "sexo_2_1", "componente": code, "estimativa_log": value}
            for code, value in [("total", -.1), ("explicada", .05), ("nao_explicada", -.15)]]
    pd.DataFrame(rows).to_csv(tmp_path / "decomposicao_oaxaca.csv", index=False)
    monkeypatch.setattr(data, "HISTORICO_OAXACA", tmp_path)
    monkeypatch.setattr(data, "file_info", lambda path: {"arquivo": path.name})
    data._oaxaca_cached.cache_clear()
    client = TestClient(app)
    response = client.get("/api/oaxaca?fonte=historico&periodo=2025T4&sexo=Mulher&regiao=Sul")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok" and result["periodo"] == summary["periodo"]
    assert len(result["rows"]) == 3
    assert result["provenance"]["filtros"] == {}
    assert result["provenance"]["filtros_globais_nao_aplicados"] == {"sexo": "Mulher", "regiao": "Sul"}
    assert result["provenance"]["periodo"] == summary["periodo"]
    unfiltered = client.get("/api/oaxaca?fonte=historico&periodo=2013T1").json()
    assert result["rows"] == unfiltered["rows"]
    data._oaxaca_cached.cache_clear()
