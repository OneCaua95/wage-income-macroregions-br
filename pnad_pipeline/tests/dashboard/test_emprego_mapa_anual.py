"""Filtros pareados do mapa, domínios independentes e contratos dos agregados."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pnad_pipeline.dashboard import data
from pnad_pipeline.dashboard.api import app


@pytest.fixture
def mapa(tmp_path, monkeypatch):
    partes = []
    for ano, trimestre in [(2013, 1), (2025, 4)]:
        for uf in [33, 35]:
            for sexo in ["Homem", "Mulher"]:
                salario = (10. if ano == 2013 else 15.) if sexo == "Mulher" else 20.
                partes.append(pd.DataFrame({"Ano": ano, "Trimestre": trimestre, "Idade": 30, "UF": uf,
                    "V1028": np.tile([1., 3.], 20), "Estrato": 1, "UPA": np.arange(1, 41),
                    "Condicao_Ocupacao": 1, "Categoria_emprego": np.tile([1, 2], 20),
                    "Contribuicao_previdencia": 1, "Rendimento_hab_Trab_princ": salario * 40 * 4.33,
                    "Horas_hab_trabalhadas": 40., "Salario_Hora": salario,
                    "Sexo_Desc": sexo, "Cor_Desc": "Branca", "Escolaridade_Nivel": "Médio Completo",
                    "Faixa_Etaria": "25–34", "Formalidade": np.tile(["Formal", "Informal"], 20),
                    "Setor": "Privado", "Regiao": "Sudeste"}))
    frame = pd.concat(partes, ignore_index=True)
    path = tmp_path / "mapa.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(data, "dataset_path", lambda source: path)
    monkeypatch.setattr(data, "provenance", lambda source, period, filters: {"periodo": period, "filtros": filters, "ponderacao": "V1028"})
    data.load_period.cache_clear()
    return frame, path


def test_mapa_crescimento_filtra_ambos_periodos_e_preserva_ufs(mapa):
    r = data.uf_map("historico", "2025T4", {"sexo": "Mulher", "uf": "35"})
    assert r["status"] == "ok" and len(r["rows"]) == 27
    sp, rj = [next(row for row in r["rows"] if row["sigla"] == sigla) for sigla in ["SP", "RJ"]]
    assert sp["crescimento_media_pct"] == pytest.approx(50.)
    assert sp["media_base"] == 10. and sp["media"] == 15.
    assert sp["n_salarios_base"] == rj["n_salarios_base"] == 40
    assert sp["informalidade_pct"] == 75.
    assert r["provenance"]["filtros"] == {"sexo": "Mulher"}
    assert data.uf_map("historico", "2013T1", {})["brazil"]["crescimento_media_pct"] == 0.
    real = data.uf_map("real", "2025T4", {})
    assert "crescimento_media_pct" not in real["metrics"]
    assert real["brazil"]["crescimento_media_pct"] is None


def test_mapa_informalidade_nao_exige_renda_ou_horas_e_suprime_salario(mapa):
    frame, path = mapa
    frame.Horas_hab_trabalhadas = 0.
    frame.to_parquet(path, index=False)
    data.load_period.cache_clear()
    r = data.uf_map("historico", "2025T4", {})
    assert r["status"] == "ok"
    assert r["brazil"]["n_salarios"] == 0 and r["brazil"]["mediana"] is None
    assert r["brazil"]["informalidade_pct"] == 75.
    assert r["brazil"]["n_formalidade"] == 160
    # Recorte sem observações não converte indisponibilidade em zero.
    vazio = data.uf_map("historico", "2025T4", {"cor": "Indígena"})
    assert vazio["status"] == "empty" and vazio["brazil"]["informalidade_pct"] is None


def test_api_agregados_anuais_e_setoriais_preservam_escopo(tmp_path, monkeypatch):
    report = {"status": "ok", "versao": "teste", "periodo": "2013 a 2025", "ponderacao": "V1028",
              "processado_em": "2026-11-13T12:00:00Z", "notas": [],
              "criterio_informalidade": "Critério previdenciário do projeto",
              "rows": [{"trimestre": "2013T1", "grupo": "Brasil"}, {"trimestre": "2025T4", "grupo": "Brasil"}],
              "composicao_setorial": [{"trimestre": "2013T1"}, {"trimestre": "2025T4"}]}
    for name in ["mercado_trabalho", "evolucao_regional_anual"]:
        (tmp_path / ("resumo_" + name + ".json")).write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(data, "HISTORICO_MERCADO", tmp_path)
    monkeypatch.setattr(data, "HISTORICO_EVOLUCAO_ANUAL", tmp_path)
    monkeypatch.setattr(data, "file_info", lambda path: {"arquivo": path.name})
    client = TestClient(app)
    mercado = client.get("/api/mercado-trabalho?fonte=historico&periodo=2025T4&sexo=Mulher").json()
    assert len(mercado["sector_rows"]) == 1 and len(mercado["all_sector_rows"]) == 2
    assert mercado["sector_rows"][0]["trimestre"] == "2025T4"
    assert mercado["provenance"]["criterio_informalidade"] == report["criterio_informalidade"]
    anual = client.get("/api/evolucao-regional-anual?fonte=historico&periodo=2025T4&sexo=Mulher").json()
    assert len(anual["rows"]) == 2 and anual["provenance"]["filtros"] == {}
    assert anual["provenance"]["filtros_globais_nao_aplicados"] == {"sexo": "Mulher"}
    assert client.get("/api/evolucao-regional-anual?fonte=real").json()["status"] == "unavailable"
