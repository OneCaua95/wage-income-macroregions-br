"""Contratos estatísticos do dashboard com microdados sintéticos pequenos."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pnad_pipeline.dashboard import data, prediction
from pnad_pipeline.dashboard.api import app, get_catalog


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    # Inclui desocupado, renda zero e classificação de formalidade ausente.
    df = pd.DataFrame({
        "Ano":[2025]*6,"Trimestre":[4]*6,"Idade":[30]*6,"UF":[35]*6,
        "V1028":[1.,3.,2.,4.,5.,6.],"Estrato":[1]*6,"UPA":[1,2,3,4,5,6],
        "Condicao_Ocupacao":[1,1,2,1,1,1],"Categoria_emprego":[1,2,1,9,9,1],
        "Contribuicao_previdencia":[1,2,1,1,np.nan,1],
        "Rendimento_hab_Trab_princ":[1732.,3464.,1000.,0.,5196.,6928.],
        "Horas_hab_trabalhadas":[40.]*6,"Salario_Hora":[10.,20.,5.,np.nan,30.,40.],
        "Sexo_Desc":["Homem","Mulher","Mulher","Homem","Mulher","Homem"],
        "Cor_Desc":["Branca"]*6,"Escolaridade_Nivel":["Médio Completo"]*6,
        "Faixa_Etaria":["25–34"]*6,"Formalidade":["Formal","Informal","Formal","Formal","Informal","Formal"],
        "Setor":["Privado"]*6,"Regiao":["Sudeste"]*6,
    })
    real = tmp_path/"real.parquet"
    historical = tmp_path/"historico.parquet"
    df.to_parquet(real,index=False)
    df.drop(columns=["V1028","Estrato","UPA"]).to_parquet(historical,index=False)
    monkeypatch.setattr(data,"dataset_path",lambda source: real if source=="real" else historical)
    monkeypatch.setattr(data,"provenance",lambda source,period,filters:{"periodo":period,"ponderacao":"V1028" if source=="real" else "Não ponderado","filtros":filters})
    data.load_period.cache_clear(); data.series_cached.cache_clear();get_catalog.cache_clear()
    return df,real


def test_weighted_domains_and_quantiles(local_data):
    r = data.describe("real","2025T4",{})
    s = r["summary"]; wage=s["estatisticas"][0]
    assert s["n"] == 6 and s["populacao"] == 21
    assert wage["n_validos"] == 4 and wage["n_ausentes"] == 2
    assert wage["media"] == pytest.approx((10+60+150+240)/15)
    assert wage["mediana"] == 30  # inversa empírica ponderada, sem interpolação
    assert wage["populacao_valida"] == 15
    assert s["n_formalidade"] == 4
    assert s["taxa_formalidade"] == pytest.approx(11/14)
    assert sum(h["n"] for h in r["histogram"]) == wage["n_validos"]
    assert sum(h["proporcao"] for h in r["histogram"]) == pytest.approx(1)


def test_filters_match_temporal_series_and_frequency_denominator(local_data):
    r = data.describe("real","2025T4",{"sexo":"Mulher"})
    series = data.series("real",{"sexo":"Mulher"})
    assert series["rows"][0]["media"] == pytest.approx(r["summary"]["estatisticas"][0]["media"])
    freq = [f for f in r["frequencies"] if f["variavel"] == "Formalidade"]
    assert sum(f["proporcao"] for f in freq) == pytest.approx(1)
    assert all(f["denominador"] == 10 for f in freq)
    # Uma ocupada com contribuição ausente fica sem classificação, não informal.
    assert data.describe("real","2025T4",{"formalidade":"Informal"})["summary"]["n"] == 1


def test_historical_source_never_invents_population(local_data):
    r = data.describe("historico","2025T4",{})
    assert r["summary"]["populacao"] is None
    assert r["provenance"]["ponderacao"] == "Não ponderado"
    assert r["summary"]["estatisticas"][0]["media"] == 25


def test_empty_and_insufficient_comparison(local_data):
    assert data.describe("real","2025T4",{"uf":"11"})["status"] == "empty"
    r = data.compare("real","2025T4",{},"sexo","Homem","Mulher")
    assert r["status"] == "insufficient" and r["diferenca_absoluta"] is None
    with pytest.raises(ValueError,match="diferentes"):
        data.compare("real","2025T4",{},"sexo","Homem","Homem")


def test_comparison_uses_a_as_denominator(monkeypatch):
    frames = [pd.DataFrame({"x":[10]*30}),pd.DataFrame({"x":[15]*30})]
    monkeypatch.setattr(data,"filtered",lambda source,period,filters:frames[0] if filters.get("sexo")=="Homem" else frames[1])
    monkeypatch.setattr(data,"summary_frame",lambda df:{"n_salario":len(df),"estatisticas":[{"media":df.x.mean()}]})
    monkeypatch.setattr(data,"provenance",lambda *args:{})
    r = data.compare("real","2025T4",{"sexo":"Homem"},"sexo","Homem","Mulher")
    assert r["diferenca_absoluta"] == 5
    assert r["diferenca_percentual"] == 50


def test_inference_rejects_unestimated_domains():
    assert data.inference("real","2025T4",{"sexo":"Mulher"})["status"] == "unavailable"
    assert data.inference("historico","2025T4",{})["status"] == "unavailable"
    assert data.inference("real","2024T1",{})["status"] == "unavailable"
    assert data.coefficient_details("Idade_centrada","geral")[2] == "continua"
    assert data.coefficient_details("Estudo_x_is_formal","geral")[2] == "interacao"
    assert data.coefficient_details("is_formal","geral")[2] == "indicadora_condicional"


def test_age_square_does_not_overflow_in_training(tmp_path):
    columns = prediction.legacy_module().MODEL_COLUMNS
    df = pd.DataFrame({c:["x"]*330 for c in columns if c not in ("Idade_2","Faixa_Etaria")})
    df["Ano"] = [2022]*110+[2023]*110+[2025]*110
    df["Idade"] = np.full(330,40,dtype=np.int8)
    df["Anos_de_Estudo"] = np.full(330,12,dtype=np.int8)
    df["Salario_Hora"] = 20.
    path=tmp_path/"train.parquet";df.to_parquet(path,index=False)
    _,frames=prediction.sample_temporal(path,100,100,42)
    assert all((f.Idade_2 == 1600).all() for f in frames.values())
    _,second=prediction.sample_temporal(path,100,100,42)
    for split in frames:pd.testing.assert_frame_equal(frames[split],second[split])


def test_api_error_and_missing_artifact(local_data,monkeypatch,tmp_path):
    client=TestClient(app)
    assert client.get("/api/saude").status_code == 200
    assert client.get("/api/descritiva?periodo=invalido").status_code == 422
    assert client.get("/api/descritiva?grupo=invalido").status_code == 422
    assert client.get("/api/descritiva?sexo=Mulher").json()["summary"]["n"] == 3
    monkeypatch.setattr(prediction,"ARTIFACT_DIR",tmp_path)
    assert client.post("/api/previsao",json={"perfil":{}}).status_code == 409


def test_launcher_reuses_existing_server(monkeypatch):
    from pnad_pipeline.dashboard import launcher
    status = {"status":"ok", "workspace":launcher.WORKSPACE_ID, "pid":12345}
    monkeypatch.setattr(launcher,"health",lambda port:status)
    def unexpected_start(*args,**kwargs):
        pytest.fail("Uma segunda abertura não deve criar outro servidor.")
    monkeypatch.setattr(launcher.subprocess,"Popen",unexpected_start)
    assert launcher.start() == status


def test_launcher_never_stops_another_workspace(monkeypatch):
    from pnad_pipeline.dashboard import launcher
    monkeypatch.setattr(launcher,"health",lambda port:{"status":"ok","workspace":"outro-projeto","pid":12345})
    calls=[]
    monkeypatch.setattr(launcher.os,"kill",lambda *args:calls.append(args))
    with pytest.raises(RuntimeError,match="Nenhum processo"):
        launcher.stop()
    assert not calls
