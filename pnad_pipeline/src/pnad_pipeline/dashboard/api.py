"""API local e arquivos estáticos do dashboard PNAD."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data
from .prediction import predict
from .prediction_cqr import get_comparison
from .prediction_diagnostics import get_diagnostics
from .prediction_smearing import get_smearing
from .prediction_intervals import get_intervals, predict_interval
from .runtime import WORKSPACE_ID

app = FastAPI(title="PNAD · Laboratório de resultados", version=data.VERSION)


@app.exception_handler(data.Unavailable)
def unavailable(request: Request, exc: data.Unavailable):
    return JSONResponse(status_code=409,content={"status":"unavailable","message":str(exc)})


@app.exception_handler(ValueError)
def invalid(request: Request, exc: ValueError):
    return JSONResponse(status_code=422,content={"status":"error","message":str(exc)})


@app.exception_handler(Exception)
def failure(request: Request, exc: Exception):
    logging.exception("Falha ao consultar o dashboard")
    return JSONResponse(status_code=500,content={"status":"error","message":"Não foi possível processar os arquivos locais. Consulte o log do servidor e tente novamente."})


def selection(request):
    query = request.query_params
    filters = {k:query[k] for k in data.FILTERS if query.get(k)}
    return query.get("fonte","real"), query.get("periodo","2025T4"), filters


@lru_cache(maxsize=1)
def get_catalog():
    return data.catalog()


@app.get("/api/catalogo")
def catalog():
    return get_catalog()


@app.get("/api/descritiva")
def describe(request: Request):
    return data.describe(*selection(request),group=request.query_params.get("grupo","regiao"))


@app.get("/api/mapa-uf")
def uf_map(request: Request):
    return data.uf_map(*selection(request))


@app.get("/api/serie")
def series(request: Request):
    source, _, filters = selection(request)
    return data.series(source,filters)


@app.get("/api/comparacao")
def compare(request: Request):
    q = request.query_params
    return data.compare(*selection(request), group=q.get("grupo","sexo"),a=q.get("a"),b=q.get("b"),mode=q.get("modo","grupos"))


@app.get("/api/inferencia")
def inference(request: Request):
    return data.inference(*selection(request))


@app.get("/api/educacao")
def education(request: Request):
    q = request.query_params
    return data.education(*selection(request), specification=q.get("especificacao", "categorias"),
                          dimension=q.get("dimensao", "geral"), contrast=q.get("contraste", "acumulado"))


@app.get("/api/selecao")
def selection_analysis(request: Request):
    return data.selection_analysis(*selection(request), dimension=request.query_params.get("dimensao", "escolaridade"))


@app.get("/api/heterogeneidade-regional")
def regional_heterogeneity(request: Request):
    return data.regional_heterogeneity(*selection(request))


@app.get("/api/pseudopainel")
def pseudo_panel(request: Request):
    return data.pseudo_panel(*selection(request))


@app.get("/api/desigualdade")
def inequality(request: Request):
    return data.inequality(*selection(request))


@app.get("/api/diferencas-grupos")
def group_gaps(request: Request):
    return data.group_gaps(*selection(request))


@app.get("/api/modelos")
def models():
    return data.models()


@app.get("/api/oaxaca")
def oaxaca(request: Request):
    return data.oaxaca(*selection(request))


@app.get("/api/robustez-m5")
def robustez_m5(request: Request):
    return data.robustez_m5(*selection(request))


@app.get("/api/estabilidade-m5")
def estabilidade_m5(request: Request):
    return data.estabilidade_m5(*selection(request))


@app.get("/api/sensibilidade-uf-m5")
def sensibilidade_uf_m5(request: Request):
    return data.sensibilidade_uf_m5(*selection(request))


@app.get("/api/sensibilidade-emprego-m5")
def sensibilidade_emprego_m5(request: Request):
    return data.sensibilidade_emprego_m5(*selection(request))


@app.get("/api/nominal-real-m5")
def nominal_real_m5(request: Request):
    return data.nominal_real_m5(*selection(request))


@app.get("/api/multiplicidade-m5")
def multiplicidade_m5(request: Request):
    return data.multiplicidade_m5(*selection(request))


@app.get("/api/mercado-trabalho")
def mercado_trabalho(request: Request):
    return data.mercado_trabalho(*selection(request))


@app.get("/api/evolucao-regional-anual")
def evolucao_regional_anual(request: Request):
    return data.evolucao_regional_anual(*selection(request))


@app.get("/api/incerteza-descritiva")
def incerteza_descritiva(request: Request):
    return data.incerteza_descritiva(*selection(request))


@app.get("/api/decomposicao-theil")
def decomposicao_theil(request: Request):
    return data.decomposicao_theil(*selection(request))


@app.get("/api/sensibilidade-coortes")
def sensibilidade_coortes(request: Request):
    return data.sensibilidade_coortes(*selection(request))


@app.get("/api/incerteza-desigualdade")
def incerteza_desigualdade(request: Request):
    return data.incerteza_desigualdade(*selection(request))


@app.get("/api/crescimento-salarial")
def crescimento_salarial(request: Request):
    return data.crescimento_salarial(*selection(request))


@app.get("/api/rif-desigualdade")
def rif_desigualdade(request: Request):
    return data.rif_desigualdade(*selection(request))


@app.get("/api/multicolinearidade-m5")
def multicolinearidade_m5(request: Request):
    return data.multicolinearidade_m5(*selection(request))


@app.get("/api/residuos-m5")
def residuos_m5(request: Request):
    return data.residuos_m5(*selection(request))


@app.get("/api/influencia-m5")
def influencia_m5(request: Request):
    return data.influencia_m5(*selection(request))


@app.get("/api/dispersao-m5")
def dispersao_m5(request: Request):
    return data.dispersao_m5(*selection(request))


class Profile(BaseModel):
    perfil: dict


@app.post("/api/previsao")
def prediction(body: Profile):
    return predict(body.perfil)


@app.get("/api/saude")
def health():
    return {"status":"ok","version":data.VERSION,"workspace":WORKSPACE_ID,"pid":os.getpid()}


@app.get("/api/diagnostico-previsao")
def prediction_diagnostics():
    return get_diagnostics()


@app.get("/api/duan")
def smearing_comparison():
    return get_smearing()


@app.get("/api/intervalos-previsao")
def prediction_interval_report():
    return get_intervals()


@app.post("/api/previsao-intervalos")
def prediction_interval(body: Profile):
    return predict_interval(body.perfil)


@app.get("/api/comparacao-intervalos")
def prediction_interval_comparison():
    return get_comparison()


@app.get("/api/comparacao-escala-m5")
def comparacao_escala_m5():
    return data.comparacao_escala_m5()


app.mount("/", StaticFiles(directory=data.ROOT / "frontend",html=True),name="frontend")
