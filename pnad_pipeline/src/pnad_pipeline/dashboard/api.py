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


@app.get("/api/modelos")
def models():
    return data.models()


class Profile(BaseModel):
    perfil: dict


@app.post("/api/previsao")
def prediction(body: Profile):
    return predict(body.perfil)


@app.get("/api/saude")
def health():
    return {"status":"ok","version":data.VERSION,"workspace":WORKSPACE_ID,"pid":os.getpid()}


app.mount("/", StaticFiles(directory=data.ROOT / "frontend",html=True),name="frontend")
