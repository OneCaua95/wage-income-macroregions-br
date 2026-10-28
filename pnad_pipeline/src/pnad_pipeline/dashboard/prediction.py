"""Exportação reproduzível e serviço do pipeline preditivo completo."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from .data import PROCESSED, ROOT, UF_NAMES, clean, file_info, read_json, Unavailable

ARTIFACT_DIR = PROCESSED / "previsao_perfis"
REGION_UF = {**dict.fromkeys([11,12,13,14,15,16,17], "Norte"),
             **dict.fromkeys([21,22,23,24,25,26,27,28,29], "Nordeste"),
             **dict.fromkeys([31,32,33,35], "Sudeste"), **dict.fromkeys([41,42,43], "Sul"),
             **dict.fromkeys([50,51,52,53], "Centro-Oeste")}
INPUT_LABELS = {"Idade":"Idade", "Anos_de_Estudo":"Anos de estudo (16 = 16 ou mais)", "Sexo_Desc":"Sexo",
                "Cor_Desc":"Cor/raça", "UF":"UF", "Categoria_emprego":"Categoria de emprego (VD4009)",
                "Grupo_atv_princ_empreedimento":"Atividade principal (VD4010)",
                "Contribuicao_previdencia":"Contribui à previdência (VD4012)", "Ano":"Ano do perfil", "Trimestre":"Trimestre do perfil"}
CATEGORY_LABELS = {"1":"Empregado privado com carteira", "2":"Empregado privado sem carteira", "3":"Doméstico com carteira",
                   "4":"Doméstico sem carteira", "5":"Empregado público com carteira", "6":"Empregado público sem carteira",
                   "7":"Militar / estatutário", "8":"Empregador", "9":"Conta própria", "10":"Trabalhador familiar auxiliar"}
ACTIVITY_LABELS = {"1":"Agricultura e pecuária", "2":"Indústria geral", "3":"Construção", "4":"Comércio e reparação",
                   "5":"Transporte e armazenagem", "6":"Alojamento e alimentação", "7":"Informação, finanças e atividades profissionais",
                   "8":"Administração pública", "9":"Educação, saúde e serviços sociais", "10":"Outros serviços", "11":"Serviços domésticos", "12":"Atividades maldefinidas"}


def legacy_module():
    # Reaproveita o pré-processamento e as métricas do benchmark existente.
    spec = importlib.util.spec_from_file_location("pnad_benchmark", ROOT / "pnad_pipeline/scripts/03_modelar_salario_hora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_temporal(path: Path, max_train=150_000, max_eval=50_000, seed=42):
    """Amostra uniforme por prioridades aleatórias, em lotes, com semente fixa.

    Não reproduz as linhas do antigo pandas.sample: é uma nova execução,
    explicitamente separada do ranking histórico.
    """
    legacy = legacy_module()
    columns = [c for c in legacy.MODEL_COLUMNS if c not in {"Faixa_Etaria", "Idade_2"}]
    frames = {}
    specs = [("treino", ds.field("Ano") <= 2022, max_train),
             ("validacao", (ds.field("Ano") >= 2023) & (ds.field("Ano") <= 2024), max_eval),
             ("teste", ds.field("Ano") == 2025, max_eval)]
    dataset = ds.dataset(path)
    for i, (split, expr, limit) in enumerate(specs):
        rng = np.random.default_rng(seed+i)
        kept = pd.DataFrame()
        for batch in dataset.scanner(columns=columns, filter=expr, batch_size=65536).to_batches():
            df = batch.to_pandas()
            df = df.loc[np.isfinite(df.Salario_Hora) & df.Salario_Hora.gt(0) & df.Idade.between(25,65)].copy()
            df["_priority"] = rng.random(len(df))
            kept = pd.concat([kept, df], ignore_index=True)
            if len(kept) > limit:
                indices = np.argpartition(kept._priority.to_numpy(), limit-1)[:limit]
                kept = kept.iloc[indices].copy()
        if len(kept) < 100:
            raise ValueError(f"Amostra insuficiente no conjunto {split}: {len(kept)} registros.")
        kept = kept.sort_values("_priority").drop(columns="_priority").reset_index(drop=True)
        # O Parquet compacto armazena idade como int8; elevar antes da conversão
        # causa overflow (40² vira 64). Treino e serviço usam float64.
        kept["Idade"] = kept.Idade.astype("float64")
        kept["Anos_de_Estudo"] = kept.Anos_de_Estudo.astype("float64")
        kept["Idade_2"] = kept.Idade ** 2
        kept[legacy.TARGET] = np.log(kept.Salario_Hora)
        for col in legacy.BASE_CATEGORICAL + legacy.FIXED_EFFECTS:
            kept[col] = kept[col].astype("string").fillna("Sem informação")
        frames[split] = kept
    return legacy, frames


def train(path, output=ARTIFACT_DIR, max_train=150_000, max_eval=50_000, seed=42):
    path, output = Path(path), Path(output)
    if max_train < 100 or max_eval < 100:
        raise ValueError("Use ao menos 100 registros por conjunto.")
    legacy, data = sample_temporal(path, max_train, max_eval, seed)
    categories = legacy.BASE_CATEGORICAL + legacy.FIXED_EFFECTS
    candidates = {
        "GradientBoosting_folhas31": legacy.build_pipeline(HistGradientBoostingRegressor(max_iter=250,learning_rate=.08,max_leaf_nodes=31,l2_regularization=1.,random_state=seed), categories, dense=True),
        "GradientBoosting_folhas15": legacy.build_pipeline(HistGradientBoostingRegressor(max_iter=250,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.,random_state=seed), categories, dense=True),
        "RandomForest": legacy.build_pipeline(RandomForestRegressor(n_estimators=80,min_samples_leaf=20,max_features=.7,n_jobs=4,random_state=seed), categories, dense=True),
        "OLS_efeitos_fixos": legacy.build_pipeline(LinearRegression(), categories),
    }
    validation = []
    # O teste não participa da seleção dos modelos ou hiperparâmetros.
    with threadpool_limits(limits=4):
        for name, model in candidates.items():
            print(f"Treinando {name}; seleção apenas na validação 2023–2024", flush=True)
            validation.extend(legacy.avaliar_sklearn(name, model, data["treino"], {"validacao":data["validacao"]}))
        leader = min(validation, key=lambda r:r["rmse_log"])
        best = candidates[leader["modelo"]]
        test = legacy.metricas(leader["modelo"], "teste", data["teste"][legacy.TARGET], best.predict(data["teste"]))
    output.mkdir(parents=True, exist_ok=True)
    # Arquivos por execução; manifesto trocado por último, de forma atômica.
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact = output / f"pipeline_{run}.joblib"
    joblib.dump(best, artifact)
    schema = []
    for name in INPUT_LABELS:
        if name in ("Idade", "Anos_de_Estudo"):
            schema.append({"name":name,"label":INPUT_LABELS[name],"type":"number","min":25 if name=="Idade" else 0,"max":65 if name=="Idade" else 16})
        else:
            vals = sorted(data["treino"][name].dropna().astype(str).unique(),key=str)
            if name == "Ano":
                vals = [str(y) for y in range(2013,2026)]
            opts = [{"value":v,"label": UF_NAMES.get(int(v),v) if name=="UF" and v.isdigit() else
                     CATEGORY_LABELS.get(v,v)+f" (cód. {v})" if name=="Categoria_emprego" else
                     ACTIVITY_LABELS.get(v,v)+f" (cód. {v})" if name=="Grupo_atv_princ_empreedimento" else
                     {"1":"Sim","2":"Não"}.get(v,v) if name=="Contribuicao_previdencia" else v} for v in vals]
            schema.append({"name":name,"label":INPUT_LABELS[name],"type":"select","options":opts})
    manifest = clean({"version":"previsao-perfis-1.0", "run_id":run, "modelo":leader["modelo"],
                      "criterio":"Menor RMSE log na validação temporal 2023–2024; teste reservado ao vencedor",
                      "divisao_temporal":{"treino":"2013–2022","validacao":"2023–2024","teste":"2025"},
                      "amostras":{k:len(v) for k,v in data.items()}, "ponderacao":"Não ponderado",
                      "alvo":"log(Salario_Hora)", "data_base":"Janeiro/2026",
                      "validacao":validation, "metricas_validacao":leader, "metricas_teste":test,
                      "arquivo":artifact.name, "sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
                      "entrada":file_info(path), "seed":seed, "amostragem":"Prioridades aleatórias uniformes por lote; nova execução, diferente do pandas.sample histórico",
                      "treinado_em":datetime.now(timezone.utc).isoformat(), "sklearn":sklearn.__version__, "python":platform.python_version(),
                      "features":legacy.BASE_NUMERIC+categories, "schema":schema,
                      "interpretacao":"exp da previsão de log(salário-hora): média geométrica condicional aproximada, sem correção de retransformação. Não é a média aritmética esperada nem uma projeção de trimestre futuro.",
                      "intervalo":"Indisponível: nenhum procedimento de intervalo de previsão foi validado.",
                      "limitacoes":["Efeitos fixos de anos 2023–2025 são categorias não vistas no treino; o OneHotEncoder os ignora, como na avaliação temporal.",
                                    "Recortes descritivos não reestimam este modelo global.", "Modelo não ponderado; previsão associativa para perfis do domínio de ocupados de 25 a 65 anos."]})
    temp = output / f"manifesto_{run}.tmp"
    temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(temp, output / "manifesto.json")
    print(f"Modelo exportado: {manifest['modelo']} em {output}", flush=True)
    return manifest


@lru_cache(maxsize=2)
def load_pipeline(manifest_stamp, artifact_stamp):
    manifest = read_json(ARTIFACT_DIR / "manifesto.json")
    path = ARTIFACT_DIR / manifest["arquivo"]
    if path.parent.resolve() != ARTIFACT_DIR.resolve() or path.suffix != ".joblib":
        raise Unavailable("Caminho do artefato inválido no manifesto.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise Unavailable("O artefato não corresponde ao SHA-256 do manifesto. Reexporte o modelo.")
    if manifest.get("sklearn") != sklearn.__version__:
        raise Unavailable("Versão do scikit-learn diferente da exportação. Reexporte o modelo neste ambiente.")
    # Somente artefatos locais produzidos pelo treinamento; nunca uploads do usuário.
    return joblib.load(path)


def predict(profile: dict):
    mpath = ARTIFACT_DIR / "manifesto.json"
    manifest = read_json(mpath)
    if not manifest:
        raise Unavailable("Modelo treinado ausente. Execute python pnad_pipeline/scripts/12_treinar_previsao.py.")
    values = {}
    for field in manifest["schema"]:
        name = field["name"]
        if name not in profile:
            raise ValueError("Campo obrigatório: " + field["label"])
        if field["type"] == "number":
            num = float(profile[name])
            if not np.isfinite(num) or not field["min"] <= num <= field["max"] or num != int(num):
                raise ValueError(field["label"] + " fora do domínio permitido; informe um número inteiro.")
            values[name] = num
        else:
            val = str(profile[name])
            if val not in {o["value"] for o in field["options"]}:
                raise ValueError("Categoria não disponível: " + field["label"])
            values[name] = val
    category = int(values["Categoria_emprego"])
    contrib = values["Contribuicao_previdencia"]
    values["Idade_2"] = values["Idade"]**2
    values["Setor"] = "Público" if category in (5,6,7) else "Privado"
    values["Formalidade"] = "Formal" if category in (1,3,5,7) or category in (8,9) and contrib == "1" else "Informal"
    values["Regiao"] = REGION_UF[int(values["UF"])]
    path = ARTIFACT_DIR / manifest["arquivo"]
    if not path.exists():
        raise Unavailable("Pipeline não encontrado. Reexporte o modelo.")
    pipeline = load_pipeline(mpath.stat().st_mtime_ns,path.stat().st_mtime_ns)
    with threadpool_limits(limits=4):
        log_value = float(pipeline.predict(pd.DataFrame([values]))[0])
    return clean({"salario_hora":np.exp(np.clip(log_value,-20,20)),"previsao_log":log_value,
                  "modelo":manifest["modelo"], "run_id":manifest["run_id"], "perfil":values,
                  "metricas_teste":manifest["metricas_teste"], "interpretacao":manifest["interpretacao"],
                  "intervalo":manifest["intervalo"], "divisao_temporal":manifest["divisao_temporal"],
                  "ponderacao":manifest["ponderacao"], "data_base":manifest["data_base"],
                  "aviso_ano":manifest["limitacoes"][0] if int(values["Ano"]) > 2022 else None})
