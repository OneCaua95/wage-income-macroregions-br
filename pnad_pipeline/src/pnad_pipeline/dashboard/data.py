"""Consultas agregadas. Nenhum endpoint expõe registros individuais."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pnad_pipeline.analise.amostragem import media_ponderada, quantis_ponderados, validar_desenho

ROOT = Path(__file__).resolve().parents[4]
PROCESSED = ROOT / "pnad_pipeline" / "data" / "processed"
REAL = PROCESSED / "validacao_real" / "2025T4"
HISTORICO_PONDERADO = PROCESSED / "recoleta_ponderada" / "pnad_analise_25_65.parquet"
HISTORICO_LEGADO = PROCESSED / "pnad_analise_25_65.parquet"
HISTORICO_INFERENCE = PROCESSED / "recoleta_ponderada" / "inferencia_amostral"
HISTORICO_EDUCATION = HISTORICO_INFERENCE / "retorno_educacao"
HISTORICO_SELECTION = HISTORICO_INFERENCE / "selecao_ocupacional"
HISTORICO_REGIONAL = HISTORICO_INFERENCE / "heterogeneidade_regional"
HISTORICO_PSEUDO = HISTORICO_INFERENCE / "pseudopainel"
HISTORICO_INEQUALITY = HISTORICO_INFERENCE / "desigualdade"
HISTORICO_GROUP_GAPS = HISTORICO_INFERENCE / "diferencas_grupos"
VERSION = "dashboard-2.3"
FILTERS = {
    "regiao": "Regiao", "uf": "UF", "sexo": "Sexo_Desc", "cor": "Cor_Desc",
    "escolaridade": "Escolaridade_Nivel", "idade": "Faixa_Etaria",
    "formalidade": "Formalidade", "setor": "Setor",
}
LABELS = {
    "regiao": "Macroregião", "uf": "UF", "sexo": "Sexo", "cor": "Cor/raça",
    "escolaridade": "Anos de estudo", "idade": "Faixa etária",
    "formalidade": "Formalidade", "setor": "Setor",
}
EDUCATION = dict(zip(
    ["Até Fundamental Incompleto", "Fundamental Completo", "Médio Completo", "Superior Completo"],
    ["0–4 anos", "5–8 anos", "9–11 anos", "12 anos ou mais"],
))
UF_NAMES = dict(zip(
    [11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,29,31,32,33,35,41,42,43,50,51,52,53],
    ["RO","AC","AM","RR","PA","AP","TO","MA","PI","CE","RN","PB","PE","AL","SE","BA","MG","ES","RJ","SP","PR","SC","RS","MS","MT","GO","DF"],
))
UF_FULL_NAMES = dict(zip(
    [11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,29,31,32,33,35,41,42,43,50,51,52,53],
    ["Rondônia","Acre","Amazonas","Roraima","Pará","Amapá","Tocantins","Maranhão","Piauí","Ceará","Rio Grande do Norte","Paraíba","Pernambuco","Alagoas","Sergipe","Bahia","Minas Gerais","Espírito Santo","Rio de Janeiro","São Paulo","Paraná","Santa Catarina","Rio Grande do Sul","Mato Grosso do Sul","Mato Grosso","Goiás","Distrito Federal"],
))
REGION_SLUGS = {"geral": "geral", "Norte": "norte", "Nordeste": "nordeste", "Sudeste": "sudeste", "Sul": "sul", "Centro-Oeste": "centro_oeste"}
REGION_BASE_UF = {"geral":11, "Norte":11, "Nordeste":21, "Sudeste":31, "Sul":41, "Centro-Oeste":50}


class Unavailable(ValueError):
    pass


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else (default or {})


def clean(value):
    """JSON estrito: valores ausentes são null, nunca NaN ou Infinity."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def file_info(path: Path):
    return {"arquivo": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "modificado_em": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()} if path.exists() else None


def dataset_path(source: str) -> Path:
    if source == "real":
        return REAL / "pnad_analise_2025T4.parquet"
    if source == "historico":
        return HISTORICO_PONDERADO if HISTORICO_PONDERADO.exists() else HISTORICO_LEGADO
    raise ValueError("Fonte inválida. Use real ou historico.")


def provenance(source: str, period: str | None, filters: dict):
    path = dataset_path(source)
    weighted = "V1028" in pq.ParquetFile(path).schema_arrow.names if path.exists() else False
    historical_weighted = source == "historico" and weighted
    validation = read_json(REAL / "validacao_observada.json") if source == "real" else {}
    return {
        "fonte": "Microdados PNAD Contínua / IBGE",
        "conjunto": "Validação oficial 2025T4" if source == "real" else (
            "Série histórica ponderada 2013–2025" if historical_weighted else "Histórico exploratório local"
        ),
        "periodo": period or "Trimestres disponíveis da fonte",
        "dominio": "Pessoas de 25 a 65 anos; salário-hora entre ocupados com renda e horas positivas" if weighted else "Base histórica previamente restrita: ocupados de 25 a 65 anos com renda e horas positivas",
        "ponderacao": "V1028" if weighted else "Não ponderado",
        "desenho": "Estrato e UPA preservados" if weighted else "V1028 e Estrato ausentes na base histórica local",
        "unidade": "Registros pessoa-trimestre; empilhamento não identifica pessoas únicas",
        "data_base": validation.get("data_base_rendimento", "Janeiro/2026 (documentado no pipeline histórico)"),
        "precos": "Reais, deflacionados pelo IPCA; valores nominais não estão disponíveis nesta base",
        "processado_em": validation.get("executado_em") or (file_info(path) or {}).get("modificado_em"),
        "versao": VERSION, "arquivo": file_info(path), "filtros": filters,
        "url_fonte": validation.get("url_fonte"), "sha256": validation.get("sha256"),
    }


def normalize(df):
    df = df.loc[df.Idade.between(25, 65)].copy()
    occupied = df.Condicao_Ocupacao.eq(1)
    valid_class = df.Categoria_emprego.isin(range(1, 11)) & (
        ~df.Categoria_emprego.isin([8, 9]) | df.Contribuicao_previdencia.isin([1, 2]))
    df["Formalidade"] = df.Formalidade.astype(object).where(occupied & valid_class)
    df["Setor"] = df.Setor.astype(object).where(occupied & df.Categoria_emprego.isin(range(1, 11)))
    df["Salario_Hora"] = df.Salario_Hora.where(occupied & df.Rendimento_hab_Trab_princ.gt(0) & df.Horas_hab_trabalhadas.gt(0))
    return df


def expression(filters, period=None):
    expr = ds.scalar(True)
    if period:
        try:
            year, quarter = map(int, period.split("T"))
        except (ValueError, AttributeError):
            raise ValueError("Período inválido; use AAAAT1 a AAAAT4.")
        if not 2013 <= year <= 2100 or quarter not in (1, 2, 3, 4):
            raise ValueError("Período fora do intervalo permitido.")
        expr = expr & (ds.field("Ano") == year) & (ds.field("Trimestre") == quarter)
    for key, value in filters.items():
        if key not in FILTERS:
            raise ValueError(f"Filtro desconhecido: {key}")
        if value:
            expr = expr & (ds.field(FILTERS[key]) == (int(value) if key == "uf" else value))
    return expr


@lru_cache(maxsize=4)
def load_period(source: str, period: str, stamp: int):
    path = dataset_path(source)
    if not path.exists():
        raise Unavailable("Base analítica ausente. Execute a preparação do pipeline antes de consultar esta fonte.")
    df = ds.dataset(path).to_table(filter=expression({}, period)).to_pandas()
    if "V1028" in df:
        validar_desenho(df)
    return normalize(df)


def filtered(source, period, filters):
    path = dataset_path(source)
    if not path.exists():
        raise Unavailable("Base analítica não disponível nesta instalação.")
    df = load_period(source, period, path.stat().st_mtime_ns)
    mask = np.ones(len(df), dtype=bool)
    for key, value in filters.items():
        if value:
            if key not in FILTERS:
                raise ValueError("Filtro inválido.")
            mask &= df[FILTERS[key]].astype("string").eq(str(value)).fillna(False).to_numpy(dtype=bool)
    return df.loc[mask]


def options_from(df):
    options = {}
    for key, col in FILTERS.items():
        values = sorted(df[col].dropna().unique(), key=str)
        options[key] = [{"value": str(v), "label": UF_NAMES.get(int(v), str(v)) if key == "uf" else EDUCATION.get(str(v), str(v))} for v in values]
    return options


def catalog():
    sources = []
    for source in ("real", "historico"):
        path = dataset_path(source)
        if not path.exists():
            sources.append({"id": source, "available": False})
            continue
        if source == "real":
            periods = ["2025T4"]
        else:
            # Extrai os períodos reais do Parquet, sem depender de uma tabela antiga.
            table = ds.dataset(path).to_table(columns=["Ano", "Trimestre"])
            pairs = table.to_pandas().drop_duplicates().sort_values(["Ano", "Trimestre"])
            periods = [f"{int(y)}T{int(q)}" for y, q in pairs.itertuples(index=False, name=None)]
        df = load_period(source, periods[-1], path.stat().st_mtime_ns)
        sources.append({"id": source, "available": True, "periods": periods,
                        "n_registros": pq.ParquetFile(path).metadata.num_rows,
                        "options": options_from(df), "provenance": provenance(source, periods[-1], {})})
    return {"sources": sources, "labels": LABELS, "version": VERSION}


def numeric_stats(df, col, mask=None):
    values = pd.to_numeric(df[col], errors="coerce").astype("float64")
    valid = np.isfinite(values) & (mask if mask is not None else True)
    sub = df.loc[valid]
    weighted = "V1028" in df
    if not len(sub):
        return {"variavel": col, "n_validos": 0, "n_ausentes": len(df), "media": None,
                "mediana": None, "q1": None, "q3": None, "desvio_padrao": None, "populacao_valida": None}
    if weighted:
        mean = media_ponderada(sub, col)
        q1, median, q3 = quantis_ponderados(sub, col, [.25, .5, .75])
        std = np.sqrt(np.average((values.loc[valid] - mean) ** 2, weights=sub.V1028))
    else:
        mean = values.loc[valid].mean()
        q1, median, q3 = values.loc[valid].quantile([.25, .5, .75])
        std = values.loc[valid].std(ddof=0)
    return clean({"variavel": col, "n_validos": len(sub), "n_ausentes": len(df)-len(sub),
                  "media": mean, "mediana": median, "q1": q1, "q3": q3, "desvio_padrao": std,
                  "populacao_valida": sub.V1028.sum() if weighted else None})


def summary_frame(df):
    occupied = df.Condicao_Ocupacao.eq(1)
    formal = occupied & df.Formalidade.isin(["Formal", "Informal"])
    weights = df.V1028 if "V1028" in df else pd.Series(1., index=df.index)
    denom = float(weights.loc[formal].sum())
    return clean({"n": len(df), "populacao": float(weights.sum()) if "V1028" in df else None,
                  "n_salario": int(df.Salario_Hora.notna().sum()),
                  "taxa_formalidade": float(weights.loc[formal & df.Formalidade.eq("Formal")].sum())/denom if denom else None,
                  "n_formalidade": int(formal.sum()), "denominador_formalidade": denom,
                  "denominador_formalidade_unidade": "População ponderada classificada" if "V1028" in df else "Registros classificados",
                  "estatisticas": [numeric_stats(df, "Salario_Hora"),
                                   numeric_stats(df, "Rendimento_hab_Trab_princ", occupied & df.Rendimento_hab_Trab_princ.gt(0)),
                                   numeric_stats(df, "Horas_hab_trabalhadas", occupied & df.Horas_hab_trabalhadas.gt(0))]})


def describe(source, period, filters, group="regiao"):
    if group not in FILTERS:
        raise ValueError("Agrupamento inválido.")
    df = filtered(source, period, filters)
    summary = summary_frame(df)
    groups, frequencies = [], []
    for value, sub in df.groupby(FILTERS[group], observed=True, dropna=False):
        label = "Sem informação" if pd.isna(value) else EDUCATION.get(str(value), UF_NAMES.get(value, str(value)))
        item = summary_frame(sub)
        groups.append({"grupo": label, "valor": None if pd.isna(value) else str(value), **item})
    total_weight = float(df.V1028.sum()) if "V1028" in df else len(df)
    for key, col in FILTERS.items():
        for val, sub in df.groupby(col, observed=True, dropna=False):
            mass = float(sub.V1028.sum()) if "V1028" in df else len(sub)
            frequencies.append({"variavel": LABELS[key], "categoria": "Sem informação" if pd.isna(val) else EDUCATION.get(str(val), UF_NAMES.get(val, str(val))),
                                "n": len(sub), "total": mass, "proporcao": mass/total_weight if total_weight else None,
                                "denominador": total_weight})
    # Todas as observações positivas entram no histograma; a última classe é aberta.
    valid = df.Salario_Hora.notna()
    edges = [0, 5, 10, 15, 20, 30, 40, 60, 100, np.inf]
    counts, _ = np.histogram(df.loc[valid, "Salario_Hora"], bins=edges)
    mass, _ = np.histogram(df.loc[valid, "Salario_Hora"], bins=edges, weights=df.loc[valid, "V1028"] if "V1028" in df else None)
    histogram = [{"faixa": f"{a}–{b}" if np.isfinite(b) else f"{a}+", "n": int(n), "proporcao": float(m/mass.sum()) if mass.sum() else None} for a,b,n,m in zip(edges[:-1],edges[1:],counts,mass)]
    status = "empty" if not len(df) else "insufficient" if summary["n_salario"] < 30 else "ok"
    return clean({"status": status, "summary": summary, "groups": groups, "frequencies": frequencies,
                  "histogram": histogram, "group": group, "provenance": provenance(source, period, filters),
                  "notes": ["Desvio-padrão descritivo da distribuição, com divisor soma dos pesos; não é erro-padrão pelo desenho.",
                            "Formalidade: ocupados com categoria válida; conta própria/empregador exige contribuição previdenciária informada.",
                            "Faixas de escolaridade representam anos de estudo, sem inferir diplomas."]})


def uf_map(source, period, filters):
    """Distribuição ponderada do salário-hora por UF no período e filtros escolhidos."""
    filters = {key: value for key, value in filters.items() if key != "uf"}
    df = filtered(source, period, filters)
    wage = df.loc[df.Salario_Hora.notna() & df.UF.notna()]
    weighted = "V1028" in wage.columns

    def summarize(sub, code=None):
        n = len(sub)
        row = {
            "uf": None if code is None else str(code),
            "sigla": "BR" if code is None else UF_NAMES.get(int(code), str(code)),
            "nome": "Brasil" if code is None else UF_FULL_NAMES.get(int(code), str(code)),
            "n_salarios": n,
            "populacao_v1028": float(sub.V1028.sum()) if weighted and n else None,
            "mediana": None,
            "p10": None,
            "p90": None,
            "p90_p10": None,
        }
        if n < 30:
            return row
        if weighted:
            p10, median, p90 = quantis_ponderados(sub, "Salario_Hora", [.10, .50, .90])
        else:
            p10, median, p90 = sub.Salario_Hora.quantile([.10, .50, .90]).to_numpy()
        row.update({
            "mediana": float(median),
            "p10": float(p10),
            "p90": float(p90),
            "p90_p10": float(p90 / p10) if p10 > 0 else None,
        })
        return row

    national = summarize(wage)
    rows = []
    groups = {int(code): sub for code, sub in wage.groupby("UF", observed=True)}
    for code, sigla in UF_NAMES.items():
        sub = groups.get(code)
        row = summarize(sub, code) if sub is not None else summarize(wage.iloc[0:0], code)
        row["regiao"] = str(sub.Regiao.iloc[0]) if sub is not None and len(sub) else None
        rows.append(row)
    status = "ok" if national["n_salarios"] >= 30 else "insufficient" if national["n_salarios"] else "empty"
    return clean({
        "status": status,
        "rows": rows,
        "brazil": national,
        "metrics": ["mediana", "p90_p10"],
        "minimum_n": 30,
        "provenance": provenance(source, period, filters),
        "notes": [
            "Valores agrupados por UF no período selecionado; o filtro de UF ativo não remove os demais estados do mapa.",
            "Quantis ponderados pela V1028 quando disponível; P90/P10 é a razão entre os quantis do salário-hora real.",
            "Estados com menos de 30 salários-hora válidos não recebem estimativa. O limite é regra de apresentação, não garantia de precisão.",
            "A malha territorial usa limites estaduais simplificados da API de Malhas Geográficas do IBGE; nenhum microdado individual é enviado ao navegador.",
        ],
    })


@lru_cache(maxsize=12)
def series_cached(source, filter_items, stamp):
    filters = dict(filter_items)
    path = dataset_path(source)
    dataset = ds.dataset(path)
    columns = list(dict.fromkeys(["Ano", "Trimestre", "Idade", "Condicao_Ocupacao", "Salario_Hora",
                                 "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas", "Categoria_emprego",
                                 "Contribuicao_previdencia", "Formalidade", "Setor"] + list(FILTERS.values()) +
                                 (["V1028", "Estrato", "UPA"] if "V1028" in dataset.schema.names else [])))
    totals = {}
    for batch in dataset.scanner(columns=columns, filter=expression(filters), batch_size=65536).to_batches():
        df = normalize(batch.to_pandas())
        # Reaplica filtros depois de validar a classificação de formalidade/setor.
        for key in ("formalidade", "setor"):
            if filters.get(key):
                df = df.loc[df[FILTERS[key]].eq(filters[key])]
        for (year, quarter), sub in df.groupby(["Ano", "Trimestre"], observed=True):
            key = f"{int(year)}T{int(quarter)}"
            v = sub.Salario_Hora.notna()
            weights = sub.V1028 if "V1028" in sub else pd.Series(1., index=sub.index)
            acc = totals.setdefault(key, np.zeros(4))
            acc += [len(sub), int(v.sum()), weights.loc[v].sum(), (sub.loc[v,"Salario_Hora"]*weights.loc[v]).sum()]
    return [{"periodo": k, "n": int(a[0]), "n_validos": int(a[1]), "media": a[3]/a[2] if a[2] else None} for k,a in sorted(totals.items())]


def series(source, filters):
    path = dataset_path(source)
    if not path.exists():
        raise Unavailable("Base da série ausente.")
    return clean({"rows": series_cached(source, tuple(sorted(filters.items())), path.stat().st_mtime_ns),
                  "provenance": provenance(source, None, filters)})


def compare(source, period, filters, group, a, b, mode="grupos"):
    if mode not in ("grupos", "periodos") or group not in FILTERS:
        raise ValueError("Modo de comparação inválido.")
    if not a or not b or a == b:
        raise ValueError("Selecione dois grupos ou períodos diferentes.")
    if mode == "periodos":
        frames = [filtered(source, p, filters) for p in (a, b)]
    else:
        frames = [filtered(source, period, {**filters, group: val}) for val in (a,b)]
    results = [summary_frame(f) for f in frames]
    va, vb = [r["estatisticas"][0]["media"] for r in results]
    enough = all(r["n_salario"] >= 30 for r in results)
    return clean({"status": "ok" if enough else "insufficient", "a": {"selecao":a, **results[0]}, "b": {"selecao":b, **results[1]},
                  "diferenca_absoluta": vb-va if enough and va is not None and vb is not None else None,
                  "diferenca_percentual": (vb-va)/va*100 if enough and va else None,
                  "denominador": "Média do salário-hora de A; diferença percentual = 100 × (B − A) / A",
                  "provenance": provenance(source, f"{a} / {b}" if mode=="periodos" else period, filters)})


def coefficient_details(name, region):
    if name == "Intercept":
        return "Constante", "Perfil de referência; idade 45 anos", "intercepto"
    if name in ("is_formal", "is_public", "Sexo_Dummy"):
        label, ref = {"is_formal":("Trabalho formal", "Informal; associação avaliada em 0 anos de estudo"),
                      "is_public":("Setor público", "Privado; associação avaliada em 0 anos de estudo"),
                      "Sexo_Dummy":("Mulher", "Homem; associação avaliada em 0 anos de estudo")}[name]
        return label, ref, "indicadora_condicional"
    if name.startswith("Estudo_x_"):
        return name.replace("Estudo_x_", "Anos de estudo × ").replace("is_formal", "formal").replace("is_public", "público").replace("Sexo_Dummy", "mulher"), "Interação: combinar com os termos principais", "interacao"
    if name.startswith("Estudo_"):
        code = name.split('_')[-1]
        return f"Escolaridade: {code} anos" + (" ou mais" if code == "16" else ""), "Código 0: sem instrução/menos de 1 ano; homem, informal e setor privado (interações nulas)", "indicadora_condicional"
    if name.startswith("Idade"):
        return ("Idade − 45 anos" if name == "Idade_centrada" else "(Idade − 45 anos)²"), "Idade e idade² devem ser interpretadas conjuntamente", "continua"
    if name.startswith("UF_"):
        return "UF: " + UF_NAMES.get(int(name[3:]), name[3:]), "UF " + UF_NAMES[REGION_BASE_UF[region]], "indicadora"
    if name.startswith("Cor_"):
        return name.replace("Cor_", "Cor/raça: ").replace("Indigena", "Indígena"), "Branca", "indicadora"
    if name.startswith("Setor_"):
        return name.replace("_", " "), "Atividade: códigos fora de 2–10 (inclui 1, 11 e ausentes); especificação do pipeline", "indicadora"
    if name.startswith("Ocup_"):
        label = "Ocupação: elementares (código 9)" if name == "Ocup_Forcas_Armadas_Policiais_Bombeiros" else name.replace("_", " ")
        return label, "Ocupação: códigos fora de 1–9 (inclui 10, 11 e ausentes); o nome legado do termo de código 9 foi corrigido na exibição conforme VD4011", "indicadora"
    if name.startswith("Ano_"):
        return name.replace("_", " "), "2013; termos anuais não identificáveis em um único trimestre", "indicadora"
    if name.startswith("Trimestre_"):
        return name.replace("_", " "), "1º trimestre; termos trimestrais constantes em 2025T4", "indicadora"
    return name.replace("_", " "), "Consultar a matriz do M5", "outro"


def inference(source, period, filters):
    # Recortes demográficos não transformam coeficientes já ajustados.
    if source == "real":
        if period != "2025T4":
            return {"status": "unavailable", "message": "O M5 com desenho amostral está disponível apenas para a execução oficial 2025T4.", "rows": []}
        folder = REAL / "inferencia"
        periodo_resultado = period
    elif source == "historico":
        folder = HISTORICO_INFERENCE
        periodo_resultado = "2013–2025 (empilhado)"
        if not HISTORICO_PONDERADO.exists():
            return {"status": "unavailable", "message": "Reextraia a série histórica com V1028, Estrato e UPA antes de solicitar o M5 ponderado.", "rows": []}
        if not (folder / "resumo_inferencia_m5.json").exists():
            return {"status": "unavailable", "message": "O M5 ponderado da série histórica ainda não foi estimado.", "rows": []}
    else:
        return {"status": "unavailable", "message": "Fonte inferencial indisponível.", "rows": []}
    incompatible = [LABELS[k] for k,v in filters.items() if v and k != "regiao"]
    if incompatible:
        return {"status": "unavailable", "message": "Este recorte exige reestimar o M5: " + ", ".join(incompatible) + ". Os coeficientes publicados só têm recortes por macroregião.", "rows": []}
    region = filters.get("regiao") or "geral"
    if region not in REGION_SLUGS:
        raise ValueError("Macroregião inválida.")
    path = folder / f"inferencial_m5_{REGION_SLUGS[region]}.csv"
    if not path.exists():
        return {"status": "unavailable", "message": "Arquivo de inferência ainda não gerado para esta região.", "rows": []}
    profile_path = folder / f"perfil_idade_m5_{REGION_SLUGS[region]}.csv"
    age_profile = pd.read_csv(profile_path).to_dict("records") if profile_path.exists() else []
    summary = read_json(folder / "resumo_inferencia_m5.json")
    rows, omitted = [], []
    for r in pd.read_csv(path).to_dict("records"):
        if not r.get("identificavel", False):
            omitted.append(r["variavel"])
            continue
        label, reference, kind = coefficient_details(r["variavel"], region)
        beta = r["coeficiente"]
        percent = 100*np.expm1(beta) if kind in ("indicadora", "indicadora_condicional") else None
        rows.append({"variavel":r["variavel"], "rotulo":label, "referencia":reference, "tipo":kind,
                     "coeficiente":beta, "associacao_pct":percent,
                     "erro_padrao":r["erro_padrao_desenho_pnad"], "ic_inferior":r["ic_inferior_desenho_pnad"],
                     "ic_superior":r["ic_superior_desenho_pnad"], "p_valor":r["p_valor_desenho_pnad"],
                     "p_bh":r.get("p_valor_principal_bh")})
    model = next((m for m in summary.get("modelos",[]) if m["modelo"] == region), {})
    metodo = (
        "Linearização por UPA dentro de estrato; IC com distribuição t e gl = UPAs − estratos. "
        f"UPA única: {summary.get('politica_upa_unica', 'não aplicável')}; "
        f"UPA em múltiplos estratos: {summary.get('politica_upa_multiestrato', 'não aplicável')}."
    )
    return clean({"status":"ok", "rows":rows, "omitted":omitted, "model":model, "perfil_idade":age_profile,
                  "selection":"A especificação M5 ponderada está definida como principal; a comparação disponível trata estimadores de variância, não seleção por desempenho.",
                  "variance":summary.get("aproximacao_variancia"), "provenance":{**provenance(source,periodo_resultado,filters), "resultado":file_info(path)},
                  "method":metodo})


def education(source, period, filters, specification="categorias", dimension="geral", contrast="acumulado"):
    """Retornos condicionais de escolaridade pré-estimados na série ponderada."""
    if source != "historico":
        return {"status":"unavailable", "message":"A análise de escolaridade usa a série histórica ponderada de 2013–2025. Selecione essa fonte para consultar os resultados."}
    summary_path = HISTORICO_EDUCATION / "resumo_retorno_educacao.json"
    effects_path = HISTORICO_EDUCATION / "efeitos_retorno_educacao.csv"
    if not summary_path.exists() or not effects_path.exists():
        return {"status":"unavailable", "message":"A estimação do retorno da educação ainda não foi processada."}
    if dimension not in {"geral", "sexo", "cor", "regiao"}:
        raise ValueError("Dimensão inválida para a análise de escolaridade.")
    if contrast not in {"acumulado", "marginal"}:
        raise ValueError("Tipo de contraste inválido para a análise de escolaridade.")
    if dimension == "geral" and specification not in {"categorias", "linear", "spline"}:
        raise ValueError("Especificação inválida para a análise de escolaridade.")
    model_key = specification if dimension == "geral" else f"spline_{dimension}"
    summary = read_json(summary_path)
    rows = pd.read_csv(effects_path)
    rows = rows.loc[rows.especificacao.eq(model_key) & rows.contraste.eq(contrast)].copy()
    if dimension == "geral":
        selected_test_names = {
            "categorias": ["associacao_escolaridade", "desvio_da_forma_linear"],
            "linear": ["associacao_escolaridade"],
            "spline": ["associacao_escolaridade", "componentes_nao_lineares"],
        }[specification]
    else:
        selected_test_names = [f"heterogeneidade_{dimension}"]
    tests = summary.get("testes", {}).get(model_key, {})
    selected_tests = {name:tests[name] for name in selected_test_names if name in tests}
    relevant_filter = {"sexo":"sexo", "cor":"cor", "regiao":"regiao"}.get(dimension)
    filter_value = filters.get(relevant_filter) if relevant_filter else None
    if filter_value:
        rows = rows.loc[rows.grupo.astype(str).eq(str(filter_value))]
    group_filter_used = bool(filter_value)
    model = summary.get("modelos", {}).get(model_key, {})
    notes = [
        "Estimativa ponderada na amostra comum de 2013–2025; filtros descritivos não reestimam este modelo.",
        "Associação condicional entre escolaridade e salário-hora de pessoas ocupadas; não representa efeito causal nem salário potencial de pessoas fora da ocupação.",
        "Os intervalos de 95% são pontuais. O código VD3005 igual a 16 representa 16 anos ou mais e recebe o valor 16 nas tendências linear e spline.",
        "Os testes conjuntos usam aproximação Wald-F com graus de liberdade do desenho.",
    ]
    if filter_value:
        notes.append(f"O filtro global de {LABELS[relevant_filter].lower()} limitou a visualização a {filter_value}; não alterou a estimação.")
    elif filters:
        ignored = ", ".join(LABELS[k].lower() for k,v in filters.items() if v)
        notes.append(f"Filtros ativos ({ignored}) não alteram esta estimação prévia.")
    return clean({
        "status":"ok", "rows":rows.to_dict("records"), "tests":selected_tests, "model":model,
        "specification":model_key, "dimension":dimension, "contrast":contrast, "group_filter_used":group_filter_used,
        "notes":notes, "variance":summary.get("desenho"),
        "provenance":{**provenance(source, "2013–2025 (empilhado)", filters),
                      "resultado":file_info(effects_path), "ponderacao":"V1028",
                      "dominio":summary.get("dominio"), "desenho":"UPA dentro de estrato"},
    })


def selection_analysis(source, period, filters, dimension="escolaridade"):
    """Seleção para salário-hora e sensibilidade por pesos inversos."""
    if source != "historico":
        return {"status":"unavailable", "message":"A análise de seleção usa a série histórica ponderada de 2013–2025. Selecione essa fonte para consultar os resultados."}
    if dimension not in {"escolaridade", "sexo", "cor", "regiao"}:
        raise ValueError("Dimensão inválida para a análise de seleção.")
    summary_path = HISTORICO_SELECTION / "resumo_selecao_ocupacional.json"
    rates_path = HISTORICO_SELECTION / "taxas_selecao.csv"
    effects_path = HISTORICO_SELECTION / "sensibilidade_ipw_escolaridade.csv"
    if not summary_path.exists() or not rates_path.exists() or not effects_path.exists():
        return {"status":"unavailable", "message":"A análise de seleção ocupacional ainda não foi processada."}
    summary = read_json(summary_path)
    rates = pd.read_csv(rates_path)
    effects = pd.read_csv(effects_path)
    notes = [
        "Taxas e propensões são da série histórica empilhada 2013–2025 e usam V1028.",
        "Os pesos inversos reponderam a amostra salarial para o perfil observado da população-alvo; a análise não identifica seleção por fatores não observados nem é causal.",
        "O piso de propensão limita cada fator inverso a 20. Os intervalos do IPW incorporam a estimação da propensão e usam aproximação de variância por UPA/estrato.",
    ]
    if filters:
        active = ", ".join(LABELS[k].lower() for k,v in filters.items() if v)
        if active:
            notes.append(f"Filtros globais ativos ({active}) não reestimam nem recortam esta análise de seleção.")
    return clean({
        "status":"ok", "dimension":dimension, "rates":rates.to_dict("records"),
        "effects":effects.to_dict("records"), "summary":summary, "notes":notes,
        "provenance":{**provenance(source, "2013–2025 (empilhado)", filters),
                      "resultado":file_info(effects_path), "ponderacao":"V1028",
                      "desenho":"UPA dentro de estrato"},
    })


def regional_heterogeneity(source, period, filters):
    """Retornos por macroregião e comparação entre modelos pooled e separados."""
    if source != "historico":
        return {"status":"unavailable", "message":"A inferência regional usa a série histórica ponderada de 2013–2025. Selecione essa fonte para consultar os resultados."}
    summary_path = HISTORICO_REGIONAL / "resumo_heterogeneidade_regional.json"
    education_path = HISTORICO_REGIONAL / "retornos_escolaridade_regiao.csv"
    formality_path = HISTORICO_REGIONAL / "associacao_formalidade_regiao.csv"
    comparison_path = HISTORICO_REGIONAL / "comparacao_pooled_separado.csv"
    paths = (summary_path, education_path, formality_path, comparison_path)
    if not all(path.exists() for path in paths):
        return {"status":"unavailable", "message":"A reestimação da heterogeneidade regional ainda não foi processada."}
    summary = read_json(summary_path)
    education = pd.read_csv(education_path)
    formality = pd.read_csv(formality_path)
    comparison = pd.read_csv(comparison_path)
    notes = [
        "Estimativas da série histórica empilhada de 2013–2025, ponderadas por V1028, com linearização por UPA dentro de estrato.",
        "O teste conjunto compara escolaridade e formalidade entre macroregiões no modelo pooled; efeitos percentuais são associações condicionais, não causais.",
        "A comparação de RMSE e R² é ponderada e feita dentro da amostra; não mede desempenho fora da amostra.",
    ]
    if filters:
        active = ", ".join(LABELS[k].lower() for k,v in filters.items() if v)
        if active:
            notes.append(f"Filtros globais ativos ({active}) não reestimam nem recortam esta análise regional.")
    return clean({
        "status":"ok", "education":education.to_dict("records"),
        "formality":formality.to_dict("records"), "comparison":comparison.to_dict("records"),
        "summary":summary, "notes":notes,
        "provenance":{**provenance(source, "2013–2025 (empilhado)", filters),
                      "resultado":file_info(education_path), "ponderacao":"V1028",
                      "desenho":"UPA dentro de estrato"},
    })


@lru_cache(maxsize=4)
def _pseudo_cached(summary_stamp: int, cells_stamp: int, models_stamp: int, coefficients_stamp: int):
    summary_path = HISTORICO_PSEUDO / "resumo_pseudopainel.json"
    cells_path = HISTORICO_PSEUDO / "medias_coorte_trimestre.csv"
    models_path = HISTORICO_PSEUDO / "comparacao_especificacoes.csv"
    coefficients_path = HISTORICO_PSEUDO / "coeficientes_pseudopainel.csv"
    summary = read_json(summary_path)
    cells = pd.read_csv(cells_path)
    cells = cells.loc[cells.celula_qualificada_principal.astype(bool)].copy()
    # Combina pares de faixas de cinco anos para manter o gráfico legível.
    cells["grupo_nascimento_plot"] = cells["faixa_nascimento_codigo"].astype(int).floordiv(2)
    bands = cells.groupby("faixa_nascimento_codigo", sort=True).faixa_nascimento.first().to_dict()
    plot_labels = {}
    for group in cells.grupo_nascimento_plot.unique():
        indexes = [i for i in bands if i // 2 == group]
        plot_labels[group] = f"{bands[indexes[0]].split('–')[0]}–{bands[indexes[-1]].split('–')[-1]}"
    cells["grupo_nascimento_plot"] = cells.grupo_nascimento_plot.map(plot_labels)
    cells["wy_plot"] = cells.media_log_salario_hora * cells.w
    cells["wsalario_plot"] = cells.media_salario_hora * cells.w
    trajectories = cells.groupby(["Ano", "sexo", "grupo_nascimento_plot"], observed=True).agg(
        n=("n", "sum"), peso_v1028=("w", "sum"), wy=("wy_plot", "sum"),
        wsalario=("wsalario_plot", "sum"), n_coortes_5_anos=("coorte_id", "nunique"),
    ).reset_index()
    trajectories["media_log_salario_hora"] = trajectories.wy / trajectories.peso_v1028
    trajectories["media_salario_hora"] = trajectories.wsalario / trajectories.peso_v1028
    trajectories = trajectories.rename(columns={"Ano":"ano", "grupo_nascimento_plot":"faixa_nascimento"})
    comparison = pd.read_csv(models_path)
    coefficients = pd.read_csv(coefficients_path)
    primary_n = int(summary["limiar_principal"]["n_amostral_minimo"])
    primary_ess = int(summary["limiar_principal"]["kish_minimo"])
    coefficients = coefficients.loc[
        coefficients.limiar_n.eq(primary_n) & coefficients.limiar_kish.eq(primary_ess)
    ].copy()
    return clean({"summary":summary, "trajectories":trajectories.to_dict("records"),
                  "comparison":comparison.to_dict("records"),
                  "coefficients":coefficients.to_dict("records"),
                  "provenance":{"resultado":file_info(cells_path), "ponderacao":"V1028 nas células; n efetivo de Kish como peso WLS",
                                "versao":"Pseudo-painel exploratório pré-estimado · 2013–2025",
                                "dominio":"Pessoas ocupadas de 25 a 65 anos com salário-hora observável"}})


def pseudo_panel(source, period, filters):
    """Pseudo-painel histórico; filtros globais não reestimam os modelos."""
    if source != "historico":
        return {"status":"unavailable", "message":"O pseudo-painel usa a série histórica ponderada de 2013–2025. Selecione essa fonte para consultar os resultados."}
    paths = [HISTORICO_PSEUDO / name for name in (
        "resumo_pseudopainel.json", "medias_coorte_trimestre.csv",
        "comparacao_especificacoes.csv", "coeficientes_pseudopainel.csv",
    )]
    if not all(path.exists() for path in paths):
        return {"status":"unavailable", "message":"A construção do pseudo-painel de coortes ainda não foi executada."}
    result = _pseudo_cached(*(path.stat().st_mtime_ns for path in paths))
    notes = ["Os filtros demográficos globais não reestimam esta análise histórica."] if filters else []
    return {"status":"ok", **result, "notes":notes}


@lru_cache(maxsize=4)
def _inequality_cached(summary_stamp: int, distribution_stamp: int, lorenz_stamp: int, quantiles_stamp: int):
    summary_path = HISTORICO_INEQUALITY / "resumo_desigualdade.json"
    distribution_path = HISTORICO_INEQUALITY / "distribuicao_desigualdade.csv"
    lorenz_path = HISTORICO_INEQUALITY / "curva_lorenz.csv"
    quantiles_path = HISTORICO_INEQUALITY / "coeficientes_quantis.csv"
    summary = read_json(summary_path)
    distribution = pd.read_csv(distribution_path)
    lorenz = pd.read_csv(lorenz_path)
    quantiles = pd.read_csv(quantiles_path)
    series = distribution.loc[
        distribution.dimensao.eq("geral") & distribution.grupo.eq("Brasil")
    ].sort_values("periodo")
    return clean({
        "summary": summary,
        "series": series.to_dict("records"),
        "distribution": distribution.to_dict("records"),
        "lorenz": lorenz.to_dict("records"),
        "quantiles": quantiles.to_dict("records"),
        "provenance": {
            "resultado": file_info(distribution_path),
            "ponderacao": "V1028",
            "desenho": "UPA dentro de estrato; aproximação com reposição",
            "versao": "Desigualdade salarial e regressões quantílicas · série histórica 2013–2025",
        },
    })


def inequality(source, period, filters):
    """Distribuição ponderada do salário-hora e regressões quantílicas pré-estimadas."""
    if source != "historico":
        return {"status": "unavailable", "message": "A análise de desigualdade usa a série histórica ponderada de 2013–2025. Selecione essa fonte para consultar os resultados."}
    paths = [HISTORICO_INEQUALITY / name for name in (
        "resumo_desigualdade.json", "distribuicao_desigualdade.csv",
        "curva_lorenz.csv", "coeficientes_quantis.csv",
    )]
    if not all(path.exists() for path in paths):
        return {"status": "unavailable", "message": "A análise de desigualdade ainda não foi processada. Execute scripts/19_desigualdade_quantis.py na pasta pnad_pipeline."}
    result = _inequality_cached(*(path.stat().st_mtime_ns for path in paths))
    selected = [row for row in result["distribution"] if row["periodo"] == period]
    curves = [row for row in result["lorenz"] if row["periodo"] == period]
    notes = list(result["summary"].get("notas", []))
    if filters:
        active = ", ".join(LABELS[key].lower() for key, value in filters.items() if value)
        if active:
            notes.append(f"Filtros globais ativos ({active}) não reestimam o modelo quantílico nem recortam os grupos desta análise.")
    return clean({
        "status": "ok", "periodo_selecionado": period,
        "summary": result["summary"], "series": result["series"],
        "distribution": selected, "all_distribution": result["distribution"], "lorenz": curves,
        "quantiles": result["quantiles"], "notes": notes,
        "provenance": result["provenance"],
    })


@lru_cache(maxsize=4)
def _group_gaps_cached(summary_stamp: int, results_stamp: int):
    summary_path = HISTORICO_GROUP_GAPS / "resumo_diferencas_grupos.json"
    results_path = HISTORICO_GROUP_GAPS / "diferencas_brutas_ajustadas.csv"
    return read_json(summary_path), pd.read_csv(results_path)


def group_gaps(source, period, filters):
    """Diferenças brutas e ajustadas por sexo e cor/raça na série M5 comum."""
    if source != "historico":
        return {"status": "unavailable", "message": "A comparação bruta e ajustada usa a amostra M5 histórica ponderada de 2013–2025. Selecione a série histórica para consultar os resultados."}
    summary_path = HISTORICO_GROUP_GAPS / "resumo_diferencas_grupos.json"
    results_path = HISTORICO_GROUP_GAPS / "diferencas_brutas_ajustadas.csv"
    if not summary_path.exists() or not results_path.exists():
        return {"status": "unavailable", "message": "A comparação bruta e ajustada ainda não foi processada. Execute scripts/20_diferencas_grupos_salariais.py na pasta pnad_pipeline."}
    summary, rows = _group_gaps_cached(summary_path.stat().st_mtime_ns, results_path.stat().st_mtime_ns)
    notes = [summary.get("contraste_bruto", ""), summary.get("contraste_ajustado", ""), summary.get("variancia", ""), summary.get("interpretacao", "")]
    if filters:
        active = ", ".join(LABELS[key].lower() for key, value in filters.items() if value)
        if active:
            notes.append(f"Filtros globais ativos ({active}) não reestimam esta comparação histórica.")
    return clean({
        "status": "ok", "periodo": summary.get("periodo", "2013–2025 (empilhado)"),
        "summary": {key: value for key, value in summary.items() if key != "linhas"},
        "rows": rows.to_dict("records"), "notes": notes,
        "provenance": {
            "resultado": file_info(results_path), "ponderacao": "V1028",
            "desenho": "UPA dentro de estrato; aproximação com reposição",
            "versao": "Diferenças salariais brutas e ajustadas · M5 ponderado 2013–2025",
        },
    })


def models():
    folder = PROCESSED / "modelos_salario_hora"
    historical = read_json(folder / "resumo_modelos.json")
    path = folder / "metricas_modelos.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    for row in rows:
        # M5 usa filtros de elegibilidade/coortes distintos antes da amostragem.
        # Sem IDs das linhas de avaliação não se afirma equivalência de amostras.
        row["protocolo"] = ("m5_04" if row["modelo"] == "M5_ajustado" else
                            "m5_06" if row["modelo"] == "M5_hierarquico_Ridge" else "benchmark_03")
    artifact = read_json(PROCESSED / "previsao_perfis" / "manifesto.json")
    return clean({"historical":historical, "rows":rows, "prediction":artifact,
                  "provenance":{"resultado":file_info(path), "ponderacao":"Não ponderado", "versao":"Benchmark histórico exploratório; seleção original pelo teste de 2025", "alvo":"log(Salario_Hora)", "data_base":"Janeiro/2026", "dominio":"Ocupados de 25–65 anos com renda e horas positivas", "amostra":historical.get("amostras"), "divisao":historical.get("divisao_temporal")}})
