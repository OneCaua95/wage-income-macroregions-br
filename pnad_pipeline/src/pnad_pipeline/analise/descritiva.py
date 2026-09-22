"""Módulo de estatísticas descritivas e análises agregadas da PNAD Contínua (2013 a 2025).

Recria e estende as análises descritivas de salário-hora, rendimento habitual real,
desigualdade, formalidade e dinâmica do mercado de trabalho a partir do dataset
consolidado em formato Parquet.
"""

from __future__ import annotations

import logging
from typing import Final, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FONTE_DESCRITIVA: Final[str] = (
    "Fonte: Microdados da PNAD Contínua – IBGE (2013–2025). "
    "Valores deflacionados pelo IPCA (Data-base: Jan/2026)."
)

ATIVIDADES_MAP: Final[dict[int, str]] = {
    1: "Agricultura e Pecuária",
    2: "Indústria Geral",
    3: "Construção",
    4: "Comércio e Reparação",
    5: "Transporte e Armazenagem",
    6: "Alojamento e Alimentação",
    7: "Informação, Comunicação e Finanças",
    8: "Adm Pública, Educação e Saúde",
    9: "Outros Serviços",
    10: "Serviços Domésticos",
}


def preparar_variaveis_analise(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas e descritivas padronizadas para análise estatística.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame carregado do Parquet consolidado.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas adicionais para análises econômicas e demográficas.
    """
    df_out = df.copy()

    # Período formatado (ex: '2013.1')
    df_out["Periodo"] = df_out["Ano"].astype(str) + "." + df_out["Trimestre"].astype(str)

    # Salário-Hora (R$/h)
    mask_horas = (df_out["Horas_hab_trabalhadas"] > 0) & (df_out["Horas_hab_trabalhadas"].notna())
    mask_renda = df_out["Rendimento_hab_Trab_princ"].notna()
    df_out["Salario_Hora"] = np.nan
    df_out.loc[mask_horas & mask_renda, "Salario_Hora"] = (
        df_out.loc[mask_horas & mask_renda, "Rendimento_hab_Trab_princ"]
        / (df_out.loc[mask_horas & mask_renda, "Horas_hab_trabalhadas"] * 4.33)
    )

    # Sexo
    df_out["Sexo_Desc"] = df_out["Sexo"].map({1: "Homem", 2: "Mulher"})

    # Raça / Cor
    cor_map = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena"}
    df_out["Cor_Desc"] = df_out["Cor"].map(cor_map).fillna("Outros")

    # Setor: Público vs Privado
    if "Categoria_emprego" in df_out.columns:
        df_out["Setor"] = np.where(
            df_out["Categoria_emprego"].isin([5, 6, 7]), "Público", "Privado"
        )

    # Formalidade no Trabalho
    if "Categoria_emprego" in df_out.columns and "Contribuicao_previdencia" in df_out.columns:
        formal_mask = (
            df_out["Categoria_emprego"].isin([1, 3, 5, 7])
            | (df_out["Categoria_emprego"].isin([8, 9]) & (df_out["Contribuicao_previdencia"] == 1))
        )
        df_out["Formalidade"] = np.where(formal_mask, "Formal", "Informal")

    # Faixa Etária
    if "Idade" in df_out.columns:
        df_out["Faixa_Etaria"] = pd.cut(
            df_out["Idade"],
            bins=[24, 34, 44, 54, 65],
            labels=["25–34", "35–44", "45–54", "55–65"],
        )

    # Nível Educacional (Anos de Estudo)
    if "Anos_de_Estudo" in df_out.columns:
        df_out["Escolaridade_Nivel"] = pd.cut(
            df_out["Anos_de_Estudo"],
            bins=[-1, 4, 8, 11, 20],
            labels=[
                "Até Fundamental Incompleto",
                "Fundamental Completo",
                "Médio Completo",
                "Superior Completo",
            ],
        )

    # Atividade Econômica
    if "Grupo_atv_princ_empreedimento" in df_out.columns:
        df_out["Atividade_Desc"] = df_out["Grupo_atv_princ_empreedimento"].map(ATIVIDADES_MAP).fillna("Outras")

    return df_out


def calcular_series_trimestrais_salario_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as séries trimestrais do salário-hora médio por categorias.

    Recria a tabela consolidada histórica com as desagregações por Sexo, Setor,
    Formalidade, Região e Brasil.
    """
    periodos = sorted(df["Periodo"].unique())
    linhas = []

    for per in periodos:
        sub = df[df["Periodo"] == per]
        sh_val = sub["Salario_Hora"].dropna()

        # Médias gerais e por subgrupos
        sh_brasil = sh_val.mean()
        sh_homem = sub.loc[sub["Sexo"] == 1, "Salario_Hora"].mean()
        sh_mulher = sub.loc[sub["Sexo"] == 2, "Salario_Hora"].mean()
        sh_publico = sub.loc[sub["Setor"] == "Público", "Salario_Hora"].mean()
        sh_privado = sub.loc[sub["Setor"] == "Privado", "Salario_Hora"].mean()
        sh_formal = sub.loc[sub["Formalidade"] == "Formal", "Salario_Hora"].mean()
        sh_informal = sub.loc[sub["Formalidade"] == "Informal", "Salario_Hora"].mean()

        row = {
            "Periodo": per,
            "Brasil": round(sh_brasil, 2),
            "Homem": round(sh_homem, 2),
            "Mulher": round(sh_mulher, 2),
            "Gap_Genero_Pct": round(((sh_homem - sh_mulher) / sh_homem) * 100, 2) if sh_homem else np.nan,
            "Publico": round(sh_publico, 2),
            "Privado": round(sh_privado, 2),
            "Premio_Publico_Pct": round(((sh_publico - sh_privado) / sh_privado) * 100, 2) if sh_privado else np.nan,
            "Formal": round(sh_formal, 2),
            "Informal": round(sh_informal, 2),
            "Premio_Formal_Pct": round(((sh_formal - sh_informal) / sh_informal) * 100, 2) if sh_informal else np.nan,
        }

        # Regiões
        for reg in ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]:
            sh_reg = sub.loc[sub["Regiao"] == reg, "Salario_Hora"].mean()
            row[reg] = round(sh_reg, 2)

        linhas.append(row)

    return pd.DataFrame(linhas)


def calcular_series_rendimento_desigualdade(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a evolução trimestral dos percentis e métricas de desigualdade do rendimento habitual real."""
    periodos = sorted(df["Periodo"].unique())
    linhas = []

    for per in periodos:
        sub = df.loc[df["Periodo"] == per, "Rendimento_hab_Trab_princ"].dropna()
        if sub.empty:
            continue

        p10 = sub.quantile(0.10)
        p25 = sub.quantile(0.25)
        p50 = sub.quantile(0.50)  # Mediana
        p75 = sub.quantile(0.75)
        p90 = sub.quantile(0.90)
        media = sub.mean()

        razao_90_10 = p90 / p10 if p10 > 0 else np.nan
        razao_90_50 = p90 / p50 if p50 > 0 else np.nan

        linhas.append({
            "Periodo": per,
            "Media": round(media, 2),
            "P10": round(p10, 2),
            "P25": round(p25, 2),
            "Mediana_P50": round(p50, 2),
            "P75": round(p75, 2),
            "P90": round(p90, 2),
            "Razao_90_10": round(razao_90_10, 2),
            "Razao_90_50": round(razao_90_50, 2),
        })

    return pd.DataFrame(linhas)


def calcular_series_taxa_formalidade(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a taxa de formalidade (%) trimestral por macrorregião e total Brasil."""
    periodos = sorted(df["Periodo"].unique())
    linhas = []

    for per in periodos:
        sub = df[df["Periodo"] == per]
        total = len(sub)
        if total == 0:
            continue

        taxa_brasil = (sub["Formalidade"] == "Formal").mean() * 100

        row = {
            "Periodo": per,
            "Brasil": round(taxa_brasil, 2),
        }

        for reg in ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]:
            sub_reg = sub[sub["Regiao"] == reg]
            taxa_reg = (sub_reg["Formalidade"] == "Formal").mean() * 100 if len(sub_reg) > 0 else np.nan
            row[reg] = round(taxa_reg, 2)

        linhas.append(row)

    return pd.DataFrame(linhas)


def calcular_retorno_educacao(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a evolução do salário-hora médio por nível educacional."""
    periodos = sorted(df["Periodo"].unique())
    niveis = [
        "Até Fundamental Incompleto",
        "Fundamental Completo",
        "Médio Completo",
        "Superior Completo",
    ]
    linhas = []

    for per in periodos:
        sub = df[df["Periodo"] == per]
        row = {"Periodo": per}
        for niv in niveis:
            sub_niv = sub.loc[sub["Escolaridade_Nivel"] == niv, "Salario_Hora"].dropna()
            row[niv] = round(sub_niv.mean(), 2) if not sub_niv.empty else np.nan
        linhas.append(row)

    return pd.DataFrame(linhas)
