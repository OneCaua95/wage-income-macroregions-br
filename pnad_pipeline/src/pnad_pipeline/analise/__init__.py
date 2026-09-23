"""Módulo de análise e modelagem estatística/econométrica da PNAD Contínua."""

from .descritiva import (
    FONTE_DESCRITIVA,
    ATIVIDADES_MAP,
    IDADE_MINIMA_ANALISE,
    IDADE_MAXIMA_ANALISE,
    VARIAVEIS_DICIONARIO_ANALISE,
    preparar_variaveis_analise,
    calcular_series_trimestrais_salario_hora,
    calcular_series_rendimento_desigualdade,
    calcular_series_taxa_formalidade,
    calcular_retorno_educacao,
)

__all__ = [
    "FONTE_DESCRITIVA",
    "ATIVIDADES_MAP",
    "IDADE_MINIMA_ANALISE",
    "IDADE_MAXIMA_ANALISE",
    "VARIAVEIS_DICIONARIO_ANALISE",
    "preparar_variaveis_analise",
    "calcular_series_trimestrais_salario_hora",
    "calcular_series_rendimento_desigualdade",
    "calcular_series_taxa_formalidade",
    "calcular_retorno_educacao",
]
