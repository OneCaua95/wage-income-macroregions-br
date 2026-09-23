"""Testes unitários para as funções analíticas e descritivas."""

import pandas as pd
import pytest

from pnad_pipeline.analise.descritiva import (
    calcular_retorno_educacao,
    calcular_series_rendimento_desigualdade,
    calcular_series_taxa_formalidade,
    calcular_series_trimestrais_salario_hora,
    preparar_variaveis_analise,
)


@pytest.fixture
def df_amostra_pnad() -> pd.DataFrame:
    """Retorna DataFrame sintético mínimo com as variáveis-chave da PNAD."""
    return pd.DataFrame({
        "Ano": [2023, 2023, 2023, 2023],
        "Trimestre": [1, 1, 1, 1],
        "Sexo": [1, 2, 1, 2],
        "Cor": [1, 4, 2, 1],
        "Idade": [30, 40, 50, 60],
        "Anos_de_Estudo": [4, 11, 15, 8],
        "Categoria_emprego": [1, 2, 5, 6],
        "Contribuicao_previdencia": [1, 2, 1, 2],
        "Rendimento_hab_Trab_princ": [2000.0, 1500.0, 6000.0, 3000.0],
        "Horas_hab_trabalhadas": [40.0, 30.0, 40.0, 20.0],
        "Regiao": ["Sudeste", "Nordeste", "Sul", "Centro-Oeste"],
    })


def test_preparar_variaveis_analise(df_amostra_pnad: pd.DataFrame):
    """Valida a criação das variáveis derivadas (Salario_Hora, Formalidade, Setor, etc.)."""
    df_prep = preparar_variaveis_analise(df_amostra_pnad)

    assert "Periodo" in df_prep.columns
    assert df_prep["Periodo"].iloc[0] == "2023.1"
    assert "Salario_Hora" in df_prep.columns
    assert (df_prep["Salario_Hora"] > 0).all()

    assert set(df_prep["Sexo_Desc"]) == {"Homem", "Mulher"}
    assert set(df_prep["Setor"]) == {"Privado", "Público"}
    assert set(df_prep["Formalidade"]) == {"Formal", "Informal"}
    assert "Escolaridade_Nivel" in df_prep.columns


def test_preparar_variaveis_filtra_idade_inclusive(df_amostra_pnad: pd.DataFrame):
    """A preparação mantém os limites 25 e 65 e remove as demais idades."""
    df = pd.concat(
        [
            df_amostra_pnad,
            df_amostra_pnad.iloc[[0]].assign(Idade=24),
            df_amostra_pnad.iloc[[0]].assign(Idade=66),
        ],
        ignore_index=True,
    )

    df_prep = preparar_variaveis_analise(df)

    assert len(df_prep) == len(df_amostra_pnad)
    assert df_prep["Idade"].between(25, 65).all()


def test_calcular_series_trimestrais_salario_hora(df_amostra_pnad: pd.DataFrame):
    """Valida o cálculo das agregações por período e grupos."""
    df_prep = preparar_variaveis_analise(df_amostra_pnad)
    df_series = calcular_series_trimestrais_salario_hora(df_prep)

    assert len(df_series) == 1
    assert "Brasil" in df_series.columns
    assert "Homem" in df_series.columns
    assert "Mulher" in df_series.columns
    assert "Publico" in df_series.columns
    assert "Privado" in df_series.columns
    assert df_series["Brasil"].iloc[0] > 0


def test_calcular_series_rendimento_desigualdade(df_amostra_pnad: pd.DataFrame):
    """Valida cálculo de percentis e razão 90/10."""
    df_prep = preparar_variaveis_analise(df_amostra_pnad)
    df_desig = calcular_series_rendimento_desigualdade(df_prep)

    assert len(df_desig) == 1
    row = df_desig.iloc[0]
    assert row["P10"] <= row["Mediana_P50"] <= row["P90"]
    assert row["Razao_90_10"] >= 1.0


def test_calcular_series_taxa_formalidade(df_amostra_pnad: pd.DataFrame):
    """Valida cálculo da taxa de formalidade."""
    df_prep = preparar_variaveis_analise(df_amostra_pnad)
    df_form = calcular_series_taxa_formalidade(df_prep)

    assert len(df_form) == 1
    # 2 formais e 2 informais = 50%
    assert df_form["Brasil"].iloc[0] == 50.0
