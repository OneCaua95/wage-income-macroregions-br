"""Testes unitários para os filtros de qualidade da PNAD Contínua."""

import numpy as np
import pandas as pd
import pytest

from pnad_pipeline.extracao.filtros import (
    aplicar_filtros,
    filtrar_idade,
    filtrar_renda_positiva,
    remover_salarios_negativos,
)


@pytest.fixture
def df_pnad_sintetico() -> pd.DataFrame:
    """Retorna DataFrame sintético com casos de borda para testes de filtro."""
    return pd.DataFrame({
        "Idade": [20, 25, 30, 65, 70, 45, 50],
        "Rendimento_hab_Trab_princ": [1500.0, 2000.0, -500.0, 3000.0, 4000.0, 0.0, np.nan],
        "UF": [35, 33, 31, 41, 21, 52, 11],
    })


def test_filtrar_renda_positiva(df_pnad_sintetico: pd.DataFrame):
    """Deve manter apenas observações com rendimento estritamente positivo (> 0)."""
    df_result = filtrar_renda_positiva(df_pnad_sintetico)
    assert len(df_result) == 4  # 1500.0, 2000.0, 3000.0, 4000.0
    assert (df_result["Rendimento_hab_Trab_princ"] > 0).all()


def test_filtrar_idade(df_pnad_sintetico: pd.DataFrame):
    """Deve manter apenas observações com 25 <= Idade <= 65."""
    df_result = filtrar_idade(df_pnad_sintetico, min_idade=25, max_idade=65)
    assert len(df_result) == 5  # idades 25, 30, 65, 45, 50
    assert df_result["Idade"].min() == 25
    assert df_result["Idade"].max() == 65


def test_remover_salarios_negativos():
    """Deve remover salários negativos e manter salários >= 0."""
    df = pd.DataFrame({
        "Rendimento_hab_Trab_princ": [-100.0, 0.0, 1500.0, -50.0, 2000.0]
    })
    df_result = remover_salarios_negativos(df)
    assert len(df_result) == 3
    assert (df_result["Rendimento_hab_Trab_princ"] >= 0).all()


def test_aplicar_filtros_combinado(df_pnad_sintetico: pd.DataFrame):
    """Testa o pipeline completo de filtros em conjunto."""
    df_result = aplicar_filtros(df_pnad_sintetico)
    # No df sintético:
    # 0: Idade 20 (fora)
    # 1: Idade 25, Renda 2000.0 (mantido)
    # 2: Idade 30, Renda -500.0 (fora por renda <= 0)
    # 3: Idade 65, Renda 3000.0 (mantido)
    # 4: Idade 70 (fora por idade > 65)
    # 5: Idade 45, Renda 0.0 (fora por renda == 0)
    # 6: Idade 50, Renda NaN (fora por renda NaN)
    assert len(df_result) == 2
    assert set(df_result["Idade"]) == {25, 65}
    assert (df_result["Rendimento_hab_Trab_princ"] > 0).all()
