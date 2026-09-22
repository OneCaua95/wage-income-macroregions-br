"""Testes unitários para o módulo de deflação."""

import pandas as pd
import pytest

from pnad_pipeline.extracao.deflacao import (
    aplicar_deflacao,
    calcular_data_referencia_trimestre,
)


def test_calcular_data_referencia_trimestre():
    """Valida o cálculo do mês final para cada trimestre."""
    assert calcular_data_referencia_trimestre(2023, 1) == "01/03/2023"
    assert calcular_data_referencia_trimestre(2023, 2) == "01/06/2023"
    assert calcular_data_referencia_trimestre(2023, 3) == "01/09/2023"
    assert calcular_data_referencia_trimestre(2023, 4) == "01/12/2023"


def test_calcular_data_referencia_trimestre_invalido():
    """Deve levantar ValueError para trimestres fora de [1, 4]."""
    with pytest.raises(ValueError, match="Trimestre inválido"):
        calcular_data_referencia_trimestre(2023, 0)
    with pytest.raises(ValueError, match="Trimestre inválido"):
        calcular_data_referencia_trimestre(2023, 5)


def test_aplicar_deflacao():
    """Valida a multiplicação da renda pelo fator sem mutação do original."""
    df_original = pd.DataFrame({
        "Id": [1, 2],
        "Rendimento_hab_Trab_princ": [1000.0, 2500.0],
    })
    fator = 1.5

    df_deflacionado = aplicar_deflacao(df_original, fator=fator)

    # Verifica multiplicação correta
    assert df_deflacionado["Rendimento_hab_Trab_princ"].tolist() == [1500.0, 3750.0]
    # Verifica que não mutou o original (função pura)
    assert df_original["Rendimento_hab_Trab_princ"].tolist() == [1000.0, 2500.0]
    # Outras colunas intactas
    assert df_deflacionado["Id"].tolist() == [1, 2]


def test_aplicar_deflacao_coluna_inexistente():
    """Deve levantar KeyError se a coluna informada não existir."""
    df = pd.DataFrame({"Outra_Coluna": [100.0]})
    with pytest.raises(KeyError, match="não encontrada"):
        aplicar_deflacao(df, fator=1.2, col_renda="Rendimento_hab_Trab_princ")


def test_aplicar_deflacao_fator_invalido():
    """Deve levantar ValueError se o fator for <= 0."""
    df = pd.DataFrame({"Rendimento_hab_Trab_princ": [1000.0]})
    with pytest.raises(ValueError, match="Fator de deflação inválido"):
        aplicar_deflacao(df, fator=0.0)
    with pytest.raises(ValueError, match="Fator de deflação inválido"):
        aplicar_deflacao(df, fator=-1.5)
