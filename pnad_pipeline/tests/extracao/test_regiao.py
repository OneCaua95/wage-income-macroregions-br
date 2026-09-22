"""Testes unitários para o mapeamento geográfico de UF em Macrorregiões."""

import pandas as pd
import pytest

from pnad_pipeline.extracao.regiao import mapear_regiao


def test_mapear_regiao_todas_as_macroregioes():
    """Valida o mapeamento de pelo menos um estado de cada região brasileira."""
    df = pd.DataFrame({
        "UF": [
            11,  # Rondônia -> Norte
            29,  # Bahia -> Nordeste
            35,  # São Paulo -> Sudeste
            43,  # Rio Grande do Sul -> Sul
            53,  # Distrito Federal -> Centro-Oeste
        ]
    })

    df_resultado = mapear_regiao(df)
    assert "Regiao" in df_resultado.columns
    assert df_resultado["Regiao"].tolist() == [
        "Norte",
        "Nordeste",
        "Sudeste",
        "Sul",
        "Centro-Oeste",
    ]


def test_mapear_regiao_uf_desconhecida():
    """Valida que UFs não mapeadas resultem em NaN sem quebrar a execução."""
    df = pd.DataFrame({"UF": [99, 35]})
    df_resultado = mapear_regiao(df)
    assert pd.isna(df_resultado.loc[0, "Regiao"])
    assert df_resultado.loc[1, "Regiao"] == "Sudeste"


def test_mapear_regiao_coluna_inexistente():
    """Deve levantar KeyError se a coluna de UF não existir."""
    df = pd.DataFrame({"Codigo": [35]})
    with pytest.raises(KeyError, match="não encontrada"):
        mapear_regiao(df, col_uf="UF")


def test_mapear_regiao_funcao_pura():
    """Garante que o DataFrame original não seja alterado."""
    df_original = pd.DataFrame({"UF": [35]})
    _ = mapear_regiao(df_original)
    assert "Regiao" not in df_original.columns
