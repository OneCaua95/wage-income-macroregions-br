"""Testes unitários para o módulo de otimização de memória RAM e downcasting de tipos."""

import numpy as np
import pandas as pd
import pytest

from pnad_pipeline.extracao.otimizacao import calcular_memoria_mb, otimizar_tipos


@pytest.fixture
def df_amostra_pesada() -> pd.DataFrame:
    """Retorna DataFrame com tipos padrão pesados do pandas (int64, float64, object)."""
    n = 1000
    return pd.DataFrame({
        "Ano": np.full(n, 2023, dtype="int64"),
        "Trimestre": np.full(n, 1, dtype="int64"),
        "UF": np.random.choice([35, 33, 31, 41, 52], size=n).astype("int64"),
        "Sexo": np.random.choice([1, 2], size=n).astype("int64"),
        "Idade": np.random.randint(25, 65, size=n).astype("int64"),
        "Rendimento_hab_Trab_princ": np.random.uniform(1500.0, 12000.0, size=n).astype("float64"),
        "Horas_hab_trabalhadas": np.random.uniform(20.0, 44.0, size=n).astype("float64"),
        "Regiao": np.random.choice(["Sudeste", "Sul", "Centro-Oeste"], size=n).astype("object"),
    })


def test_otimizar_tipos_reduz_memoria(df_amostra_pesada: pd.DataFrame):
    """Verifica se a conversão de tipos reduz significativamente a memória em RAM."""
    mem_antes = calcular_memoria_mb(df_amostra_pesada)
    df_opt = otimizar_tipos(df_amostra_pesada)
    mem_depois = calcular_memoria_mb(df_opt)

    assert mem_depois < mem_antes
    assert (mem_antes - mem_depois) / mem_antes > 0.40  # Pelo menos 40% de redução no conjunto sintético


def test_otimizar_tipos_preserva_valores(df_amostra_pesada: pd.DataFrame):
    """Garante que os valores não sofram perda ou distorção indevida após o cast."""
    df_opt = otimizar_tipos(df_amostra_pesada)

    assert (df_opt["Ano"] == df_amostra_pesada["Ano"]).all()
    assert (df_opt["Idade"] == df_amostra_pesada["Idade"]).all()
    assert (df_opt["UF"] == df_amostra_pesada["UF"]).all()
    assert (df_opt["Regiao"] == df_amostra_pesada["Regiao"]).all()
    # Floats tolerância numérica de precisão float32
    assert np.allclose(
        df_opt["Rendimento_hab_Trab_princ"],
        df_amostra_pesada["Rendimento_hab_Trab_princ"],
        rtol=1e-5,
    )


def test_otimizar_tipos_dtypes_especificos(df_amostra_pesada: pd.DataFrame):
    """Verifica se as colunas assumiram os tipos compactos esperados."""
    df_opt = otimizar_tipos(df_amostra_pesada)

    assert df_opt["Ano"].dtype.name == "int16"
    assert df_opt["Trimestre"].dtype.name == "int8"
    assert df_opt["Idade"].dtype.name == "int8"
    assert df_opt["Rendimento_hab_Trab_princ"].dtype.name == "float32"
    assert df_opt["Regiao"].dtype.name == "category"
