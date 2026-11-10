"""Testes unitários para o cálculo de fator acumulado do IPCA."""

import pandas as pd
import pytest

from pnad_pipeline.extracao.ipca import calcular_fator_acumulado, calcular_fator_para_base


def test_conversao_de_trimestre_posterior_preserva_a_convencao_da_base():
    ipca = pd.DataFrame({"data": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
                         "valor": [0.5, 0.7, 0.9]})
    assert calcular_fator_para_base(ipca, "01/03/2026", "01/01/2026") == pytest.approx(1 / 1.007)
    assert calcular_fator_para_base(ipca, "01/02/2026", "01/01/2026") == 1
    assert calcular_fator_para_base(ipca, "01/01/2026", "01/03/2026") == pytest.approx(1.005 * 1.007 * 1.009)
    with pytest.raises(ValueError, match="IPCA mensal incompleto"):
        calcular_fator_para_base(ipca.iloc[[0, 2]], "01/03/2026", "01/01/2026")


@pytest.fixture
def df_ipca_sintetico() -> pd.DataFrame:
    """Retorna uma série sintética do IPCA para testes controlados."""
    datas = ["01/01/2020", "01/02/2020", "01/03/2020", "01/04/2020"]
    valores = [1.0, 2.0, 0.5, 3.0]  # em percentual (%)
    return pd.DataFrame({
        "data": pd.to_datetime(datas, dayfirst=True),
        "valor": valores,
    })


def test_calcular_fator_acumulado_intervalo_completo(df_ipca_sintetico: pd.DataFrame):
    """Testa o produto dos fatores em todo o intervalo."""
    # (1 + 0.01) * (1 + 0.02) * (1 + 0.005) * (1 + 0.03)
    esperado = 1.01 * 1.02 * 1.005 * 1.03
    fator = calcular_fator_acumulado(df_ipca_sintetico, "01/01/2020", "01/04/2020")
    assert pytest.approx(fator, rel=1e-6) == esperado


def test_calcular_fator_acumulado_subintervalo(df_ipca_sintetico: pd.DataFrame):
    """Testa o produto em um subconjunto de meses (fev e mar)."""
    esperado = 1.02 * 1.005
    fator = calcular_fator_acumulado(df_ipca_sintetico, "01/02/2020", "01/03/2020")
    assert pytest.approx(fator, rel=1e-6) == esperado


def test_calcular_fator_acumulado_unico_mes(df_ipca_sintetico: pd.DataFrame):
    """Testa para apenas um mês."""
    esperado = 1.02
    fator = calcular_fator_acumulado(df_ipca_sintetico, "01/02/2020", "01/02/2020")
    assert pytest.approx(fator, rel=1e-6) == esperado


def test_calcular_fator_acumulado_data_inicio_maior_que_fim(df_ipca_sintetico: pd.DataFrame):
    """Deve levantar ValueError se data_inicio for posterior a data_fim."""
    with pytest.raises(ValueError, match="não pode ser posterior"):
        calcular_fator_acumulado(df_ipca_sintetico, "01/04/2020", "01/01/2020")


def test_calcular_fator_acumulado_intervalo_sem_dados(df_ipca_sintetico: pd.DataFrame):
    """Deve levantar ValueError se nenhuma data do DataFrame estiver no intervalo."""
    with pytest.raises(ValueError, match="Nenhum dado de IPCA encontrado"):
        calcular_fator_acumulado(df_ipca_sintetico, "01/01/2021", "01/06/2021")
