"""Testes unitários para a orquestração da coleta e do loop de períodos."""

import pandas as pd
import pytest

from pnad_pipeline.extracao.coleta import coletar_pnad, processar_periodo


class MockPNADExtractor:
    """Mock do extrator para testes unitários rápidos e sem rede."""

    def __init__(self, falhar_em: tuple[int, int] | None = None):
        self.falhar_em = falhar_em

    def download_data(self, year: int, quarter: int, cols: list[str]) -> pd.DataFrame:
        if self.falhar_em and (year, quarter) == self.falhar_em:
            raise RuntimeError(f"Erro simulado para {year}T{quarter}")

        return pd.DataFrame({
            "Ano": [year, year],
            "Trimestre": [quarter, quarter],
            "UF": [35, 21],  # SP, MA
            "V2009": [30, 40],  # Idade
            "VD4016": [2500.0, 1800.0],  # Rendimento_hab_Trab_princ
            "V2007": [1, 2],  # Sexo
        })


@pytest.fixture
def df_ipca_teste() -> pd.DataFrame:
    """Série mínima de IPCA para cobrir trimestres de teste."""
    return pd.DataFrame({
        "data": pd.to_datetime(["01/03/2023", "01/06/2023", "01/01/2026"], dayfirst=True),
        "valor": [0.5, 0.4, 0.3],
    })


def test_processar_periodo_sucesso(df_ipca_teste: pd.DataFrame):
    """Valida o processamento individual de um trimestre com dados sintéticos."""
    extractor = MockPNADExtractor()
    df_res, periodo, fator = processar_periodo(
        ano=2023,
        trimestre=1,
        df_ipca=df_ipca_teste,
        extractor=extractor,
        data_base_deflacao="01/01/2026",
    )

    assert periodo == "2023.1"
    assert fator > 1.0
    assert len(df_res) == 2
    # Verifica renomeações
    assert "Idade" in df_res.columns
    assert "Rendimento_hab_Trab_princ" in df_res.columns
    # Verifica região mapeada
    assert "Regiao" in df_res.columns
    assert set(df_res["Regiao"]) == {"Sudeste", "Nordeste"}


def test_coletar_pnad_com_tolerancia_a_falha(df_ipca_teste: pd.DataFrame):
    """Valida que uma falha em um trimestre é capturada sem interromper o restante do loop."""
    # Falha apenas em 2023T2
    extractor = MockPNADExtractor(falhar_em=(2023, 2))

    df_consolidado, erros = coletar_pnad(
        ano_inicio=2023,
        ano_fim=2023,
        df_ipca=df_ipca_teste,
        extractor=extractor,
        data_base_deflacao="01/01/2026",
    )

    # 4 trimestres no ano: T1, T3 e T4 com sucesso, T2 com falha
    assert len(erros) == 1
    assert "2023T2" in erros[0]
    # 3 trimestres * 2 observações = 6
    assert len(df_consolidado) == 6
