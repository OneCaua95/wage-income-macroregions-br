"""Módulo de obtenção e cálculo de fatores do IPCA via Banco Central do Brasil (SGS)."""

from __future__ import annotations

import logging
from typing import Optional, Union
import numpy as np
import pandas as pd

from .constants import IPCA_SGS_URL_PADRAO

logger = logging.getLogger(__name__)


def get_ipca_bc(url: Optional[str] = None) -> pd.DataFrame:
    """Obtém a série histórica do IPCA mensal via API do Banco Central (SGS código 433).

    Parameters
    ----------
    url : str, optional
        URL do endpoint da API do SGS BCB. Se omitido, usa o padrão do SGS 433.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo as colunas 'data' (datetime64[ns]) e 'valor' (float).
    """
    endpoint = url or IPCA_SGS_URL_PADRAO
    logger.info("Obtendo dados do IPCA via Banco Central (SGS): %s", endpoint)

    try:
        df = pd.read_json(endpoint)
    except Exception as e:
        logger.error("Falha ao consultar API do BCB: %s", e, exc_info=True)
        raise

    if "data" not in df.columns or "valor" not in df.columns:
        raise ValueError("DataFrame do IPCA retornado pela API não contém as colunas esperadas ('data', 'valor').")

    df = df.copy()
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    logger.info("Série do IPCA carregada com sucesso (%d registros de %s a %s).",
                len(df), df["data"].min().strftime("%d/%m/%Y"), df["data"].max().strftime("%d/%m/%Y"))
    return df


def calcular_fator_acumulado(
    df_ipca: pd.DataFrame,
    data_inicio: Union[str, pd.Timestamp],
    data_fim: Union[str, pd.Timestamp],
) -> float:
    """Calcula o fator de deflação acumulado do IPCA entre duas datas (inclusive).

    Multiplica os fatores mensais (1 + valor/100) no intervalo fechado [data_inicio, data_fim].

    Parameters
    ----------
    df_ipca : pd.DataFrame
        DataFrame com colunas 'data' e 'valor' (IPCA mensal em percentual).
    data_inicio : str ou pd.Timestamp
        Data inicial no formato aceito pelo pandas (ex.: '01/03/2013' ou Timestamp).
    data_fim : str ou pd.Timestamp
        Data final (data-base) no formato aceito pelo pandas (ex.: '01/01/2026').

    Returns
    -------
    float
        Fator acumulado da inflação no intervalo.

    Raises
    ------
    ValueError
        Se nenhuma observação for encontrada no intervalo especificado.
    """
    dt_inicio = pd.to_datetime(data_inicio, dayfirst=True)
    dt_fim = pd.to_datetime(data_fim, dayfirst=True)

    if dt_inicio > dt_fim:
        raise ValueError(f"data_inicio ({dt_inicio}) não pode ser posterior a data_fim ({dt_fim}).")

    mascara = (df_ipca["data"] >= dt_inicio) & (df_ipca["data"] <= dt_fim)
    intervalo = df_ipca.loc[mascara].copy()

    if intervalo.empty:
        raise ValueError(
            f"Nenhum dado de IPCA encontrado no intervalo de {dt_inicio.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}."
        )

    intervalo["fator_mensal"] = 1.0 + (intervalo["valor"] / 100.0)
    fator_acumulado = float(intervalo["fator_mensal"].prod())

    logger.debug(
        "Fator acumulado IPCA de %s até %s (%d meses): %.6f",
        dt_inicio.strftime("%d/%m/%Y"),
        dt_fim.strftime("%d/%m/%Y"),
        len(intervalo),
        fator_acumulado,
    )
    return fator_acumulado


def calcular_fator_para_base(df_ipca: pd.DataFrame, data_inicio: str, data_base: str) -> float:
    """Estende a convenção inclusiva para fontes posteriores à data-base.

    O fator histórico equivale a índice(base) / índice(mês anterior à fonte).
    Para uma fonte posterior, inverte a inflação entre esses dois índices.
    """
    inicio = pd.to_datetime(data_inicio, dayfirst=True)
    base = pd.to_datetime(data_base, dayfirst=True)
    if inicio <= base:
        return calcular_fator_acumulado(df_ipca, inicio, base)
    meses = pd.date_range(base + pd.offsets.MonthBegin(), inicio - pd.offsets.MonthBegin(), freq="MS")
    taxas = df_ipca.set_index("data")["valor"].reindex(meses).to_numpy(dtype=float)
    if not np.isfinite(taxas).all() or np.any(taxas <= -100):
        raise ValueError("IPCA mensal incompleto ou inválido para a conversão à data-base.")
    return float(1 / np.prod(1 + taxas / 100))
