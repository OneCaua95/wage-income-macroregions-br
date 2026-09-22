"""Módulo para aplicação do deflator do IPCA sobre as variáveis de renda da PNAD Contínua."""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def calcular_data_referencia_trimestre(ano: int, trimestre: int) -> str:
    """Retorna a data de referência no formato DD/MM/AAAA para o final do trimestre.

    O mês final do trimestre é trimestre * 3 (T1 -> 03, T2 -> 06, T3 -> 09, T4 -> 12).

    Parameters
    ----------
    ano : int
        Ano de referência.
    trimestre : int
        Trimestre (1 a 4).

    Returns
    -------
    str
        Data no formato '01/MM/AAAA'.
    """
    if not (1 <= trimestre <= 4):
        raise ValueError(f"Trimestre inválido: {trimestre}. Deve estar entre 1 e 4.")
    mes_final = trimestre * 3
    return f"01/{mes_final:02d}/{ano}"


def aplicar_deflacao(
    df: pd.DataFrame,
    fator: float,
    col_renda: str = "Rendimento_hab_Trab_princ",
) -> pd.DataFrame:
    """Aplica o fator acumulado do IPCA multiplicando a coluna de rendimento especificada.

    Esta é uma função pura: retorna uma cópia com a coluna deflacionada, sem alterar
    o DataFrame original.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna de rendimento.
    fator : float
        Fator deflator acumulado (ex.: 1.5432). Deve ser estritamente positivo.
    col_renda : str, default "Rendimento_hab_Trab_princ"
        Nome da coluna a ter seus valores deflacionados.

    Returns
    -------
    pd.DataFrame
        Cópia do DataFrame com a coluna de renda deflacionada.

    Raises
    ------
    KeyError
        Se col_renda não estiver presente no DataFrame.
    ValueError
        Se o fator for menor ou igual a zero.
    """
    if col_renda not in df.columns:
        raise KeyError(f"Coluna '{col_renda}' não encontrada no DataFrame.")
    if fator <= 0:
        raise ValueError(f"Fator de deflação inválido: {fator}. Deve ser maior que zero.")

    df_out = df.copy()
    df_out[col_renda] = df_out[col_renda] * fator
    logger.debug("Deflação aplicada na coluna '%s' com fator %.6f.", col_renda, fator)
    return df_out
