"""Módulo para mapeamento geográfico das Unidades da Federação (UF) em Macrorregiões."""

from __future__ import annotations

import logging
from typing import Optional
import pandas as pd

from .constants import UF_PARA_REGIAO

logger = logging.getLogger(__name__)


def mapear_regiao(
    df: pd.DataFrame,
    col_uf: str = "UF",
    col_destino: str = "Regiao",
    mapa_uf: Optional[dict[int, str]] = None,
) -> pd.DataFrame:
    """Mapeia os códigos de UF do IBGE para suas respectivas Grandes Regiões brasileiras.

    Esta é uma função pura: retorna uma cópia do DataFrame com a nova coluna adicionada.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna de UF.
    col_uf : str, default "UF"
        Nome da coluna que contém os códigos numéricos da UF.
    col_destino : str, default "Regiao"
        Nome da coluna que receberá o nome da Macrorregião.
    mapa_uf : dict[int, str], optional
        Dicionário customizado de mapeamento UF -> Região. Se None, utiliza UF_PARA_REGIAO.

    Returns
    -------
    pd.DataFrame
        Cópia do DataFrame com a coluna de região mapeada.

    Raises
    ------
    KeyError
        Se col_uf não existir no DataFrame.
    """
    if col_uf not in df.columns:
        raise KeyError(f"Coluna de UF '{col_uf}' não encontrada no DataFrame.")

    mapa = mapa_uf or UF_PARA_REGIAO
    df_out = df.copy()

    # Garante que os códigos UF sejam inteiros para casar com as chaves do dicionário
    uf_series = pd.to_numeric(df_out[col_uf], errors="coerce").astype("Int64")
    df_out[col_destino] = uf_series.map(mapa)

    nao_mapeados = int(df_out[col_destino].isna().sum())
    if nao_mapeados > 0:
        logger.warning(
            "⚠️ %d observações possuem códigos de UF não reconhecidos no mapeamento.",
            nao_mapeados,
        )
    else:
        logger.debug("Regiões mapeadas com sucesso para todas as observações.")

    return df_out
