"""Módulo responsável exclusivamente pelo download de dados brutos da PNAD Contínua via pnadium."""

from __future__ import annotations

import logging
from typing import Sequence
import pandas as pd

logger = logging.getLogger(__name__)


class PNADExtractor:
    """Extrator de dados da PNAD Contínua via pacote pnadium."""

    def download_data(self, year: int, quarter: int, cols: Sequence[str]) -> pd.DataFrame:
        """Baixa os microdados da PNAD Contínua para um dado ano e trimestre.

        Parameters
        ----------
        year : int
            Ano de referência (ex.: 2013).
        quarter : int
            Trimestre de referência (1 a 4).
        cols : Sequence[str]
            Lista de nomes originais de variáveis a serem baixadas.

        Returns
        -------
        pd.DataFrame
            DataFrame com os microdados brutos do trimestre.
        """
        logger.info("Iniciando download PNAD Contínua %dT%d (%d colunas)...", year, quarter, len(cols))
        try:
            import pnadium  # Importação atrasada para não falhar caso ambiente de teste não tenha pnadium instalado
            df = pnadium.trimestral.download(ano=year, t=quarter, colunas=list(cols))
            logger.info("Download concluído para %dT%d com shape %s.", year, quarter, df.shape)
            return df
        except Exception as e:
            logger.error("Erro no download de %dT%d via pnadium: %s", year, quarter, e, exc_info=True)
            raise


def download_pnad(year: int, quarter: int, cols: Sequence[str]) -> pd.DataFrame:
    """Função utilitária direta para download dos microdados da PNAD Contínua."""
    extractor = PNADExtractor()
    return extractor.download_data(year, quarter, cols)
