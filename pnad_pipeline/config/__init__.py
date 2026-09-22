"""Módulo de configuração do pacote pnad_pipeline."""

from .settings import (
    BASE_DIR,
    DATA_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DEFAULT_PARQUET_PATH,
    DEFAULT_STATA_PATH,
    IPCA_BCB_URL,
    DATA_BASE_DEFLACAO,
    DEFAULT_ANO_INICIO,
    DEFAULT_ANO_FIM,
    LOG_LEVEL,
    setup_logging,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "DEFAULT_PARQUET_PATH",
    "DEFAULT_STATA_PATH",
    "IPCA_BCB_URL",
    "DATA_BASE_DEFLACAO",
    "DEFAULT_ANO_INICIO",
    "DEFAULT_ANO_FIM",
    "LOG_LEVEL",
    "setup_logging",
]
