"""Configurações centrais do pipeline PNAD Contínua + IPCA.

Carrega configurações a partir de variáveis de ambiente (.env) com fallbacks
locais e define caminhos padrão de arquivos e parâmetros da coleta.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    # Carrega .env do diretório base do pacote se existir
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
    else:
        load_dotenv()
except ImportError:
    # Se python-dotenv não estiver instalado, utiliza as variáveis do ambiente do SO
    pass

# Raiz do projeto (diretório pnad_pipeline/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Diretórios de dados
DATA_DIR: Path = Path(os.getenv("PNAD_DATA_DIR", str(BASE_DIR / "data")))
DATA_RAW_DIR: Path = Path(os.getenv("PNAD_DATA_RAW_DIR", str(DATA_DIR / "raw")))
DATA_PROCESSED_DIR: Path = Path(
    os.getenv("PNAD_DATA_PROCESSED_DIR", str(DATA_DIR / "processed"))
)

# Caminhos padrão para persistência
DEFAULT_PARQUET_PATH: Path = Path(
    os.getenv(
        "PNAD_DEFAULT_PARQUET_PATH",
        str(DATA_PROCESSED_DIR / "pnad_consolidada_2013_2025.parquet"),
    )
)
DEFAULT_STATA_PATH: Path = Path(
    os.getenv(
        "PNAD_DEFAULT_STATA_PATH",
        str(DATA_PROCESSED_DIR / "pnad_descritiva_2013_2025.dta"),
    )
)

# Parâmetros de extração e deflação
IPCA_BCB_URL: str = os.getenv(
    "IPCA_BCB_URL",
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json",
)
DATA_BASE_DEFLACAO: str = os.getenv("DATA_BASE_DEFLACAO", "01/01/2026")
DEFAULT_ANO_INICIO: int = int(os.getenv("DEFAULT_ANO_INICIO", "2013"))
DEFAULT_ANO_FIM: int = int(os.getenv("DEFAULT_ANO_FIM", "2025"))

# Nível de logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging(level: Optional[str] = None) -> None:
    """Configura o formato padrão de logs para toda a aplicação."""
    log_level_str = level or LOG_LEVEL
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def garantir_diretorios() -> None:
    """Garante que os diretórios de dados brutos e processados existam."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
