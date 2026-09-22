"""Subpacote de extração de dados da PNAD Contínua e IPCA."""

from .constants import (
    COLUMNS,
    COLUMNS_RENAME,
    DATA_BASE_DEFLACAO_PADRAO,
    IPCA_SGS_URL_PADRAO,
    UF_PARA_REGIAO,
)
from .deflacao import aplicar_deflacao, calcular_data_referencia_trimestre
from .filtros import (
    aplicar_filtros,
    filtrar_idade,
    filtrar_renda_positiva,
    remover_salarios_negativos,
)
from .io import (
    ParquetStreamWriter,
    converter_parquet_para_stata,
    salvar_parquet,
    salvar_stata,
)
from .ipca import calcular_fator_acumulado, get_ipca_bc
from .otimizacao import calcular_memoria_mb, otimizar_tipos
from .pnad_extractor import PNADExtractor, download_pnad
from .regiao import mapear_regiao
from .coleta import processar_periodo, coletar_pnad

__all__ = [
    "COLUMNS",
    "COLUMNS_RENAME",
    "DATA_BASE_DEFLACAO_PADRAO",
    "IPCA_SGS_URL_PADRAO",
    "UF_PARA_REGIAO",
    "aplicar_deflacao",
    "calcular_data_referencia_trimestre",
    "aplicar_filtros",
    "filtrar_idade",
    "filtrar_renda_positiva",
    "remover_salarios_negativos",
    "ParquetStreamWriter",
    "converter_parquet_para_stata",
    "salvar_parquet",
    "salvar_stata",
    "calcular_fator_acumulado",
    "get_ipca_bc",
    "calcular_memoria_mb",
    "otimizar_tipos",
    "PNADExtractor",
    "download_pnad",
    "mapear_regiao",
    "processar_periodo",
    "coletar_pnad",
]
