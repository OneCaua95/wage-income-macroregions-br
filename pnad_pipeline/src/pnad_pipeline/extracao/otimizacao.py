"""Módulo para otimização de tipos de dados (downcasting) visando redução do consumo de memória RAM."""

from __future__ import annotations

import logging
from typing import Final
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapeamento explícito de tipos para garantir compactação máxima e compatibilidade
DTYPES_MAPEAMENTO: Final[dict[str, str]] = {
    "Ano": "int16",
    "Trimestre": "int8",
    "UF": "int8",
    "Sexo": "int8",
    "Ano_Nascimento": "Int16",
    "Cor": "Int8",
    "Idade": "int8",
    "Escolaridade": "Int8",
    "Anos_de_Estudo": "Int8",
    "Renda_Faixa": "Int8",
    "Renda_PEA_Reais": "float32",
    "Renda_Efetivo_14_Reais": "float32",
    "F_Trabalho": "Int8",
    "Condicao_Ocupacao": "Int8",
    "Forca_Trabalho_potencial": "Int8",
    "Subocupacao_por_insu_horas": "Int8",
    "Pessoa_desalentadas": "Int8",
    "Categoria_emprego": "Int8",
    "Grupo_atv_princ_empreedimento": "Int8",
    "Grupo_ocupacional_no_emprego": "Int8",
    "Contribuicao_previdencia": "Int8",
    "Rendimento_hab_Trab_princ": "float32",
    "Rendimento_hab_Tds_Trabs": "float32",
    "Horas_hab_trabalhadas": "float32",
    "Regiao": "category",
}


def calcular_memoria_mb(df: pd.DataFrame) -> float:
    """Calcula o uso real de memória RAM do DataFrame em Megabytes."""
    return float(df.memory_usage(deep=True).sum() / (1024 * 1024))


def otimizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica downcasting e conversão para tipos compactos e categóricos.

    Reduz em média 70% a 85% do tamanho em RAM por linha.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada com tipos padrão.

    Returns
    -------
    pd.DataFrame
        Cópia do DataFrame com dtypes otimizados.
    """
    mem_antes = calcular_memoria_mb(df)
    df_opt = df.copy()

    for col in df_opt.columns:
        if col in DTYPES_MAPEAMENTO:
            target_type = DTYPES_MAPEAMENTO[col]
            try:
                if target_type.startswith("Int") or target_type.startswith("int"):
                    # Converte para numérico e depois aplica o tipo inteiro nullable ou padrão
                    num = pd.to_numeric(df_opt[col], errors="coerce")
                    df_opt[col] = num.astype(target_type)
                elif target_type == "float32":
                    df_opt[col] = pd.to_numeric(df_opt[col], errors="coerce").astype("float32")
                elif target_type == "category":
                    df_opt[col] = df_opt[col].astype("category")
            except Exception as e:
                logger.debug("Não foi possível converter coluna '%s' para '%s': %s", col, target_type, e)
        else:
            # Para colunas não mapeadas, faz downcast genérico
            if pd.api.types.is_float_dtype(df_opt[col]):
                df_opt[col] = df_opt[col].astype("float32")
            elif pd.api.types.is_integer_dtype(df_opt[col]):
                df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")

    mem_depois = calcular_memoria_mb(df_opt)
    reducao = (1 - (mem_depois / mem_antes)) * 100 if mem_antes > 0 else 0
    logger.debug(
        "Downcasting de tipos concluído: %.2f MB → %.2f MB (redução de %.1f%%).",
        mem_antes,
        mem_depois,
        reducao,
    )
    return df_opt
