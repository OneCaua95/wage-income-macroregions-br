"""Filtros de qualidade de dados para a PNAD Contínua.

Aplica regras de seleção amostral: renda habitual > 0, faixa etária 25 a 65 anos,
e remoção de inconsistências como salários negativos, registrando a contagem
de observações descartadas em cada etapa.
Otimizado para baixa pegada de memória com filtragem em passada única.
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def filtrar_renda_positiva(
    df: pd.DataFrame,
    col_renda: str = "Rendimento_hab_Trab_princ",
) -> pd.DataFrame:
    """Filtra observações garantindo que o rendimento seja estritamente positivo (> 0).

    Descarta valores nulos, zerados ou negativos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    col_renda : str, default "Rendimento_hab_Trab_princ"
        Nome da coluna de rendimento.

    Returns
    -------
    pd.DataFrame
        Cópia filtrada contendo apenas observações com renda > 0.
    """
    obs_antes = len(df)
    mascara = df[col_renda] > 0
    df_filtrado = df.loc[mascara].copy()
    obs_depois = len(df_filtrado)
    removidas = obs_antes - obs_depois

    logger.info(
        "Filtro [Renda > 0]: %s → %s obs. (removidas: %s)",
        f"{obs_antes:,}",
        f"{obs_depois:,}",
        f"{removidas:,}",
    )
    return df_filtrado


def filtrar_idade(
    df: pd.DataFrame,
    min_idade: int = 25,
    max_idade: int = 65,
    col_idade: str = "Idade",
) -> pd.DataFrame:
    """Filtra observações dentro da faixa etária [min_idade, max_idade].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    min_idade : int, default 25
        Idade mínima inclusive.
    max_idade : int, default 65
        Idade máxima inclusive.
    col_idade : str, default "Idade"
        Nome da coluna de idade.

    Returns
    -------
    pd.DataFrame
        Cópia filtrada contendo apenas indivíduos na faixa etária especificada.
    """
    obs_antes = len(df)
    mascara = (df[col_idade] >= min_idade) & (df[col_idade] <= max_idade)
    df_filtrado = df.loc[mascara].copy()
    obs_depois = len(df_filtrado)
    removidas = obs_antes - obs_depois

    logger.info(
        "Filtro [Idade %d-%d]: %s → %s obs. (removidas: %s)",
        min_idade,
        max_idade,
        f"{obs_antes:,}",
        f"{obs_depois:,}",
        f"{removidas:,}",
    )
    return df_filtrado


def remover_salarios_negativos(
    df: pd.DataFrame,
    col_renda: str = "Rendimento_hab_Trab_princ",
) -> pd.DataFrame:
    """Verifica e descarta qualquer observação com rendimento negativo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    col_renda : str, default "Rendimento_hab_Trab_princ"
        Nome da coluna de rendimento.

    Returns
    -------
    pd.DataFrame
        Cópia sem valores negativos de rendimento.
    """
    negativos = int((df[col_renda] < 0).sum())
    if negativos > 0:
        logger.warning(
            "⚠️ %s observações com salário NEGATIVO encontradas — removendo...",
            f"{negativos:,}",
        )
        return df.loc[df[col_renda] >= 0].copy()

    logger.info("✓ Nenhum salário negativo encontrado.")
    return df.copy()


def aplicar_filtros(
    df: pd.DataFrame,
    col_renda: str = "Rendimento_hab_Trab_princ",
    col_idade: str = "Idade",
    min_idade: int = 25,
    max_idade: int = 65,
) -> pd.DataFrame:
    """Orquestra a aplicação de todos os filtros em passada única vetorizada.

    Elimina a criação de cópias intermediárias pesadas em memória RAM.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto renomeado da PNAD Contínua.
    col_renda : str
        Nome da coluna de rendimento habitual.
    col_idade : str
        Nome da coluna de idade.
    min_idade : int
        Idade mínima inclusive.
    max_idade : int
        Idade máxima inclusive.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado e otimizado.
    """
    total_inicial = len(df)
    logger.info("Aplicando filtros de qualidade amostral (Total inicial: %s obs)...", f"{total_inicial:,}")

    # Máscaras booleanas vetorizadas em memória direta
    mask_renda_pos = df[col_renda] > 0
    removidas_renda = total_inicial - int(mask_renda_pos.sum())
    logger.info(
        "Filtro [Renda > 0]: %s → %s obs. (removidas: %s)",
        f"{total_inicial:,}",
        f"{int(mask_renda_pos.sum()):,}",
        f"{removidas_renda:,}",
    )

    mask_idade = (df[col_idade] >= min_idade) & (df[col_idade] <= max_idade)
    obs_idade = int(mask_idade.sum())
    removidas_idade = total_inicial - obs_idade
    logger.info(
        "Filtro [Idade %d-%d]: %s → %s obs. (removidas: %s)",
        min_idade,
        max_idade,
        f"{total_inicial:,}",
        f"{obs_idade:,}",
        f"{removidas_idade:,}",
    )

    # Verificação de salários negativos para auditoria
    negativos = int((df[col_renda] < 0).sum())
    if negativos > 0:
        logger.warning("⚠️ %s observações com salário NEGATIVO encontradas — removendo...", f"{negativos:,}")
    else:
        logger.info("✓ Nenhum salário negativo encontrado.")

    # Aplica combinação das máscaras em uma única fatia (única cópia necessária)
    mask_final = mask_renda_pos & mask_idade
    df_filtrado = df.loc[mask_final].copy()

    total_final = len(df_filtrado)
    logger.info(
        "Filtros concluídos: %s → %s obs. (Total descartado: %s)",
        f"{total_inicial:,}",
        f"{total_final:,}",
        f"{total_inicial - total_final:,}",
    )
    return df_filtrado
