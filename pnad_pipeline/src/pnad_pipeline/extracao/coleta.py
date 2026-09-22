"""Módulo orquestrador da coleta e pré-processamento inicial da PNAD Contínua + IPCA.

Otimizado para manter o consumo de memória RAM baixo e estável mesmo ao coletar
séries históricas de longa duração (ex.: 2013 a 2025 com 52 trimestres).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from .constants import (
    COLUMNS,
    COLUMNS_RENAME,
    DATA_BASE_DEFLACAO_PADRAO,
)
from .deflacao import aplicar_deflacao, calcular_data_referencia_trimestre
from .filtros import aplicar_filtros
from .io import ParquetStreamWriter
from .ipca import calcular_fator_acumulado, get_ipca_bc
from .otimizacao import calcular_memoria_mb, otimizar_tipos
from .pnad_extractor import PNADExtractor
from .regiao import mapear_regiao

logger = logging.getLogger(__name__)


def processar_periodo(
    ano: int,
    trimestre: int,
    df_ipca: pd.DataFrame,
    extractor: Optional[PNADExtractor] = None,
    data_base_deflacao: Optional[str] = None,
) -> tuple[pd.DataFrame, str, float]:
    """Orquestra a extração, renomeação, filtragem, deflação, região e downcasting de tipos para um período.

    Executa em memória controlada, liberando os DataFrames brutos intermediários.

    Parameters
    ----------
    ano : int
        Ano da pesquisa (ex.: 2013).
    trimestre : int
        Trimestre da pesquisa (1 a 4).
    df_ipca : pd.DataFrame
        Série histórica do IPCA obtida do Banco Central.
    extractor : PNADExtractor, optional
        Instância do extrator de dados. Se None, instancia PNADExtractor().
    data_base_deflacao : str, optional
        Data-base para cálculo do deflator (ex.: '01/01/2026').

    Returns
    -------
    tuple[pd.DataFrame, str, float]
        Tupla contendo (DataFrame processado e compactado, string do período 'AAAA.T', fator deflator).
    """
    periodo_str = f"{ano}.{trimestre}"
    logger.info("→ Processando período %s (%dT%d)...", periodo_str, ano, trimestre)

    ext = extractor or PNADExtractor()
    dt_base = data_base_deflacao or DATA_BASE_DEFLACAO_PADRAO

    # 1) Download e renomeação
    df_raw = ext.download_data(year=ano, quarter=trimestre, cols=COLUMNS)
    df = df_raw.rename(columns=COLUMNS_RENAME)
    del df_raw

    # 2) Filtros de qualidade (passada única na memória)
    df_filtrado = aplicar_filtros(df)
    del df

    # 3) Deflação com IPCA
    data_inicio = calcular_data_referencia_trimestre(ano, trimestre)
    fator = calcular_fator_acumulado(df_ipca, data_inicio=data_inicio, data_fim=dt_base)
    df_deflacionado = aplicar_deflacao(df_filtrado, fator=fator)
    del df_filtrado

    # 4) Mapeamento de Região
    df_regiao = mapear_regiao(df_deflacionado)
    del df_deflacionado

    # 5) Otimização agressiva de tipos de dados (downcasting de RAM)
    df_final = otimizar_tipos(df_regiao)
    del df_regiao

    mem_mb = calcular_memoria_mb(df_final)
    logger.info(
        "Período %s finalizado | Fator IPCA: %.6f | Obs. finais: %s | RAM: %.2f MB",
        periodo_str,
        fator,
        f"{len(df_final):,}",
        mem_mb,
    )
    return df_final, periodo_str, fator


def coletar_pnad(
    ano_inicio: int = 2013,
    ano_fim: int = 2025,
    df_ipca: Optional[pd.DataFrame] = None,
    extractor: Optional[PNADExtractor] = None,
    data_base_deflacao: Optional[str] = None,
    saida_parquet: Optional[Union[str, Path]] = None,
    retornar_dataframe: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Executa o loop completo de coleta e extração da PNAD Contínua para o intervalo de anos especificado.

    Caso `saida_parquet` seja especificado, utiliza escrita incremental via streaming (ParquetStreamWriter)
    gravando diretamente no disco trimestre a trimestre e invocando coleta de lixo periódica,
    garantindo que o consumo de memória RAM permaneça em O(1) e não cresça ao longo dos 52 trimestres.

    TODO: O cálculo de estatísticas descritivas (salário-hora médio por sexo, setor,
    formalidade, macrorregião, etc.) foi transferido para o módulo 'analise/'.

    Parameters
    ----------
    ano_inicio : int, default 2013
        Ano inicial da coleta.
    ano_fim : int, default 2025
        Ano final da coleta.
    df_ipca : pd.DataFrame, optional
        Série histórica do IPCA. Se None, busca automaticamente via get_ipca_bc().
    extractor : PNADExtractor, optional
        Instância do extrator para download.
    data_base_deflacao : str, optional
        Data-base para deflação pelo IPCA.
    saida_parquet : str ou Path, optional
        Caminho do arquivo Parquet para gravação em streaming direto no disco.
    retornar_dataframe : bool, default True
        Se True, retorna o DataFrame consolidado (se saida_parquet não for informado,
        ou lendo o Parquet final). Se False e saida_parquet for informado, evita manter
        ou ler a base consolidada na memória RAM, retornando um DataFrame vazio.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Tupla com o DataFrame consolidado (ou vazio se retorno desativado) e a lista
        de erros ocorridos.
    """
    total_trimestres = (ano_fim - ano_inicio + 1) * 4
    logger.info("=" * 60)
    logger.info(
        "Iniciando pipeline de extração PNAD Contínua: %d a %d (%d trimestres)",
        ano_inicio,
        ano_fim,
        total_trimestres,
    )
    if saida_parquet:
        logger.info("Modo Streaming ativado: escrita direta no disco em: %s", saida_parquet)
    logger.info("=" * 60)

    if df_ipca is None:
        df_ipca = get_ipca_bc()

    fragmentos: list[pd.DataFrame] = []
    periodos_com_erro: list[str] = []
    trimestres_sucesso = 0

    stream_writer: Optional[ParquetStreamWriter] = (
        ParquetStreamWriter(saida_parquet) if saida_parquet else None
    )

    try:
        for ano in range(ano_inicio, ano_fim + 1):
            for trimestre in range(1, 5):
                periodo_label = f"{ano}T{trimestre}"
                try:
                    data, _, _ = processar_periodo(
                        ano=ano,
                        trimestre=trimestre,
                        df_ipca=df_ipca,
                        extractor=extractor,
                        data_base_deflacao=data_base_deflacao,
                    )

                    if stream_writer is not None:
                        # Grava o trimestre imediatamente no disco
                        stream_writer.append(data)
                        del data
                    else:
                        fragmentos.append(data)

                    trimestres_sucesso += 1

                except Exception as e:
                    msg = f"{periodo_label}: {e}"
                    logger.error("Falha ao processar período %s: %s", periodo_label, e, exc_info=True)
                    periodos_com_erro.append(msg)
                finally:
                    # Coleta de lixo forçada a cada trimestre para liberar páginas de RAM imediatamente
                    gc.collect()

    finally:
        if stream_writer is not None:
            stream_writer.close()

    logger.info("=" * 60)
    logger.info(
        "Coleta finalizada. Sucesso em %d/%d trimestres. Períodos com erro: %d",
        trimestres_sucesso,
        total_trimestres,
        len(periodos_com_erro),
    )
    for p in periodos_com_erro:
        logger.warning("  • Falha registrada: %s", p)

    if stream_writer is not None:
        if retornar_dataframe and trimestres_sucesso > 0:
            logger.info("Lendo dataset consolidado otimizado a partir do disco: %s...", saida_parquet)
            df_consolidado = pd.read_parquet(saida_parquet)
        else:
            df_consolidado = pd.DataFrame()
    else:
        if fragmentos:
            logger.info("Consolidando fragmentos em memória...")
            df_consolidado = pd.concat(fragmentos, ignore_index=True)
            del fragmentos
            gc.collect()
            logger.info("Shape consolidado final: %s", df_consolidado.shape)
        else:
            logger.warning("Nenhum dado processado com sucesso no intervalo selecionado.")
            df_consolidado = pd.DataFrame()

    return df_consolidado, periodos_com_erro
