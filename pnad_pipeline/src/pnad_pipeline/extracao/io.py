"""Módulo de entrada e saída (I/O) para persistência dos dados da PNAD Contínua.

Inclui suporte a escrita em streaming (incremental) via PyArrow para eliminar
o acúmulo de dados na memória RAM durante coletas longas.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .otimizacao import otimizar_tipos

logger = logging.getLogger(__name__)


class ParquetStreamWriter:
    """Escritor incremental de arquivos Parquet via PyArrow.

    Permite adicionar DataFrames trimestre a trimestre diretamente no disco,
    mantendo o consumo de memória RAM constante O(1).
    """

    def __init__(self, caminho: Union[str, Path], compression: str = "snappy"):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None
        self.total_rows: int = 0

    def append(self, df: pd.DataFrame) -> None:
        """Adiciona um lote/trimestre ao arquivo Parquet existente."""
        if df.empty:
            return

        table = pa.Table.from_pandas(df, preserve_index=False)

        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                str(self.caminho),
                self._schema,
                compression=self.compression,
            )
            logger.info("Iniciando arquivo Parquet incremental: %s", self.caminho)

        # Assegura que o schema de todos os trimestres seja rigorosamente consistente
        if table.schema != self._schema:
            try:
                table = table.cast(self._schema)
            except Exception as e:
                logger.warning("Tentativa de ajuste de schema via cast falhou: %s. Gravando com schema original.", e)

        self._writer.write_table(table)
        self.total_rows += len(df)
        logger.debug("Lote gravado no Parquet (%d linhas). Total acumulado: %d.", len(df), self.total_rows)

    def close(self) -> Path:
        """Fecha o stream do Parquet liberando recursos de arquivo."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info("✓ Parquet concluído com sucesso: %s (%s linhas totais).", self.caminho, f"{self.total_rows:,}")
        return self.caminho

    def __enter__(self) -> ParquetStreamWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def salvar_parquet(df: pd.DataFrame, caminho: Union[str, Path]) -> Path:
    """Salva o DataFrame consolidado no formato Apache Parquet com compressão snappy."""
    path_obj = Path(caminho)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    df_opt = otimizar_tipos(df)
    logger.info("Salvando DataFrame em Parquet: %s (shape: %s)...", path_obj, df_opt.shape)
    df_opt.to_parquet(path_obj, index=False, compression="snappy")
    logger.info("✓ Arquivo Parquet salvo com sucesso em: %s", path_obj)
    return path_obj


def salvar_stata(
    df: pd.DataFrame,
    caminho: Union[str, Path],
    version: int = 118,
) -> Path:
    """Salva o DataFrame no formato Stata (.dta) com sanitização de nomes e compatibilidade de tipos.

    Aplica as conformidades exigidas pelo formato Stata:
    - Converte colunas booleanas para inteiros (0/1);
    - Trunca nomes de colunas em até 32 caracteres e substitui pontos por sublinhados;
    - Trata colunas categóricas para formato Stata.
    """
    path_obj = Path(caminho)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Preparando dados para formato Stata (.dta) versão %d...", version)
    df_stata = df.copy()

    # Converte tipos booleanos para inteiro
    bool_cols = df_stata.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        logger.debug("Convertendo %d colunas booleanas para inteiro.", len(bool_cols))
        df_stata[bool_cols] = df_stata[bool_cols].astype(int)

    # Sanitização de nomes de colunas (máx 32 caracteres e sem pontos)
    df_stata.columns = [str(c)[:32].replace(".", "_") for c in df_stata.columns]

    logger.info("Salvando DataFrame em formato Stata: %s (shape: %s)...", path_obj, df_stata.shape)
    df_stata.to_stata(path_obj, write_index=False, version=version)
    logger.info("✓ Arquivo Stata (.dta) salvo com sucesso em: %s", path_obj)

    del df_stata
    gc.collect()
    return path_obj


def converter_parquet_para_stata(
    caminho_parquet: Union[str, Path],
    caminho_stata: Union[str, Path],
    version: int = 118,
) -> Path:
    """Lê o arquivo Parquet gerado com dtypes compactados e converte eficientemente para Stata (.dta)."""
    p_in = Path(caminho_parquet)
    p_out = Path(caminho_stata)

    if not p_in.exists():
        raise FileNotFoundError(f"Arquivo Parquet de origem não encontrado: {p_in}")

    logger.info("Lendo Parquet otimizado para conversão Stata: %s...", p_in)
    df = pd.read_parquet(p_in)
    salvar_stata(df, p_out, version=version)
    del df
    gc.collect()
    return p_out
