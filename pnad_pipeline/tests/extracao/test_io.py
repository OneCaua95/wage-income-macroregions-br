"""Testes unitários para persistência de dados em Parquet, Stata e streaming."""

from pathlib import Path
import pandas as pd
import pytest

from pnad_pipeline.extracao.io import (
    ParquetStreamWriter,
    converter_parquet_para_stata,
    salvar_parquet,
    salvar_stata,
)


@pytest.fixture
def df_exemplo() -> pd.DataFrame:
    """Retorna DataFrame de exemplo com diferentes tipos de dados incluindo booleanos."""
    return pd.DataFrame({
        "coluna_normal": [1, 2, 3],
        "coluna.com.ponto": ["a", "b", "c"],
        "flag_booleana": [True, False, True],
        "coluna_com_nome_muito_longo_que_ultrapassa_trinta_e_dois_caracteres": [10, 20, 30],
    })


def test_salvar_parquet(tmp_path: Path, df_exemplo: pd.DataFrame):
    """Valida salvamento e leitura íntegra em formato Parquet."""
    caminho = tmp_path / "subdir" / "teste.parquet"
    retorno = salvar_parquet(df_exemplo, caminho)

    assert retorno.exists()
    df_lido = pd.read_parquet(caminho)
    assert len(df_lido) == len(df_exemplo)
    assert list(df_lido.columns) == list(df_exemplo.columns)


def test_salvar_stata(tmp_path: Path, df_exemplo: pd.DataFrame):
    """Valida salvamento em formato Stata (.dta) com sanitização de nomes e booleanos."""
    caminho = tmp_path / "subdir" / "teste.dta"
    retorno = salvar_stata(df_exemplo, caminho)

    assert retorno.exists()
    df_lido = pd.read_stata(caminho)
    assert len(df_lido) == len(df_exemplo)

    # Nomes sanitizados: sem pontos e com no máximo 32 caracteres
    for col in df_lido.columns:
        assert "." not in col
        assert len(col) <= 32

    # Booleano convertido para inteiro (0/1)
    assert df_lido["flag_booleana"].dtype in ["int8", "int16", "int32", "int64"]


def test_parquet_stream_writer(tmp_path: Path):
    """Valida a escrita incremental lote a lote no Parquet."""
    caminho = tmp_path / "streaming.parquet"

    df_lote1 = pd.DataFrame({"id": [1, 2], "valor": [10.5, 20.0]})
    df_lote2 = pd.DataFrame({"id": [3, 4], "valor": [30.0, 40.5]})

    with ParquetStreamWriter(caminho) as writer:
        writer.append(df_lote1)
        writer.append(df_lote2)

    assert caminho.exists()
    df_lido = pd.read_parquet(caminho)
    assert len(df_lido) == 4
    assert df_lido["id"].tolist() == [1, 2, 3, 4]


def test_converter_parquet_para_stata(tmp_path: Path, df_exemplo: pd.DataFrame):
    """Valida a conversão desacoplada de Parquet para Stata."""
    caminho_parquet = tmp_path / "origem.parquet"
    caminho_stata = tmp_path / "destino.dta"

    salvar_parquet(df_exemplo, caminho_parquet)
    converter_parquet_para_stata(caminho_parquet, caminho_stata)

    assert caminho_stata.exists()
    df_stata = pd.read_stata(caminho_stata)
    assert len(df_stata) == len(df_exemplo)
