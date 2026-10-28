#!/usr/bin/env python3
"""Trata a base consolidada da PNAD para as análises de indivíduos de 25 a 65 anos.

Uso:
    python scripts/02_tratar_pnad.py
    python scripts/02_tratar_pnad.py --entrada data/processed/pnad_consolidada_2013_2025.parquet
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config.settings import setup_logging
from pnad_pipeline.analise import (
    calcular_retorno_educacao,
    calcular_series_rendimento_desigualdade,
    calcular_series_taxa_formalidade,
    calcular_series_trimestrais_salario_hora,
    preparar_variaveis_analise,
)
from pnad_pipeline.analise.amostragem import (
    COLUNAS_DESENHO, diagnostico_representatividade, exigir_colunas_desenho,
)
from pnad_pipeline.extracao.io import ParquetStreamWriter

logger = logging.getLogger("scripts.02_tratar_pnad")

COLUNAS_ANALISE = [
    "Ano",
    "Trimestre",
    "UF",
    *COLUNAS_DESENHO,
    "Sexo",
    "Idade",
    "Cor",
    "Anos_de_Estudo",
    "Categoria_emprego",
    "Condicao_Ocupacao",
    "Contribuicao_previdencia",
    "Grupo_atv_princ_empreedimento",
    "Rendimento_hab_Trab_princ",
    "Horas_hab_trabalhadas",
    "Regiao",
]


def parse_args() -> argparse.Namespace:
    """Configura e processa os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Prepara a base analítica da PNAD para indivíduos de 25 a 65 anos."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet",
        help="Parquet consolidado de entrada.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "pnad_analise_25_65.parquet",
        help="Parquet tratado de saída.",
    )
    parser.add_argument(
        "--tabelas-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "tabelas_descritivas",
        help="Diretório das tabelas descritivas de saída.",
    )
    parser.add_argument("--min-idade", type=int, default=25, help="Idade mínima inclusiva.")
    parser.add_argument("--max-idade", type=int, default=65, help="Idade máxima inclusiva.")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Linhas por lote de tratamento.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nível de detalhamento do logging.",
    )
    return parser.parse_args()


def tratar_base(args: argparse.Namespace) -> Path:
    """Lê, transforma e salva a base analítica."""
    if not args.entrada.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.entrada}")
    if args.min_idade > args.max_idade:
        raise ValueError("--min-idade deve ser menor ou igual a --max-idade.")

    if args.entrada.resolve() == args.saida.resolve():
        raise ValueError("Entrada e saída devem ser arquivos diferentes.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser positivo.")
    parquet = pq.ParquetFile(args.entrada)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    logger.info("Lendo base consolidada em lotes: %s", args.entrada)
    with ParquetStreamWriter(args.saida, metadata={"pnad_amostra_completa": "false"}) as writer:
        for batch in parquet.iter_batches(columns=COLUNAS_ANALISE, batch_size=args.batch_size):
            df_tratado = preparar_variaveis_analise(
                batch.to_pandas(), min_idade=args.min_idade, max_idade=args.max_idade,
            )
            writer.append(df_tratado)
            del df_tratado
            gc.collect()
        if writer.total_rows == 0:
            raise ValueError("Nenhuma observação na faixa etária solicitada.")
        logger.info("Base analítica salva: %s (%s observações)", args.saida, f"{writer.total_rows:,}")
    return args.saida


def gerar_tabelas(caminho_base: Path, tabelas_dir: Path) -> None:
    """Gera as tabelas históricas usadas nas análises do notebook."""
    tabelas_dir.mkdir(parents=True, exist_ok=True)
    tarefas = {
        "series_salario_hora_2013_2025.csv": (
            calcular_series_trimestrais_salario_hora,
            ["Periodo", "Sexo", "Salario_Hora", "Setor", "Formalidade", "Regiao"],
        ),
        "series_desigualdade_2013_2025.csv": (
            calcular_series_rendimento_desigualdade,
            ["Periodo", "Rendimento_hab_Trab_princ"],
        ),
        "series_formalidade_2013_2025.csv": (
            calcular_series_taxa_formalidade,
            ["Periodo", "Formalidade", "Regiao"],
        ),
        "series_educacao_2013_2025.csv": (
            calcular_retorno_educacao,
            ["Periodo", "Escolaridade_Nivel", "Salario_Hora"],
        ),
    }
    for nome, (funcao, colunas) in tarefas.items():
        logger.info("Calculando %s a partir de %s", nome, caminho_base)
        df_tabela = pd.read_parquet(caminho_base, columns=colunas + ["V1028"])
        tabela = funcao(df_tabela)
        destino = tabelas_dir / nome
        tabela.to_csv(destino, index=False)
        logger.info("Tabela salva: %s (%s linhas)", destino, f"{len(tabela):,}")
        del df_tabela, tabela
        gc.collect()
    desenho = pd.read_parquet(caminho_base, columns=["Ano", "Trimestre", "UF", *COLUNAS_DESENHO])
    diagnostico_representatividade(desenho).to_csv(
        tabelas_dir / "representatividade_trimestral.csv", index=False,
    )


def main() -> int:
    """Executa o tratamento completo da base analítica."""
    args = parse_args()
    setup_logging(level=args.log_level)

    try:
        caminho_base = tratar_base(args)
        gerar_tabelas(caminho_base, args.tabelas_dir)
    except Exception as exc:
        logger.error("Falha no tratamento da PNAD: %s", exc, exc_info=True)
        return 1

    logger.info("Tratamento concluído para a faixa etária %d–%d.", args.min_idade, args.max_idade)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
