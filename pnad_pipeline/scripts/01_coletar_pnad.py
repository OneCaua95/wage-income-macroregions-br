#!/usr/bin/env python3
"""Script CLI para execução da etapa de extração e consolidação da PNAD Contínua + IPCA.

Otimizado para manter o uso de memória RAM baixo através de gravação incremental em
disco (streaming Parquet) e conversão posterior para Stata.

Uso:
    python scripts/01_coletar_pnad.py --ano-inicio 2013 --ano-fim 2025
    python scripts/01_coletar_pnad.py --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Garante que os módulos internos do pacote sejam encontrados
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config.settings import (
    DEFAULT_ANO_FIM,
    DEFAULT_ANO_INICIO,
    DEFAULT_PARQUET_PATH,
    DEFAULT_STATA_PATH,
    setup_logging,
)
from pnad_pipeline.extracao.coleta import coletar_pnad
from pnad_pipeline.extracao.io import converter_parquet_para_stata

logger = logging.getLogger("scripts.01_coletar_pnad")


def parse_args() -> argparse.Namespace:
    """Configura e processa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="CLI para extração e consolidação de microdados da PNAD Contínua com deflação IPCA (Otimizado para baixa RAM)."
    )
    parser.add_argument(
        "--ano-inicio",
        type=int,
        default=DEFAULT_ANO_INICIO,
        help=f"Ano inicial para a coleta (padrão: {DEFAULT_ANO_INICIO}).",
    )
    parser.add_argument(
        "--ano-fim",
        type=int,
        default=DEFAULT_ANO_FIM,
        help=f"Ano final para a coleta (padrão: {DEFAULT_ANO_FIM}).",
    )
    parser.add_argument(
        "--saida-parquet",
        type=str,
        default=str(DEFAULT_PARQUET_PATH),
        help=f"Caminho do arquivo Parquet de saída (padrão: {DEFAULT_PARQUET_PATH}).",
    )
    parser.add_argument(
        "--saida-stata",
        type=str,
        default=str(DEFAULT_STATA_PATH),
        help=f"Caminho do arquivo Stata (.dta) de saída (padrão: {DEFAULT_STATA_PATH}).",
    )
    parser.add_argument(
        "--pular-stata",
        action="store_true",
        help="Se informado, pula a exportação para o formato Stata (.dta), gerando apenas o Parquet.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nível de detalhamento do logging (padrão: INFO).",
    )
    return parser.parse_args()


def main() -> int:
    """Função principal do CLI de coleta."""
    args = parse_args()
    setup_logging(level=args.log_level)

    logger.info("Iniciando rotina de extração via CLI com otimização de memória RAM...")
    logger.info("Período solicitado: %d a %d", args.ano_inicio, args.ano_fim)
    logger.info("Destino Parquet: %s", args.saida_parquet)
    if not args.pular_stata:
        logger.info("Destino Stata: %s", args.saida_stata)

    # Coleta em modo streaming direto para disco (RAM O(1))
    _, erros = coletar_pnad(
        ano_inicio=args.ano_inicio,
        ano_fim=args.ano_fim,
        saida_parquet=args.saida_parquet,
        retornar_dataframe=False,
    )

    parquet_path = Path(args.saida_parquet)
    if not parquet_path.exists() or parquet_path.stat().st_size == 0:
        logger.error("Nenhum dado foi persistido no arquivo Parquet. Abortando.")
        return 1

    logger.info("✓ Arquivo consolidado Parquet gerado com sucesso: %s (%d KB)",
                parquet_path, parquet_path.stat().st_size // 1024)

    # Conversão eficiente para Stata se solicitado
    if not args.pular_stata and args.saida_stata:
        logger.info("Iniciando conversão para formato Stata (.dta)...")
        try:
            converter_parquet_para_stata(args.saida_parquet, args.saida_stata)
            logger.info("✓ Arquivo Stata (.dta) gerado com sucesso em: %s", args.saida_stata)
        except Exception as e:
            logger.error("Erro ao converter para Stata: %s", e, exc_info=True)

    logger.info("Processo finalizado.")
    if erros:
        logger.warning("%d períodos registraram falhas durante a coleta:", len(erros))
        for err in erros:
            logger.warning("  • %s", err)

    return 0


if __name__ == "__main__":
    sys.exit(main())
