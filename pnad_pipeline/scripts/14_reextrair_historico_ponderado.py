#!/usr/bin/env python3
"""Reextrai a série PNAD 2013–2025 com pesos e desenho amostral.

Cada ano é gravado em uma parte independente para permitir retomada após falha.
Ao final, as partes são consolidadas atomicamente em um único Parquet.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import DEFAULT_ANO_FIM, DEFAULT_ANO_INICIO, setup_logging
from pnad_pipeline.analise.amostragem import (
    COLUNAS_DESENHO,
    exigir_colunas_desenho,
    validar_desenho,
)
from pnad_pipeline.extracao.coleta import coletar_pnad
from pnad_pipeline.extracao.ipca import get_ipca_bc

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "recoleta_ponderada"
PERIOD_COLUMNS = ["Ano", "Trimestre"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ano-inicio", type=int, default=DEFAULT_ANO_INICIO)
    parser.add_argument("--ano-fim", type=int, default=DEFAULT_ANO_FIM)
    parser.add_argument("--saida-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4, help="Anos coletados em paralelo (1–4).")
    return parser.parse_args()


def validar_arquivo_ano(path: Path, ano: int) -> int:
    """Exige os quatro trimestres e valida os campos do desenho em todos os lotes."""
    parquet = pq.ParquetFile(path)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    exigir = {*PERIOD_COLUMNS, "UF", *COLUNAS_DESENHO}
    faltantes = sorted(exigir.difference(parquet.schema_arrow.names))
    if faltantes:
        raise ValueError(f"{path}: colunas ausentes: {faltantes}")

    periodos: set[tuple[int, int]] = set()
    for batch in parquet.iter_batches(columns=PERIOD_COLUMNS, batch_size=250_000):
        df = batch.to_pandas()
        periodos.update(
            (int(a), int(t))
            for a, t in df.drop_duplicates().itertuples(index=False, name=None)
        )
    esperados = {(ano, trimestre) for trimestre in range(1, 5)}
    if periodos != esperados:
        raise ValueError(
            f"{path}: cobertura incorreta; esperado {sorted(esperados)}, observado {sorted(periodos)}"
        )

    for batch in parquet.iter_batches(
        columns=["UF", *COLUNAS_DESENHO], batch_size=250_000
    ):
        validar_desenho(batch.to_pandas())
    return parquet.metadata.num_rows


def configurar_cache_worker(cache_root: str) -> None:
    """Dá a cada processo um cache próprio: o pnadium remove seu cache ao terminar."""
    import appdirs

    local_cache = Path(cache_root) / f"worker-{os.getpid()}"
    local_cache.mkdir(parents=True, exist_ok=True)

    def cache_por_processo(appname=None, appauthor=None, version=None, opinion=True):
        return str(local_cache)

    appdirs.user_cache_dir = cache_por_processo


def coletar_ano(ano: int, partes_dir: Path, logs_dir: Path, df_ipca) -> dict:
    """Coleta um ano inteiro em processo separado e publica a parte após validá-la."""
    destino = partes_dir / f"pnad_{ano}.parquet"
    parcial = partes_dir / f"pnad_{ano}.partial.parquet"
    log_path = logs_dir / f"coleta_{ano}.log"
    if parcial.exists():
        parcial.unlink()

    with log_path.open("w", encoding="utf-8", errors="replace") as log, redirect_stdout(log), redirect_stderr(log):
        setup_logging(level="INFO")
        logging.info("Iniciando coleta dos quatro trimestres de %s", ano)
        _, erros = coletar_pnad(
            ano_inicio=ano,
            ano_fim=ano,
            df_ipca=df_ipca,
            saida_parquet=parcial,
            retornar_dataframe=False,
        )
        if erros:
            raise RuntimeError("Falhas de coleta: " + " | ".join(erros))
        linhas = validar_arquivo_ano(parcial, ano)
        os.replace(parcial, destino)
        logging.info("Ano %s validado e salvo: %s linhas", ano, f"{linhas:,}")
    return {"ano": ano, "linhas": linhas, "arquivo": str(destino)}


def adotar_parte_existente(ano: int, partes_dir: Path, output_dir: Path) -> bool:
    """Reaproveita parte válida de uma execução anterior ou do piloto de 2013."""
    destino = partes_dir / f"pnad_{ano}.parquet"
    if destino.exists():
        try:
            validar_arquivo_ano(destino, ano)
            return True
        except Exception as exc:
            logging.warning("Parte %s incompleta; será reprocessada: %s", ano, exc)

    piloto = output_dir / f"pnad_consolidada_{ano}.parquet"
    if not destino.exists() and piloto.exists():
        try:
            validar_arquivo_ano(piloto, ano)
            os.replace(piloto, destino)
            logging.info("Parte válida do piloto de %s reaproveitada.", ano)
            return True
        except Exception as exc:
            logging.warning("Piloto %s incompleto; será reprocessado: %s", ano, exc)
    return False


def consolidar(partes: list[Path], destino: Path, ano_inicio: int, ano_fim: int) -> int:
    """Consolida partes verificadas e só troca o arquivo final após escrita completa."""
    parcial = destino.with_name(destino.stem + ".partial" + destino.suffix)
    if parcial.exists():
        parcial.unlink()

    schemas = [pq.ParquetFile(path).schema_arrow.remove_metadata() for path in partes]
    schema = pa.unify_schemas(schemas)
    metadata = {
        b"pnad_amostra_completa": b"true",
        b"ponderacao": b"V1028",
        b"variaveis_desenho": b"V1028,Estrato,UPA",
        b"periodo": f"{ano_inicio}-{ano_fim}".encode(),
        b"deflacao_data_base": b"01/01/2026",
        b"fonte": b"Microdados trimestrais PNAD Continua / IBGE",
    }
    output_schema = schema.with_metadata(metadata)

    with pq.ParquetWriter(parcial, output_schema, compression="snappy") as writer:
        for path in partes:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch]).cast(schema)
                writer.write_table(table, row_group_size=100_000)

    linhas = pq.ParquetFile(parcial).metadata.num_rows
    os.replace(parcial, destino)
    return linhas


def registrar_arquivos_fonte(anos: list[int]) -> dict[str, str | None]:
    """Registra os nomes das versões trimestrais selecionadas pelo índice do IBGE."""
    from pnadium.trimestral import map_files

    arquivos, _, _ = map_files()
    fontes = {}
    for ano in anos:
        for trimestre in range(1, 5):
            chave = f"{ano}T{trimestre}"
            valor = arquivos.loc[trimestre - 1, str(ano)] if str(ano) in arquivos.columns else None
            fontes[chave] = valor if isinstance(valor, str) else None
    return fontes


def main() -> int:
    args = parse_args()
    setup_logging()
    if args.ano_inicio > args.ano_fim:
        raise ValueError("--ano-inicio deve ser menor ou igual a --ano-fim.")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers deve estar entre 1 e 4.")

    output_dir = args.saida_dir.resolve()
    partes_dir = output_dir / "partes_anuais"
    logs_dir = output_dir / "logs"
    cache_root = output_dir / "cache_pnadium"
    for directory in (output_dir, partes_dir, logs_dir, cache_root):
        directory.mkdir(parents=True, exist_ok=True)

    anos = list(range(args.ano_inicio, args.ano_fim + 1))
    pendentes = []
    for ano in anos:
        if adotar_parte_existente(ano, partes_dir, output_dir):
            logging.info("%s já estava coletado e foi validado; pulando download.", ano)
        else:
            pendentes.append(ano)

    falhas = []
    if pendentes:
        logging.info("Coletando %d anos com %d processo(s).", len(pendentes), args.workers)
        df_ipca = get_ipca_bc()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=configurar_cache_worker,
            initargs=(str(cache_root),),
        ) as executor:
            futures = {
                executor.submit(coletar_ano, ano, partes_dir, logs_dir, df_ipca): ano
                for ano in pendentes
            }
            for future in as_completed(futures):
                ano = futures[future]
                try:
                    resultado = future.result()
                    logging.info(
                        "Ano %d concluído: %s observações.",
                        ano,
                        f"{resultado['linhas']:,}",
                    )
                except Exception as exc:
                    falhas.append((ano, str(exc)))
                    logging.error("Falha no ano %d: %s", ano, exc)

    if falhas:
        for ano, erro in falhas:
            print(f"Falha {ano}: {erro}", file=sys.stderr)
        print(f"Coleta incompleta. As partes concluídas foram preservadas em {partes_dir}.", file=sys.stderr)
        return 1

    partes = [partes_dir / f"pnad_{ano}.parquet" for ano in anos]
    for ano, path in zip(anos, partes):
        validar_arquivo_ano(path, ano)

    destino = output_dir / f"pnad_consolidada_{args.ano_inicio}_{args.ano_fim}.parquet"
    linhas = consolidar(partes, destino, args.ano_inicio, args.ano_fim)
    arquivos_fonte = registrar_arquivos_fonte(anos)
    manifesto = {
        "arquivo": destino.name,
        "periodo": [args.ano_inicio, args.ano_fim],
        "trimestres_esperados": len(anos) * 4,
        "trimestres_validados": len(anos) * 4,
        "linhas": linhas,
        "colunas_desenho": COLUNAS_DESENHO,
        "data_base_deflacao": "01/01/2026",
        "fonte": "Microdados trimestrais da PNAD Contínua / IBGE via pnadium",
        "coletado_em_utc": datetime.now(timezone.utc).isoformat(),
        "versao_pnadium": version("pnadium"),
        "arquivos_microdados_ibge": arquivos_fonte,
    }
    (output_dir / "manifesto_recoleta.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifesto, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
