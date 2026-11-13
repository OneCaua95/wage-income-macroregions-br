#!/usr/bin/env python3
"""Compara a multicolinearidade do M5 completo e sem controles do emprego."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import pesos_amostrais
from pnad_pipeline.analise.regressao.colinearidade import diagnosticar_blocos
from pnad_pipeline.analise.regressao.comparacao_m5 import controles_emprego, exportar_comparacao

SPEC = importlib.util.spec_from_file_location("_pnad_m5_multicolinearidade", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida, batch_size=100_000):
    entrada = Path(entrada)
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    nomes = ["Intercept", *M5.nomes_features(M5.UFS)]
    blocos = {
        "Escolaridade e interações": {nome for nome in nomes if nome.startswith("Estudo_")}
        | {"Sexo_Dummy", "is_formal", "is_public"},
        "Idade": {"Idade_centrada", "Idade_centrada_2"},
        "Cor/raça": set(M5.CORES.values()),
        "Setor de atividade": set(M5.SETORES.values()),
        "Ocupação": set(M5.OCUPACOES.values()),
        "UF": {nome for nome in nomes if nome.startswith("UF_")},
        "Ano": {nome for nome in nomes if nome.startswith("Ano_")},
        "Trimestre": {nome for nome in nomes if nome.startswith("Trimestre_")},
    }
    if sorted(termo for bloco in blocos.values() for termo in bloco) != sorted(nomes[1:]):
        raise ValueError("Os blocos devem cobrir cada termo do M5 exatamente uma vez.")

    completo = M5.ResultadoModelo("completo", M5.UFS)
    periodos = set()
    print("Acumulando a matriz ponderada do M5 na amostra histórica", flush=True)
    for lote, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        completo.acumular_normal(M5.matriz_m5(df, M5.UFS), df["y"].to_numpy(), pesos_amostrais(df))
        if lote % 80 == 0:
            print(f"  {lote} lotes · {completo.n:,} registros elegíveis", flush=True)

    fim = entrada.stat()
    if (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento. Repita a análise com a mesma entrada.")
    if not completo.n:
        raise ValueError("A amostra elegível do M5 ficou vazia.")
    excluir = controles_emprego(M5)
    manter = np.array([nome not in excluir for nome in nomes])
    reduzido = M5.ResultadoModelo(
        "sem controles do emprego", M5.UFS, k=int(manter.sum()), n=completo.n,
        xtx=completo.xtx[np.ix_(manter, manter)], xty=completo.xty[manter],
    )
    completo.resolver()
    reduzido.resolver()
    geral_completo, blocos_completo = diagnosticar_blocos(completo.xtx, nomes, completo.indices_ativos, blocos)
    nomes_reduzidos = [nome for nome in nomes if nome not in excluir]
    geral_reduzido, blocos_reduzido = diagnosticar_blocos(reduzido.xtx, nomes_reduzidos, reduzido.indices_ativos, blocos)
    linhas = [{
        "bloco": rotulo,
        **{f"completo_{key}": value for key, value in blocos_completo[rotulo].items()},
        **{f"reduzido_{key}": value for key, value in blocos_reduzido[rotulo].items()},
    } for rotulo in blocos]
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "multicolinearidade-m5-1.0",
        "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}",
        "processado_em": datetime.now(timezone.utc).isoformat(), "ponderacao": "V1028",
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size},
        "n_observacoes": completo.n, "soma_pesos": completo.soma_pesos, "n_periodos": len(periodos),
        "modelos": {"completo": geral_completo, "reduzido": geral_reduzido},
        "rows": linhas,
        "notas": [
            "O GVIF ajustado mede a sobreposição de cada bloco com os demais, a partir das variáveis do M5 e dos pesos V1028. Valores próximos de 1 indicam pouca sobreposição. A medida descreve a matriz da regressão. Os intervalos de confiança do M5 usam o desenho amostral da PNAD.",
            "O bloco de escolaridade reúne as categorias, suas interações e os termos principais que interagem com ela. No ajuste reduzido, esse bloco passa a incluir as categorias, o sexo e sua interação. A comparação reflete a mudança na composição do bloco e no restante do modelo.",
            "O número de condição resume a dependência entre as colunas estimáveis, após centrar e padronizar com os pesos. Colunas constantes ou redundantes seguem o critério de posto do M5 e são identificadas à parte.",
        ],
    }
    exportar_comparacao(saida, "multicolinearidade_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/multicolinearidade_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
