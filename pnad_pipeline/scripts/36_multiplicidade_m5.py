#!/usr/bin/env python3
"""Testes por bloco e IC95% simultâneos para três contrastes do M5 histórico."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao
from pnad_pipeline.analise.regressao.multiplicidade import ajustar_holm, calcular_testes_blocos
from pnad_pipeline.analise.regressao.robustez import contrastes_m5

SPEC = importlib.util.spec_from_file_location("_pnad_m5_multiplicidade", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)
HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida, batch_size=100_000):
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
    modelo = M5.ResultadoModelo("geral", M5.UFS)
    periodos = set()
    print("Passada 1/2: ajustando o M5 histórico", flush=True)
    for batch in M5.iterar_lotes(entrada, batch_size):
        desenho.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        modelo.acumular_normal(M5.matriz_m5(df, M5.UFS), df.y.to_numpy(), pesos_amostrais(df))
    modelo.resolver()
    desenho.iniciar_scores(modelo.k)
    n_scores = 0
    print("Passada 2/2: linearizando todos os coeficientes", flush=True)
    for batch in M5.iterar_lotes(entrada, batch_size):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        x = M5.matriz_m5(df, M5.UFS)
        residuo = pesos_amostrais(df) * (df.y.to_numpy() - x @ modelo.beta)
        desenho.acumular(df, x * residuo[:, None])
        n_scores += len(df)
    fim = entrada.stat()
    if n_scores != modelo.n or (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou entre as duas passadas.")
    cov = desenho.covariancia(modelo.bread)
    nomes = ["Intercept", *M5.nomes_features(M5.UFS)]
    rotulos, contrastes, media = contrastes_m5(nomes, modelo.xtx)
    if not np.isin(np.flatnonzero(np.any(contrastes != 0, axis=0)), modelo.indices_ativos).all():
        raise ValueError("A amostra não identifica os três contrastes.")
    gl = desenho.graus_liberdade
    estimativas = contrastes @ modelo.beta
    erros = np.sqrt(np.maximum(np.diag(contrastes @ cov @ contrastes.T), 0))
    p = 2 * stats.t.sf(np.abs(estimativas / erros), gl)
    holm = ajustar_holm(p)
    critico, simultaneo = stats.t.ppf([.975, 1 - .05 / (2 * len(rotulos))], gl)
    rows = []
    for i, rotulo in enumerate(rotulos):
        beta, se = estimativas[i], erros[i]
        rows.append({
            "contraste": rotulo, "associacao_pct": float(100 * np.expm1(beta)),
            "ic_inferior_pct": float(100 * np.expm1(beta - critico * se)),
            "ic_superior_pct": float(100 * np.expm1(beta + critico * se)),
            "ic_simultaneo_inferior_pct": float(100 * np.expm1(beta - simultaneo * se)),
            "ic_simultaneo_superior_pct": float(100 * np.expm1(beta + simultaneo * se)),
            "p_valor": float(p[i]), "p_holm": float(holm[i]),
        })
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "multiplicidade-m5-1.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}", "ponderacao": "V1028",
        "n_observacoes": modelo.n, "n_periodos": len(periodos), "graus_liberdade": gl,
        "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size}, "nivel_confianca": .95,
        "media_ponderada_anos_estudo": media, "rows": rows,
        "testes_blocos": calcular_testes_blocos(nomes, modelo.beta, cov, modelo.indices_ativos, gl),
        "notas": [
            "Cada teste conjunto verifica se todos os coeficientes do bloco são zero. A correção de Holm considera a família fixa de blocos identificados, antes dos filtros de visualização.",
            "Os três contrastes formam uma família separada. Os IC95% simultâneos usam Bonferroni para os três contrastes e a distribuição t do desenho. O perfil ponderado de referência é tratado como fixo.",
            "A tabela de coeficientes oferece p-valores BH e Holm para todos os coeficientes identificados de cada ajuste, incluindo o intercepto, como na família BH histórica. As correções não mudam com busca ou filtros da tabela.",
        ],
    }
    exportar_comparacao(saida, "multiplicidade_m5", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/multiplicidade_m5")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
