#!/usr/bin/env python3
"""Regressões RIF ponderadas para Gini e Theil, com controles do M5."""

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.distribuicao_inferencia import estatisticas_influencias
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao
from pnad_pipeline.analise.regressao.multiplicidade import ajustar_holm
from pnad_pipeline.analise.regressao.rif import correcoes_distribuicao
from pnad_pipeline.analise.regressao.robustez import contrastes_m5

SPEC = importlib.util.spec_from_file_location("_m5_rif", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)
HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"


def executar(entrada, saida, batch_size=100_000):
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
    modelo = M5.ResultadoModelo("rif", M5.UFS)
    salarios, pesos, periodos = [], [], set()
    print("Passada 1/3: distribuição histórica e matriz do M5", flush=True)
    for i, batch in enumerate(M5.iterar_lotes(entrada, batch_size), 1):
        desenho.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        w = pesos_amostrais(df)
        modelo.acumular_normal(M5.matriz_m5(df, M5.UFS), df.y.to_numpy(), w)
        salarios.append(np.exp(df.y.to_numpy(dtype=float)))
        pesos.append(w)
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        if i % 100 == 0:
            print(f"  {i} lotes", flush=True)
    modelo.resolver()
    y, w = np.concatenate(salarios), np.concatenate(pesos)
    del salarios, pesos
    pontos, influencias = estatisticas_influencias(y, w)
    rif = influencias[:, :2] + [pontos["gini"], pontos["theil_t"]]
    del influencias
    nomes = ["Intercept", *M5.nomes_features(M5.UFS)]
    rotulos, contrastes, escolaridade = contrastes_m5(nomes, modelo.xtx)
    if not np.isin(np.flatnonzero(np.any(contrastes != 0, axis=0)), modelo.indices_ativos).all():
        raise ValueError("A amostra não identifica os contrastes RIF.")
    projecao = modelo.bread @ contrastes.T
    projecoes = np.empty((modelo.n, len(rotulos)))
    xty = np.zeros((modelo.k, 2))
    offset = 0
    print("Passada 2/3: ajustando Gini e Theil na mesma amostra", flush=True)
    for batch in M5.iterar_lotes(entrada, batch_size):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        fim = offset + len(df)
        x = M5.matriz_m5(df, M5.UFS)
        xty += x.T @ (w[offset:fim, None] * rif[offset:fim])
        projecoes[offset:fim] = x @ projecao
        offset = fim
    if offset != modelo.n:
        raise ValueError("A amostra mudou entre as passadas RIF.")
    beta = modelo.bread @ xty
    print("Calculando a contribuição da estimação da distribuição", flush=True)
    correcoes = correcoes_distribuicao(y, w, projecoes, rif, pontos)
    del projecoes, y
    dimensao = 2 * len(rotulos)
    desenho.iniciar_scores(dimensao)
    offset = 0
    print("Passada 3/3: linearização conjunta por UPA e estrato", flush=True)
    for batch in M5.iterar_lotes(entrada, batch_size):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        fim = offset + len(df)
        x = M5.matriz_m5(df, M5.UFS)
        f = x @ projecao
        residuo = rif[offset:fim] - x @ beta
        scores = np.column_stack((f * residuo[:, 0, None], f * residuo[:, 1, None])) + correcoes[offset:fim]
        desenho.acumular(df, scores * w[offset:fim, None])
        offset = fim
    final = entrada.stat()
    if offset != modelo.n or (origem.st_size, origem.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento RIF.")
    cov = desenho.covariancia(np.eye(dimensao))
    estimativas = (contrastes @ beta).T.ravel()
    erros = np.sqrt(np.maximum(np.diag(cov), 0))
    p = 2 * t.sf(np.abs(estimativas / erros), desenho.graus_liberdade)
    holm = ajustar_holm(p)
    critico = t.ppf(.975, desenho.graus_liberdade)
    rows = []
    for j, (estatistica, escala, unidade) in enumerate([("gini", 100., "Pontos de Gini · escala 0–100"), ("theil_t", 1., "Unidades de Theil T")]):
        for i, contraste in enumerate(rotulos):
            indice = j * len(rotulos) + i
            b, se = estimativas[indice], erros[indice]
            rows.append({"estatistica": estatistica, "contraste": contraste, "unidade": unidade,
                         "estimativa": float(escala * b), "erro_padrao": float(escala * se),
                         "ic_inferior": float(escala * (b - critico * se)), "ic_superior": float(escala * (b + critico * se)),
                         "p_valor": float(p[indice]), "p_holm": float(holm[indice])})
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "rif-desigualdade-1.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}", "ponderacao": "V1028",
        "n_observacoes": modelo.n, "n_periodos": len(periodos), "graus_liberdade": desenho.graus_liberdade,
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size}, "estatisticas_distribuicao": pontos,
        "media_ponderada_anos_estudo": escolaridade, "rows": rows,
        "notas": [
            "As RIFs são calculadas na distribuição nacional ponderada do salário-hora real da série 2013–2025 empilhada. A amostra, os controles e os três perfis de contraste seguem o M5 histórico.",
            "As estimativas são contrastes da projeção linear da RIF. Gini é expresso em pontos da escala 0–100; Theil usa suas unidades originais. Sinal positivo indica associação com valores maiores da RIF no perfil de referência.",
            "A linearização conjunta inclui a estimação da distribuição empírica, da média, do Gini e do Theil que geram as respostas RIF. Os perfis médios dos contrastes são tratados como fixos. IC95% pontuais por UPA dentro de estrato, com aproximação temporal empilhada.",
            "Holm considera a família fixa dos seis contrastes publicados. As associações dependem da projeção e dos controles e não identificam efeito causal ou uma previsão de mudança individual do índice.",
        ],
        "referencia": "https://www.mdpi.com/2225-1146/6/2/28",
    }
    exportar_comparacao(saida, "rif_desigualdade", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/rif_desigualdade")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        executar(args.entrada, args.saida_dir, args.batch_size)
