"""Funções de influência ponderadas para Gini, Theil e participação do topo."""

import numpy as np
from scipy.stats import t

from pnad_pipeline.analise.amostragem import pesos_amostrais, quantis_ponderados


def estatisticas_influencias(y, w):
    y, w = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    if y.ndim != 1 or y.shape != w.shape or not len(y):
        raise ValueError("Salários e pesos devem ser vetores não vazios do mesmo tamanho.")
    if not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(y <= 0) or np.any(w <= 0):
        raise ValueError("Salários e pesos devem ser positivos e finitos.")
    ordem = np.argsort(y, kind="stable")
    valores, pesos = y[ordem], w[ordem]
    acumulados = np.cumsum(pesos)
    renda = np.cumsum(pesos * valores)
    total, soma_renda = acumulados[-1], renda[-1]
    media = soma_renda / total
    distancia = (2 * (valores * acumulados - renda) - valores * total + soma_renda) / total
    gini = float(max(pesos @ distancia / (2 * media * total), 0))
    relativo = valores / media
    theil = float(max(pesos @ (relativo * np.log(relativo)) / total, 0))
    q90 = valores[min(np.searchsorted(acumulados, .9 * total), len(valores) - 1)]
    excedente = np.maximum(valores - q90, 0)
    topo = float((pesos @ excedente / total + .1 * q90) / media)
    influencias = np.empty((len(y), 3))
    influencias[ordem, 0] = distancia / media - gini * (1 + relativo)
    influencias[ordem, 1] = relativo * np.log(relativo) - (theil + 1) * relativo + 1
    influencias[ordem, 2] = (excedente + .1 * q90 - topo * valores) / media
    return {"gini": gini, "theil_t": theil, "participacao_top10": topo, "media": float(media)}, influencias


def intervalos_desigualdade(df, desenho):
    if len(df) < 30:
        raise ValueError("O domínio exige pelo menos 30 salários válidos.")
    w = pesos_amostrais(df)
    y = df.Salario_Hora.to_numpy(dtype=float)
    pontos, influencias = estatisticas_influencias(y, w)
    q10, q90 = quantis_ponderados(df, "Salario_Hora", [.1, .9])
    indicadores = y[:, None] <= [q10, q90]
    proporcoes = w @ indicadores / w.sum()
    scores = np.column_stack((influencias, indicadores - proporcoes)) * (w / w.sum())[:, None]
    desenho.iniciar_scores(5)
    desenho.acumular(df, scores)
    erros = np.sqrt(np.maximum(np.diag(desenho.covariancia(np.eye(5))), 0))
    critico = t.ppf(.975, desenho.graus_liberdade)
    rows = []
    for i, (indicador, escala, minimo, maximo) in enumerate([
        ("gini", 1., 0., 1.), ("theil_t", 1., 0., np.inf), ("participacao_top10", 100., .1, 1.),
    ]):
        ponto = pontos[indicador]
        rows.append({"indicador": "participacao_top10_pct" if i == 2 else indicador,
                     "estimativa": float(escala * ponto), "erro_padrao": float(escala * erros[i]),
                     "ic_inferior": float(escala * max(minimo, ponto - critico * erros[i])),
                     "ic_superior": float(escala * min(maximo, ponto + critico * erros[i])),
                     "metodo": "Função de influência e linearização por UPA/estrato", "status": "ok"})
    # Dois ICs de Woodruff a 97,5% dão uma região simultânea conservadora de 95%.
    critico_quantis = t.ppf(1 - .05 / 4, desenho.graus_liberdade)
    limites = []
    for i in range(2):
        probabilidades = proporcoes[i] + np.array([-1, 1]) * critico_quantis * erros[i + 3]
        limites.append([float(quantis_ponderados(df, "Salario_Hora", [p])[0]) if 0 <= p <= 1 else None for p in probabilidades])
    disponivel = all(v is not None for par in limites for v in par)
    rows.append({"indicador": "razao_P90_P10", "estimativa": float(q90 / q10),
                 "ic_inferior": float(max(1, limites[1][0] / limites[0][1])) if disponivel else None,
                 "ic_superior": float(limites[1][1] / limites[0][0]) if disponivel else None,
                 "metodo": "Woodruff simultâneo para P10/P90 com Bonferroni; intervalo conservador",
                 "status": "ok" if disponivel else "limites_indisponiveis"})
    return rows
