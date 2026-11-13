"""Linearização da distribuição estimada nas regressões RIF de Gini e Theil."""

import numpy as np


def correcoes_distribuicao(y, w, projecoes, rif, pontos):
    """Derivada da RIF gerada, projetada nos contrastes de interesse.

    Complementa o score usual da regressão. Inclui a estimação da média,
    do Gini, do Theil e da distribuição empírica ponderada.
    """
    y, w = np.asarray(y), np.asarray(w)
    probabilidades = w / w.sum()
    media = pontos["media"]
    ordem = np.argsort(y, kind="stable")
    valores = y[ordem]
    ponderadas = projecoes[ordem] * probabilidades[ordem, None]
    acumuladas = np.cumsum(ponderadas, axis=0)
    acumuladas_y = np.cumsum(ponderadas * valores[:, None], axis=0)
    # E[projeção(X) |Y-z|], avaliada em cada salário z da amostra.
    distancia = valores[:, None] * (2 * acumuladas - acumuladas[-1]) + acumuladas_y[-1] - 2 * acumuladas_y
    media_rif = probabilidades @ (projecoes * rif[:, 0, None])
    media_xy = probabilidades @ (projecoes * y[:, None])
    gini = np.empty_like(projecoes)
    gini[ordem] = (distancia - valores[:, None] * media_rif - rif[ordem, 0, None] * media_xy) / media
    del ponderadas, acumuladas, acumuladas_y, distancia
    relativo = y / media
    derivada_theil = probabilidades @ (projecoes * (1 - relativo[:, None]))
    derivada_media = probabilidades @ (projecoes * (relativo * (np.log(relativo) - pontos["theil_t"]))[:, None])
    theil = (rif[:, 1] - pontos["theil_t"])[:, None] * derivada_theil - (relativo - 1)[:, None] * derivada_media
    return np.column_stack((gini, theil))
