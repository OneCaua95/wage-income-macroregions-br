"""Mudanças em contrastes do M5 ao excluir uma observação por vez."""

import numpy as np


def mudancas_por_exclusao(x, y, pesos, beta, bread, contrastes):
    """Atualização exata de mínimos quadrados ponderados com contrastes fixos."""
    ajustes = x @ beta
    x_bread = x @ bread
    alavancagem = pesos * np.einsum("ij,ij->i", x_bread, x)
    denominador = 1 - alavancagem
    fator = np.full(len(x), np.nan)
    # Perto de h=1, retirar a linha pode eliminar a identificação de um termo.
    np.divide(-pesos * (y - ajustes), denominador, out=fator, where=denominador > 1e-10)
    mudanca_log = (x_bread @ contrastes.T) * fator[:, None]
    mudanca_pp = 100 * np.exp(contrastes @ beta) * np.expm1(mudanca_log)
    return ajustes, mudanca_pp


def selecionar_influentes(mudancas, validos):
    """Seleciona o 1% de maiores mudanças absolutas, incluindo empates no corte."""
    magnitudes = np.abs(mudancas)
    ordenaveis = magnitudes[validos]
    inicio_cauda = len(ordenaveis) - (len(ordenaveis) + 99) // 100
    ordenaveis.partition(inicio_cauda)
    corte = ordenaveis[inicio_cauda]
    return float(corte), validos & (magnitudes >= corte) & (magnitudes > 0)
