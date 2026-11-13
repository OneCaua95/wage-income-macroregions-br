"""Estatísticas descritivas ponderadas dos resíduos por grupo."""

import numpy as np


def resumir_residuos(codigos, residuos, pesos, n_grupos):
    contagens = np.bincount(codigos, minlength=n_grupos)
    somas = np.bincount(codigos, weights=pesos, minlength=n_grupos)
    primeiras = np.bincount(codigos, weights=pesos * residuos, minlength=n_grupos)
    segundas = np.bincount(codigos, weights=pesos * residuos ** 2, minlength=n_grupos)
    total = pesos.sum()
    linhas = []
    for n, peso, primeira, segunda in zip(contagens, somas, primeiras, segundas):
        media = desvio = None
        if peso > 0:
            media = float(primeira / peso)
            desvio = float(np.sqrt(max(0, segunda / peso - media ** 2)))
        linhas.append({
            "n_registros_grupo": int(n), "soma_pesos_grupo": float(peso),
            "parcela_ponderada": float(peso / total),
            "media_residuo_log": media, "desvio_padrao_residuo_log": desvio,
        })
    return linhas
