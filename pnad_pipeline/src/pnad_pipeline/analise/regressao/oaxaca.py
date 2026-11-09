"""Oaxaca em duas parcelas com referência pooled e indicador de grupo.

O último coeficiente do modelo é o indicador do grupo comparado. As médias
de X não incluem esse indicador. Os scores incorporam a incerteza conjunta
das médias ponderadas e dos coeficientes; devem ser agregados pelo desenho.
"""
from __future__ import annotations

import numpy as np


def decompor(beta, media_x_grupo, media_x_referencia, media_y_grupo, media_y_referencia):
    """Retorna total, explicada e não explicada, em unidades de log."""
    beta = np.asarray(beta, dtype=float)
    delta_x = np.asarray(media_x_grupo) - np.asarray(media_x_referencia)
    if beta.shape != (len(delta_x) + 1,):
        raise ValueError("O último coeficiente deve representar o indicador de grupo.")
    total = float(media_y_grupo - media_y_referencia)
    explicada = float(delta_x @ beta[:-1])
    return np.array([total, explicada, total - explicada])


def influencias(x, y, grupo, pesos, beta, bread, media_x_grupo, media_x_referencia,
                media_y_grupo, media_y_referencia, peso_grupo, peso_referencia):
    """Linearização de total, explicada e não explicada em cada observação.

    `grupo` é booleano: True para o grupo comparado, False para a referência.
    `x` contém os controles comuns, incluindo intercepto. `bread` é a inversa
    de Z'WZ no espaço identificável, Z = [X, indicador de grupo]. As linhas
    devem conter apenas os dois grupos; UPAs fora do domínio recebem zero
    no cadastro completo do desenho.
    """
    if peso_grupo <= 0 or peso_referencia <= 0:
        raise ValueError("Os dois grupos precisam ter peso positivo.")
    grupo = np.asarray(grupo, dtype=bool)
    pesos, beta = np.asarray(pesos, dtype=float), np.asarray(beta, dtype=float)
    media_x_grupo, media_x_referencia = np.asarray(media_x_grupo), np.asarray(media_x_referencia)
    z = np.column_stack([x, grupo.astype(float)])
    residuo = y - z @ beta
    delta_x = media_x_grupo - media_x_referencia
    gradiente = np.r_[delta_x, 0.0]
    multiplicador_g = grupo / peso_grupo
    multiplicador_r = (~grupo) / peso_referencia
    total = pesos * (multiplicador_g * (y - media_y_grupo)
                     - multiplicador_r * (y - media_y_referencia))
    # Variação das médias de X e dos coeficientes, com suas covariâncias.
    explicada = pesos * (
        multiplicador_g * ((x - media_x_grupo) @ beta[:-1])
        - multiplicador_r * ((x - media_x_referencia) @ beta[:-1])
        + residuo * (z @ (bread @ gradiente))
    )
    return np.column_stack([total, explicada, total - explicada])
