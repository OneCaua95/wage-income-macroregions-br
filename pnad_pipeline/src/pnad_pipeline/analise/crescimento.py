"""Crescimento de médias com covariância conjunta entre dois períodos."""

import numpy as np
from scipy.stats import t


def comparar_medias(base, atual, cov, gl):
    if not np.isfinite([base, atual]).all() or min(base, atual) <= 0:
        raise ValueError("As médias salariais devem ser positivas e finitas.")
    gradiente = np.array([-1 / base, 1 / atual])
    ep_log = np.sqrt(max(float(gradiente @ cov @ gradiente), 0))
    margem = t.ppf(.975, gl) * ep_log
    log_razao = np.log(atual / base)
    return {"estimativa": float(100 * np.expm1(log_razao)),
            "ic_inferior": float(100 * np.expm1(log_razao - margem)),
            "ic_superior": float(100 * np.expm1(log_razao + margem)),
            "diferenca_reais_hora": float(atual - base), "erro_padrao_log_razao": float(ep_log),
            "covariancia_medias": float(cov[0, 1])}
