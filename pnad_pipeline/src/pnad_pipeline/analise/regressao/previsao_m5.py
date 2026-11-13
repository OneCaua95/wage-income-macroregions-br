"""Métricas ponderadas para previsões comparadas na escala salarial real."""

import numpy as np


def metricas_ponderadas(observado, previsto, pesos):
    total = float(pesos.sum())
    erro = previsto - observado
    return {
        "n_registros_grupo": len(erro), "soma_pesos_grupo": total,
        "mae_rs_h": float(pesos @ np.abs(erro) / total) if total else None,
        "rmse_rs_h": float(np.sqrt(pesos @ (erro ** 2) / total)) if total else None,
        "vies_rs_h": float(pesos @ erro / total) if total else None,
        "n_previsoes_nao_positivas": int((previsto <= 0).sum()),
    }
