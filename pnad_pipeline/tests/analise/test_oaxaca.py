"""Identidade e linearização Oaxaca contra perturbações independentes dos pesos."""
import numpy as np
import pandas as pd
import pytest

from pnad_pipeline.analise.amostragem import LinearizacaoUPA
from pnad_pipeline.analise.regressao.oaxaca import decompor, influencias


def dados():
    rng = np.random.default_rng(2026)
    grupo = np.tile([False, True], 20)
    x = np.column_stack([np.ones(40), rng.normal(size=40) + grupo * .7, rng.uniform(0, 2, 40)])
    y = x @ [2, .45, -.2] - grupo * .3 + rng.normal(0, .2, 40)
    return x, y, grupo, rng.uniform(.5, 4, 40)


def ajustar(x, y, grupo, w):
    z = np.column_stack([x, grupo.astype(float)])
    # Solução independente por mínimos quadrados, sem usar o acumulador do pipeline.
    beta = np.linalg.lstsq(z * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    bread = np.linalg.inv(z.T @ (w[:, None] * z))
    mxg, mxr = (np.average(x[mask], axis=0, weights=w[mask]) for mask in (grupo, ~grupo))
    myg, myr = (np.average(y[mask], weights=w[mask]) for mask in (grupo, ~grupo))
    values = decompor(beta, mxg, mxr, myg, myr)
    scores = influencias(x, y, grupo, w, beta, bread, mxg, mxr, myg, myr, w[grupo].sum(), w[~grupo].sum())
    return values, scores, beta, bread


def test_scores_incluem_variacao_das_medias_e_dos_coeficientes():
    x, y, grupo, w = dados()
    values, scores, beta, _ = ajustar(x, y, grupo, w)
    assert values[0] == pytest.approx(values[1] + values[2])
    assert values[2] == pytest.approx(beta[-1])
    np.testing.assert_allclose(scores.sum(axis=0), 0, atol=1e-13)
    for i in (0, 1, 8, 17, 38, 39):
        step = 1e-5
        wp, wm = w.copy(), w.copy()
        wp[i] *= 1 + step
        wm[i] *= 1 - step
        numerical = (ajustar(x, y, grupo, wp)[0] - ajustar(x, y, grupo, wm)[0]) / (2 * step)
        np.testing.assert_allclose(scores[i], numerical, atol=1e-9, rtol=1e-6)


def test_resultados_e_scores_invariantes_a_escala_dos_pesos_e_ordem_dos_grupos():
    x, y, grupo, w = dados()
    values, scores, _, _ = ajustar(x, y, grupo, w)
    scaled, scaled_scores, _, _ = ajustar(x, y, grupo, w * 1000)
    reversed_values, reversed_scores, _, _ = ajustar(x, y, ~grupo, w)
    np.testing.assert_allclose(values, scaled, atol=1e-12)
    np.testing.assert_allclose(scores, scaled_scores, atol=1e-12)
    np.testing.assert_allclose(values, -reversed_values, atol=1e-12)
    np.testing.assert_allclose(scores, -reversed_scores, atol=1e-12)


def test_variancia_por_upa_fecha_com_indicador_pooled_e_covariancia_das_parcelas():
    x, y, grupo, w = dados()
    _, scores, beta, bread = ajustar(x, y, grupo, w)
    df = pd.DataFrame({"UF": [35]*40, "Estrato": np.repeat([1, 2], 20), "UPA": np.repeat(np.arange(1, 11), 4), "V1028": w})
    desenho = LinearizacaoUPA()
    desenho.registrar(df)
    desenho.iniciar_scores(3)
    desenho.acumular(df, scores)
    cov = desenho.covariancia(np.eye(3))
    z = np.column_stack([x, grupo.astype(float)])
    desenho_beta = LinearizacaoUPA()
    desenho_beta.registrar(df)
    desenho_beta.iniciar_scores(z.shape[1])
    desenho_beta.acumular(df, z * (w * (y - z @ beta))[:, None])
    assert cov[2, 2] == pytest.approx(desenho_beta.covariancia(bread)[-1, -1])
    assert cov[0, 0] == pytest.approx(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2])
    assert np.linalg.eigvalsh(cov).min() > -1e-12
