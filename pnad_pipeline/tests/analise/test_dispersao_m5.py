import numpy as np

from pnad_pipeline.analise.regressao.dispersao import linearizar_desvios


def test_linearizacao_reproduz_perturbacao_do_peso_da_upa_com_reajuste():
    x = np.column_stack((np.ones(9), [-3., 2., 1., 0., 3., -2., 1., -1., 2.]))
    y = np.array([1., 2., 1., 5., 0., 1., 3., 2., 7.])
    pesos = np.array([1., 3., 2., 4., 2., 2., 5., 1., 4.])
    grupos = np.repeat(np.arange(3), 3)
    upas = np.tile(np.arange(3), 3)
    bread = np.linalg.inv(x.T @ (x * pesos[:, None]))
    r = y - x @ (bread @ x.T @ (pesos * y))
    wg = pesos[:, None] * (grupos[:, None] == np.arange(3))
    wr = wg * r[:, None]
    momentos = np.zeros((3, 3, 3))
    np.add.at(momentos, upas, np.stack((wg, wr, wr * r[:, None]), axis=2))
    scores = np.zeros((3, x.shape[1]))
    np.add.at(scores, upas, x * (pesos * r)[:, None])

    desvios, derivadas = linearizar_desvios(momentos, wg.T @ x, wr.T @ x, scores, bread)

    def medir(w):
        raiz = np.sqrt(w)
        beta = np.linalg.lstsq(x * raiz[:, None], y * raiz, rcond=None)[0]
        residuos = y - x @ beta
        resultado = []
        for g in range(3):
            rg, pg = residuos[grupos == g], w[grupos == g]
            media = np.average(rg, weights=pg)
            resultado.append(np.sqrt(np.average((rg - media) ** 2, weights=pg)))
        return np.array(resultado)

    np.testing.assert_allclose(desvios, medir(pesos))
    passo = 1e-5
    for upa in range(3):
        delta = passo * (upas == upa)
        derivada = (medir(pesos * (1 + delta)) - medir(pesos * (1 - delta))) / (2 * passo)
        np.testing.assert_allclose(derivadas[upa], derivada, atol=1e-8)
