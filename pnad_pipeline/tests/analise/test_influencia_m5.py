import numpy as np

from pnad_pipeline.analise.regressao.influencia import mudancas_por_exclusao, selecionar_influentes


def test_atualizacao_ponderada_reproduz_cada_reajuste_com_contrastes_fixos():
    x = np.array([[1., -2, 0], [1, -1, 1], [1, 0, 0], [1, 1, 1], [1, 3, 0]])
    y = np.array([.4, .9, .2, .6, .7])
    pesos = np.array([2., 5, 1, 3, 9])
    contrastes = np.array([[0, 1, 0], [0, 2, -1]])
    xw = x * np.sqrt(pesos)[:, None]
    yw = y * np.sqrt(pesos)
    beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
    bread = np.linalg.inv(xw.T @ xw)

    _, mudancas = mudancas_por_exclusao(x, y, pesos, beta, bread, contrastes)

    for i in range(len(y)):
        manter = np.arange(len(y)) != i
        beta_sem = np.linalg.lstsq(xw[manter], yw[manter], rcond=None)[0]
        esperado = 100 * (np.exp(contrastes @ beta_sem) - np.exp(contrastes @ beta))
        np.testing.assert_allclose(mudancas[i], esperado, atol=1e-11)


def test_exclusao_que_torna_um_termo_inestimavel_nao_recebe_uma_mudanca():
    x = np.array([[1., 0], [1, 0], [1, 1]])
    y, pesos = np.array([1., 2, 5]), np.array([1., 1, 2])
    bread = np.linalg.inv(x.T @ (x * pesos[:, None]))
    beta = bread @ x.T @ (y * pesos)
    _, mudancas = mudancas_por_exclusao(x, y, pesos, beta, bread, np.array([[0., 1]]))
    assert np.isfinite(mudancas[:2]).all()
    assert np.isnan(mudancas[2, 0])


def test_selecao_por_magnitude_preserva_empates_e_descarta_nulos():
    mudancas = np.r_[np.zeros(106), [-4., 2., -2., np.nan]]
    corte, influentes = selecionar_influentes(mudancas, np.isfinite(mudancas))
    assert corte == 2
    np.testing.assert_array_equal(np.flatnonzero(influentes), [106, 107, 108])
    assert not selecionar_influentes(np.zeros(3), np.ones(3, dtype=bool))[1].any()
