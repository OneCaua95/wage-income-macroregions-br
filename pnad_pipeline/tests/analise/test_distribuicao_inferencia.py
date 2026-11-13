"""Derivadas numéricas independentes das estatísticas e regressões RIF."""

import numpy as np
import pandas as pd
import pytest

from pnad_pipeline.analise.crescimento import comparar_medias
from pnad_pipeline.analise.amostragem import LinearizacaoUPA
from pnad_pipeline.analise.distribuicao_inferencia import estatisticas_influencias, intervalos_desigualdade
from pnad_pipeline.analise.regressao.rif import correcoes_distribuicao


def dados():
    rng = np.random.default_rng(43)
    return np.exp(rng.normal(2, .7, 60)), rng.uniform(.5, 4, 60), np.column_stack((np.ones(60), rng.normal(size=(60, 3))))


def ajustar(y, w, x):
    pontos, influencias = estatisticas_influencias(y, w)
    rif = influencias[:, :2] + [pontos["gini"], pontos["theil_t"]]
    beta = np.linalg.lstsq(x * np.sqrt(w[:, None]), rif * np.sqrt(w[:, None]), rcond=None)[0]
    return beta, pontos, rif


def test_funcoes_influencia_reproduzem_perturbacoes_dos_pesos():
    y, w, _ = dados()
    pontos, influencias = estatisticas_influencias(y, w)
    np.testing.assert_allclose(w @ influencias, 0, atol=1e-12)
    for i in [0, 7, 30, 59]:
        wp, wm = w.copy(), w.copy()
        wp[i] *= 1 + 1e-5
        wm[i] *= 1 - 1e-5
        a, _ = estatisticas_influencias(y, wp)
        b, _ = estatisticas_influencias(y, wm)
        numerico = np.array([(a[k] - b[k]) / 2e-5 for k in ["gini", "theil_t", "participacao_top10"]])
        np.testing.assert_allclose(numerico, w[i] / w.sum() * influencias[i], atol=1e-9, rtol=1e-5)
    relativo = y / np.average(y, weights=w)
    assert pontos["theil_t"] == pytest.approx(np.average(relativo * np.log(relativo), weights=w))
    absoluto = np.abs(y[:, None] - y)
    assert pontos["gini"] == pytest.approx(w @ absoluto @ w / (2 * w.sum() * (w @ y)))


def test_scores_rif_incluem_a_distribuicao_gerada_e_a_regressao():
    y, w, x = dados()
    beta, pontos, rif = ajustar(y, w, x)
    contrastes = np.eye(x.shape[1])[[0, 1, 3]]
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    f = x @ bread @ contrastes.T
    correcoes = correcoes_distribuicao(y, w, f, rif, pontos)
    residuo = rif - x @ beta
    scores = w[:, None] * (np.column_stack((f * residuo[:, 0, None], f * residuo[:, 1, None])) + correcoes)
    np.testing.assert_allclose(scores.sum(axis=0), 0, atol=1e-12)
    for i in [0, 7, 30, 59]:
        wp, wm = w.copy(), w.copy()
        wp[i] *= 1 + 1e-5
        wm[i] *= 1 - 1e-5
        bp, _, _ = ajustar(y, wp, x)
        bm, _, _ = ajustar(y, wm, x)
        numerico = (contrastes @ ((bp - bm) / 2e-5)).T.ravel()
        np.testing.assert_allclose(scores[i], numerico, atol=1e-9, rtol=1e-5)


def test_crescimento_usa_covariancia_e_mantem_referencia_identica():
    sem_cov = comparar_medias(10, 12, np.diag([.25, .36]), 100)
    com_cov = comparar_medias(10, 12, np.array([[.25, .25], [.25, .36]]), 100)
    assert sem_cov["estimativa"] == pytest.approx(20)
    assert com_cov["ic_superior"] - com_cov["ic_inferior"] < sem_cov["ic_superior"] - sem_cov["ic_inferior"]
    identidade = comparar_medias(10, 10, np.full((2, 2), .25), 100)
    assert identidade["estimativa"] == identidade["ic_inferior"] == identidade["ic_superior"] == 0


def test_intervalos_respeitam_suporte_empates_e_escala_dos_pesos():
    y, w, _ = dados()
    df = pd.DataFrame({"UF": 35, "Estrato": np.repeat([1, 2], 30),
                       "UPA": np.repeat([11, 12, 21, 22], 15), "V1028": w, "Salario_Hora": y})
    desenho = LinearizacaoUPA()
    desenho.registrar(df)
    rows = intervalos_desigualdade(df, desenho)
    df.V1028 *= 10
    scaled = intervalos_desigualdade(df, desenho)
    for a, b in zip(rows, scaled):
        assert a["estimativa"] == pytest.approx(b["estimativa"])
        assert a["ic_inferior"] == pytest.approx(b["ic_inferior"])
        assert a["ic_superior"] == pytest.approx(b["ic_superior"])
        if a["status"] == "ok":
            assert 0 <= a["ic_inferior"] <= a["estimativa"] <= a["ic_superior"]
    df.Salario_Hora = 5.
    iguais = intervalos_desigualdade(df, desenho)
    assert iguais[0]["estimativa"] == pytest.approx(0, abs=1e-12)
    assert iguais[1]["estimativa"] == pytest.approx(0, abs=1e-12)
    assert iguais[2]["estimativa"] == pytest.approx(10)
    assert iguais[3]["estimativa"] == pytest.approx(1)
