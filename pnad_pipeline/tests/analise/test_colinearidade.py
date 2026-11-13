import numpy as np
import pytest

from pnad_pipeline.analise.regressao.colinearidade import diagnosticar_blocos


def test_gvif_de_bloco_ponderado_e_colunas_omitidas():
    # A e C têm correlação .6. B é ortogonal e Constante replica o intercepto.
    correlacao = np.array([[1, 0, .6, 0], [0, 1, 0, 0], [.6, 0, 1, 0], [0, 0, 0, 0]])
    escala = [2, 3, 7, 1]
    medias = [5, -2, 9, 1]
    xtwx = np.ones((5, 5)) * 17
    xtwx[0, 1:] *= medias
    xtwx[1:, 0] = xtwx[0, 1:]
    xtwx[1:, 1:] *= correlacao * np.outer(escala, escala) + np.outer(medias, medias)

    geral, blocos = diagnosticar_blocos(
        xtwx, ["Intercept", "A", "B", "C", "Constante"], np.array([0, 1, 2, 3]),
        {"AB": {"A", "B"}, "C": {"C"}, "Constante": {"Constante"}, "Excluído": {"D"}},
    )

    gvif = 1 / (1 - .6**2)
    assert blocos["AB"]["gvif"] == pytest.approx(gvif)
    assert blocos["AB"]["gvif_ajustado"] == pytest.approx(gvif**.25)
    assert blocos["C"]["gvif_ajustado"] == pytest.approx(gvif**.5)
    assert geral["numero_condicao"] == pytest.approx(2)
    assert geral["omitidos"] == ["Constante"]
    assert blocos["Constante"]["n_colunas"] == 1
    assert blocos["Constante"]["n_ativas"] == 0
    assert blocos["Constante"]["gvif"] is None
    assert blocos["Excluído"]["n_colunas"] == 0
    assert blocos["Excluído"]["gvif"] is None
