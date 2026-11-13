import numpy as np
import pytest

from pnad_pipeline.analise.regressao.residuos import resumir_residuos


def test_resumos_ponderados_mantem_o_sinal_e_a_faixa_vazia():
    grupo, vazio, constante = resumir_residuos(
        np.array([0, 0, 2]), np.array([1., -1., -2.]), np.array([3., 1., 2.]), 3
    )
    assert grupo["n_registros_grupo"] == 2
    assert grupo["parcela_ponderada"] == pytest.approx(2 / 3)
    assert grupo["media_residuo_log"] == .5
    assert grupo["desvio_padrao_residuo_log"] == pytest.approx(np.sqrt(.75))
    assert vazio["n_registros_grupo"] == 0
    assert vazio["media_residuo_log"] is None
    assert vazio["desvio_padrao_residuo_log"] is None
    assert constante["media_residuo_log"] == -2
    assert constante["desvio_padrao_residuo_log"] == 0
