import numpy as np
import pytest

from pnad_pipeline.analise.regressao.previsao_m5 import metricas_ponderadas


def test_metricas_ponderadas_usam_previsao_em_nivel_sem_truncar():
    observado = np.array([2., 6.])
    previsto = np.array([-2., 7.])
    pesos = np.array([1., 3.])

    resultado = metricas_ponderadas(observado, previsto, pesos)

    assert resultado["mae_rs_h"] == 1.75
    assert resultado["rmse_rs_h"] == pytest.approx(np.sqrt(19 / 4))
    assert resultado["vies_rs_h"] == -.25
    assert resultado["n_previsoes_nao_positivas"] == 1
    vazio = metricas_ponderadas(observado[:0], previsto[:0], pesos[:0])
    assert vazio["n_registros_grupo"] == 0
    assert vazio["mae_rs_h"] is None
    assert vazio["rmse_rs_h"] is None
    assert vazio["vies_rs_h"] is None
