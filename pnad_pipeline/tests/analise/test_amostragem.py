import numpy as np
import pandas as pd
import pytest

from pnad_pipeline.analise.amostragem import (
    LinearizacaoUPA, diagnostico_representatividade, media_ponderada,
    pesos_amostrais, quantis_ponderados, validar_desenho,
)


@pytest.fixture
def amostra():
    return pd.DataFrame({
        "UF": [35] * 4, "Estrato": [1, 1, 2, 2], "UPA": [11, 12, 21, 22],
        "V1028": [1.0, 3.0, 2.0, 4.0], "valor": [10.0, 20.0, 30.0, 40.0],
        "Ano": [2025] * 4, "Trimestre": [1] * 4,
    })


@pytest.mark.parametrize("peso", [0, -1, np.nan, np.inf, "inválido"])
def test_rejeita_peso_invalido(amostra, peso):
    with pytest.raises(ValueError, match="V1028"):
        pesos_amostrais(amostra.assign(V1028=peso))


def test_medias_quantis_e_diagnostico(amostra):
    assert media_ponderada(amostra, "valor") == 29.0
    np.testing.assert_array_equal(quantis_ponderados(amostra, "valor", [0, .5, .9, 1]), [10, 30, 40, 40])
    row = diagnostico_representatividade(amostra).iloc[0]
    assert row["n_amostral"] == 4
    assert row["total_pessoas_dominio"] == 10
    assert row["n_efetivo_kish"] == pytest.approx(100 / 30)


def test_peso_de_resultado_ausente_nao_entra_no_denominador(amostra):
    amostra.loc[3, "valor"] = np.nan
    assert media_ponderada(amostra, "valor") == pytest.approx(130 / 6)


def test_linearizacao_concorda_com_calculo_manual(amostra):
    desenho = LinearizacaoUPA()
    desenho.registrar(amostra)
    desenho.iniciar_scores(1)
    desenho.acumular(amostra, np.array([[3.0], [1.0], [-1.0], [1.0]]))
    # Cada estrato: 2/(2-1) * (1² + (-1)²) = 4; bread = 1/10.
    np.testing.assert_allclose(desenho.covariancia(np.array([[.1]])), [[.08]])
    assert desenho.graus_liberdade == 2


def test_upa_sem_observacoes_no_dominio_contribui_com_zero(amostra):
    desenho = LinearizacaoUPA()
    desenho.registrar(amostra)
    desenho.iniciar_scores(1)
    desenho.acumular(amostra.iloc[[0]], np.array([[2.0]]))
    np.testing.assert_allclose(desenho.covariancia(np.eye(1)), [[4.0]])
    assert len(desenho.grupos) == 4


def test_mesma_upa_entre_trimestres_nao_vira_novo_conglomerado(amostra):
    desenho = LinearizacaoUPA()
    desenho.registrar(amostra)
    outro = amostra.assign(Trimestre=2)
    desenho.registrar(outro)
    desenho.iniciar_scores(1)
    desenho.acumular(amostra, np.ones((4, 1)))
    desenho.acumular(outro, np.full((4, 1), 2.0))
    assert len(desenho.grupos) == 4
    np.testing.assert_allclose(desenho.scores, np.full((4, 1), 3.0))


def test_estrato_com_uma_upa_nao_ganha_variancia_zero(amostra):
    desenho = LinearizacaoUPA()
    desenho.registrar(amostra.iloc[[0]])
    desenho.iniciar_scores(1)
    with pytest.raises(ValueError, match="única UPA"):
        desenho.covariancia(np.eye(1))


def test_mudanca_de_estrato_exige_tratamento_explicito(amostra):
    desenho = LinearizacaoUPA()
    desenho.registrar(amostra)
    with pytest.raises(ValueError, match="mudou de estrato"):
        desenho.registrar(amostra.iloc[[0]].assign(Estrato=3))


def test_identificadores_nao_podem_ser_truncados(amostra):
    with pytest.raises(ValueError, match="UPA"):
        validar_desenho(amostra.assign(UPA=1.5))


def test_ajuste_upa_unica_reproduz_referencia_publica_do_survey():
    # Caso numérico de cran/survey/tests/lonely.psu.R, sem executar R.
    df = pd.DataFrame({"UF": [35] * 5, "Estrato": [1, 1, 2, 2, 3],
                       "UPA": [1, 2, 3, 4, 5], "V1028": [10.5, 10.5, 20, 20, 15]})
    idade = np.array([30.5, 40.5, 35, 52, 44])
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar")
    desenho.registrar(df)
    desenho.iniciar_scores(1)
    desenho.acumular(df, (df.V1028.to_numpy() * idade)[:, None])
    assert desenho.covariancia(np.eye(1))[0, 0] == pytest.approx(127579.81)
