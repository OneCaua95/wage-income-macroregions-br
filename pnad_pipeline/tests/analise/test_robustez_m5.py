import numpy as np
import pytest

from pnad_pipeline.analise.regressao.robustez import contrastes_m5, tabela_comparacao


def test_contrastes_m5_incluem_as_interacoes_na_media_ponderada():
    interacoes = ["is_formal", "is_public", "Sexo_Dummy"]
    nomes = ["Intercepto", *interacoes, *(f"Estudo_{i}" for i in range(1, 17)),
             *(f"Estudo_x_{nome}" for nome in interacoes)]
    xtwx = np.zeros((len(nomes), len(nomes)))
    xtwx[0, :4] = [100, 60, 20, 40]
    xtwx[0, nomes.index("Estudo_8")] = 25
    xtwx[0, nomes.index("Estudo_12")] = 75

    _, matriz, anos = contrastes_m5(nomes, xtwx)

    assert anos == 11
    assert [dict(zip(np.array(nomes)[linha != 0], linha[linha != 0])) for linha in matriz] == [
        {"Estudo_8": -1, "Estudo_12": 1, "Estudo_x_is_formal": 2.4,
         "Estudo_x_is_public": .8, "Estudo_x_Sexo_Dummy": 1.6},
        {"is_formal": 1, "Estudo_x_is_formal": 11},
        {"Sexo_Dummy": 1, "Estudo_x_Sexo_Dummy": 11},
    ]


def test_ic_da_mudanca_usa_covariancia_pareada_na_escala_percentual():
    cov = np.array([[.01, .012], [.012, .04]])
    original, limitado = 1.1, 1.2
    critico = 1.96
    linha, = tabela_comparacao(["contraste"], [np.log(original)], [np.log(limitado)], cov, critico)

    variancia = 110**2 * .01 + 120**2 * .04 - 2 * 110 * 120 * .012
    margem = critico * np.sqrt(variancia)
    assert linha["original_pct"] == pytest.approx(10)
    assert linha["limitado_pct"] == pytest.approx(20)
    assert linha["mudanca_pp"] == pytest.approx(10)
    assert linha["mudanca_ic_inferior_pp"] == pytest.approx(10 - margem)
    assert linha["mudanca_ic_superior_pp"] == pytest.approx(10 + margem)
