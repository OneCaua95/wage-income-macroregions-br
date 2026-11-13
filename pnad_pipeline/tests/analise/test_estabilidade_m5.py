import numpy as np
import pytest
from scipy.stats import f, t

from pnad_pipeline.analise.regressao.estabilidade import resumir_estabilidade


def test_wald_anual_incorpora_covariancia_entre_anos_do_mesmo_contraste():
    anos = [2023, 2024, 2025]
    estimativas = np.array([[.10, -.15], [.14, -.15], [.18, -.15]])
    cov_escolaridade = np.array([[.0005, .0001, .0002], [.0001, .0004, .0001], [.0002, .0001, .0006]])
    cov = np.diag([.0005, .0003, .0004, .0002, .0006, .0004])
    cov[::2, ::2] = cov_escolaridade

    linhas, (escolaridade, sexo) = resumir_estabilidade(
        anos, ["Escolaridade", "Sexo"], estimativas, cov, 100, dict.fromkeys(anos, 300)
    )

    # Calcula as mesmas restrições com o último ano como referência.
    restricoes = np.array([[1, 0, -1], [0, 1, -1]])
    diferencas = restricoes @ estimativas[:, 0]
    cov_diferencas = restricoes @ cov_escolaridade @ restricoes.T
    f_esperado = diferencas @ np.linalg.inv(cov_diferencas) @ diferencas / 2
    assert escolaridade["wald_f"] == pytest.approx(f_esperado)
    assert escolaridade["p_holm"] == pytest.approx(2 * f.sf(f_esperado, 2, 100))
    assert escolaridade["rejeita_estabilidade"]
    assert sexo["wald_f"] == 0
    assert sexo["p_holm"] == 1
    assert not sexo["rejeita_estabilidade"]
    assert len(linhas) == 6
    margem = t.ppf(.975, 100) * np.sqrt(.0005)
    assert linhas[0]["ic_inferior_pct"] == pytest.approx(100 * np.expm1(.10 - margem))
