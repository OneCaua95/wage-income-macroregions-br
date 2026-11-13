"""Inferência sobre contrastes padronizados do M5 ao longo dos anos."""

import numpy as np
from scipy.linalg import solve
from scipy.stats import f, t
from statsmodels.stats.multitest import multipletests


def resumir_estabilidade(anos, rotulos, estimativas, cov, gl, contagens):
    """Usa a covariância conjunta, em ordem de ano e depois de contraste."""
    margem = t.ppf(.975, gl) * np.sqrt(np.diag(cov)).reshape(estimativas.shape)
    percentuais = 100 * np.expm1(estimativas)
    inferiores = 100 * np.expm1(estimativas - margem)
    superiores = 100 * np.expm1(estimativas + margem)
    linhas, testes = [], []
    for j, rotulo in enumerate(rotulos):
        for i, ano in enumerate(anos):
            linhas.append({
                "ano": ano, "contraste": rotulo, "n_observacoes": contagens[ano],
                "estimativa_pct": float(percentuais[i, j]),
                "ic_inferior_pct": float(inferiores[i, j]),
                "ic_superior_pct": float(superiores[i, j]),
            })

        indices = np.arange(j, cov.shape[0], len(rotulos))
        anual = cov[np.ix_(indices, indices)]
        # Igualdade de todos os anos equivale a diferenças nulas contra o primeiro.
        diferencas = estimativas[1:, j] - estimativas[0, j]
        cov_diferencas = anual[1:, 1:] - anual[:1, 1:] - anual[1:, :1] + anual[0, 0]
        restricoes = len(diferencas)
        estatistica = float(diferencas @ solve(cov_diferencas, diferencas, assume_a="pos") / restricoes)
        testes.append({
            "contraste": rotulo, "wald_f": estatistica,
            "gl_numerador": restricoes, "gl_denominador": gl,
            "p_valor": float(f.sf(estatistica, restricoes, gl)),
        })

    rejeicoes, p_ajustados, _, _ = multipletests([teste["p_valor"] for teste in testes], method="holm")
    for teste, rejeita, p_ajustado in zip(testes, rejeicoes, p_ajustados):
        teste["p_holm"] = float(p_ajustado)
        teste["rejeita_estabilidade"] = bool(rejeita)
    return linhas, testes
