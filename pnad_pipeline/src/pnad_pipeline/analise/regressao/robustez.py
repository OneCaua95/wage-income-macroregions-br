"""Contrastes para a análise de sensibilidade do M5 histórico."""

import numpy as np


def contrastes_m5(nomes, xtwx):
    indices = {nome: i for i, nome in enumerate(nomes)}
    medias = xtwx[0] / xtwx[0, 0]
    anos = float(sum(nivel * medias[indices[f"Estudo_{nivel}"]] for nivel in range(1, 17)))
    escolaridade = np.zeros(len(nomes))
    escolaridade[indices["Estudo_12"]] = 1
    escolaridade[indices["Estudo_8"]] = -1
    for nome in ("is_formal", "is_public", "Sexo_Dummy"):
        escolaridade[indices[f"Estudo_x_{nome}"]] = 4 * medias[indices[nome]]
    formalidade, sexo = np.zeros((2, len(nomes)))
    formalidade[indices["is_formal"]] = sexo[indices["Sexo_Dummy"]] = 1
    formalidade[indices["Estudo_x_is_formal"]] = sexo[indices["Estudo_x_Sexo_Dummy"]] = anos
    rotulos = ["Escolaridade · 12 vs. 8 anos", "Formalidade · formal vs. informal", "Sexo · mulheres vs. homens"]
    return rotulos, np.vstack([escolaridade, formalidade, sexo]), anos


def tabela_comparacao(rotulos, log_original, log_limitado, cov, critico, *, prefixos=("original", "limitado")):
    """Resume os contrastes e a diferença pareada na escala percentual."""
    estimativas = np.r_[log_original, log_limitado]
    efeitos = 100 * np.expm1(estimativas)
    margem = critico * np.sqrt(np.maximum(np.diag(cov), 0))
    inferior = 100 * np.expm1(estimativas - margem)
    superior = 100 * np.expm1(estimativas + margem)
    base, alternativo = prefixos
    rows = []
    for i, rotulo in enumerate(rotulos):
        j = i + len(rotulos)
        diferenca = efeitos[j] - efeitos[i]
        gradiente = 100 * np.exp(estimativas[[i, j]]) * [-1, 1]
        variancia = gradiente @ cov[np.ix_([i, j], [i, j])] @ gradiente
        margem_pareada = critico * np.sqrt(max(variancia, 0))
        rows.append({
            "contraste": rotulo,
            f"{base}_pct": float(efeitos[i]), f"{base}_ic_inferior_pct": float(inferior[i]), f"{base}_ic_superior_pct": float(superior[i]),
            f"{alternativo}_pct": float(efeitos[j]), f"{alternativo}_ic_inferior_pct": float(inferior[j]), f"{alternativo}_ic_superior_pct": float(superior[j]),
            "mudanca_pp": float(diferenca), "mudanca_ic_inferior_pp": float(diferenca - margem_pareada),
            "mudanca_ic_superior_pp": float(diferenca + margem_pareada),
        })
    return rows
