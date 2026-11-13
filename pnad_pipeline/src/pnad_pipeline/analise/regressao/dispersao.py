"""Linearização do desvio-padrão de resíduos em grupos de composição fixa."""

import numpy as np
from scipy.stats import t


def linearizar_desvios(momentos_upa, soma_x, soma_rx, scores_beta, bread):
    peso, soma_r, soma_r2 = momentos_upa.sum(axis=0).T
    if np.any(peso <= 0):
        raise ValueError("As três faixas precisam de observações elegíveis.")
    media = soma_r / peso
    variancia = soma_r2 / peso - media ** 2
    if np.any(variancia <= 0):
        raise ValueError("As três faixas precisam de desvio-padrão positivo.")
    desvio = np.sqrt(variancia)

    w, r, r2 = momentos_upa.transpose(2, 0, 1)
    contribuicao_direta = (r2 - 2 * media * r + (media ** 2 - variancia) * w) / (2 * peso * desvio)
    derivada_beta = -(soma_rx - media[:, None] * soma_x) / (peso * desvio)[:, None]
    return desvio, contribuicao_direta + scores_beta @ bread @ derivada_beta.T


def contrastar_desvios(desvios, cov, gl):
    critico = t.ppf(.975, gl)
    linhas = []
    for i, nome in ((0, "D1"), (2, "D10")):
        bloco = cov[np.ix_([i, 1], [i, 1])]
        diferenca = desvios[i] - desvios[1]
        gradiente_diferenca = np.array([1., -1.])
        margem = critico * np.sqrt(max(0, gradiente_diferenca @ bloco @ gradiente_diferenca))
        log_razao = np.log(desvios[i] / desvios[1])
        gradiente_log_razao = gradiente_diferenca / desvios[[i, 1]]
        margem_log = critico * np.sqrt(max(0, gradiente_log_razao @ bloco @ gradiente_log_razao))
        linhas.append({
            "grupo": nome, "referencia": "D5 e D6", "desvio_grupo_log": float(desvios[i]),
            "desvio_referencia_log": float(desvios[1]), "diferenca_log": float(diferenca),
            "diferenca_ic_inferior_log": float(diferenca - margem), "diferenca_ic_superior_log": float(diferenca + margem),
            "aumento_pct": float(100 * np.expm1(log_razao)),
            "aumento_ic_inferior_pct": float(100 * np.expm1(log_razao - margem_log)),
            "aumento_ic_superior_pct": float(100 * np.expm1(log_razao + margem_log)),
        })
    return linhas
