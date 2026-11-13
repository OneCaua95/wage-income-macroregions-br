"""Diagnósticos de colinearidade do desenho ponderado da regressão."""

import numpy as np


def diagnosticar_blocos(xtwx, nomes, indices_ativos, blocos):
    """Calcula o GVIF de cada bloco frente aos demais termos estimáveis."""
    indices = indices_ativos[indices_ativos != 0]
    ativos = [nomes[i] for i in indices]
    omitidos = [nome for nome in nomes[1:] if nome not in ativos]
    centrada = xtwx[np.ix_(indices, indices)] - np.outer(xtwx[indices, 0], xtwx[0, indices]) / xtwx[0, 0]
    escala = np.sqrt(np.diag(centrada))
    correlacao = centrada / np.outer(escala, escala)
    autovalores = np.linalg.eigvalsh(correlacao)
    if autovalores.size and (autovalores[0] <= 0 or not np.isfinite(autovalores).all()):
        raise ValueError("A matriz de correlação dos termos estimáveis deve ser definida positiva.")
    condicao = float(np.sqrt(autovalores[-1] / autovalores[0])) if autovalores.size else None
    _, logdet = np.linalg.slogdet(correlacao)
    resultados = {}

    for rotulo, termos in blocos.items():
        grupo = np.array([nome in termos for nome in ativos], dtype=bool)
        df = int(grupo.sum())
        gvif = ajustado = None
        if df:
            _, log_bloco = np.linalg.slogdet(correlacao[np.ix_(grupo, grupo)])
            _, log_resto = np.linalg.slogdet(correlacao[np.ix_(~grupo, ~grupo)])
            log_gvif = max(0.0, log_bloco + log_resto - logdet)
            gvif = float(np.exp(log_gvif))
            ajustado = float(np.exp(log_gvif / (2 * df)))
        resultados[rotulo] = {
            "n_colunas": sum(nome in termos for nome in nomes[1:]), "n_ativas": df,
            "gvif": gvif, "gvif_ajustado": ajustado,
        }

    return {
        "n_colunas": len(nomes) - 1, "n_ativas": len(ativos),
        "omitidos": omitidos, "numero_condicao": condicao,
    }, resultados
