"""Decomposição exata do Theil T ponderado a partir de totais regionais."""

import numpy as np


def decompor_theil(totais):
    peso = totais.peso_v1028.to_numpy(dtype=float)
    renda = totais.soma_salario_ponderado.to_numpy(dtype=float)
    renda_log = totais.soma_salario_log_ponderado.to_numpy(dtype=float)
    if not np.isfinite(np.column_stack((peso, renda, renda_log))).all() or np.any(peso <= 0) or np.any(renda <= 0):
        raise ValueError("Os totais do Theil devem ser finitos, com pesos e salários agregados positivos.")
    media = renda.sum() / peso.sum()
    medias = renda / peso
    participacoes = renda / renda.sum()
    theil_regional = renda_log / renda - np.log(medias)
    if np.any(theil_regional < -1e-10):
        raise ValueError("Theil regional negativo. Confira os totais usados na decomposição.")
    theil_regional = np.maximum(theil_regional, 0)
    dentro = participacoes * theil_regional
    entre = participacoes * np.log(medias / media)
    total = float(renda_log.sum() / renda.sum() - np.log(media))
    if not np.isclose(total, dentro.sum() + entre.sum(), atol=1e-10):
        raise ValueError("A identidade aditiva do Theil não fechou.")
    resumo = {"theil_total": max(total, 0), "theil_dentro": float(dentro.sum()),
              "theil_entre": float(entre.sum()),
              "parcela_entre_pct": float(100 * entre.sum() / total) if total > 1e-12 else None}
    linhas = []
    for i, regiao in enumerate(totais.regiao):
        linhas.append({"regiao": regiao, "n_salarios": int(totais.iloc[i].n_salarios),
                       "media_salario_hora": float(medias[i]), "peso_v1028": float(peso[i]),
                       "participacao_populacao_pct": float(100 * peso[i] / peso.sum()),
                       "participacao_salarios_pct": float(100 * participacoes[i]),
                       "theil_regional": float(theil_regional[i]),
                       "contribuicao_dentro": float(dentro[i]), "contribuicao_entre": float(entre[i])})
    return resumo, linhas
