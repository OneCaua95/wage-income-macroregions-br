"""Intervalos pontuais da média e de quantis sob o desenho amostral."""

import numpy as np
from scipy.stats import t

from pnad_pipeline.analise.amostragem import pesos_amostrais, quantis_ponderados


def intervalos_salario(df, desenho):
    metricas = [("media", None), ("P10", .1), ("P50", .5), ("P90", .9)]
    contexto = {"n_salarios": len(df), "graus_liberdade": desenho.graus_liberdade}
    if len(df) < 30:
        return [{**contexto, "indicador": nome, "estimativa": None, "ic_inferior": None,
                 "ic_superior": None, "status": "amostra_insuficiente"} for nome, _ in metricas]
    w = pesos_amostrais(df)
    y = df.Salario_Hora.to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("A amostra salarial deve conter apenas salários-hora positivos e finitos.")
    media = float(w @ y / w.sum())
    quantis = quantis_ponderados(df, "Salario_Hora", [.1, .5, .9])
    indicadores = y[:, None] <= quantis
    proporcoes = w @ indicadores / w.sum()
    scores = np.column_stack((w * (y - media), w[:, None] * (indicadores - proporcoes))) / w.sum()
    desenho.iniciar_scores(4)
    desenho.acumular(df, scores)
    erros = np.sqrt(np.maximum(np.diag(desenho.covariancia(np.eye(4))), 0))
    critico = t.ppf(.975, desenho.graus_liberdade)
    rows = [{**contexto, "indicador": "media", "estimativa": media, "erro_padrao": float(erros[0]),
             "ic_inferior": float(media - critico * erros[0]), "ic_superior": float(media + critico * erros[0]),
             "metodo": "Linearização da razão e intervalo t", "status": "ok"}]
    for i, (nome, _) in enumerate(metricas[1:]):
        probabilidades = proporcoes[i] + critico * erros[i + 1] * np.array([-1, 1])
        limites = [float(quantis_ponderados(df, "Salario_Hora", [p])[0]) if 0 <= p <= 1 else None
                   for p in probabilidades]
        rows.append({**contexto, "indicador": nome, "estimativa": float(quantis[i]),
                     "ic_inferior": limites[0], "ic_superior": limites[1],
                     "probabilidade_cdf": float(proporcoes[i]), "erro_padrao_cdf": float(erros[i + 1]),
                     "metodo": "Woodruff com intervalo t da CDF e inversa empírica ponderada",
                     "status": "ok" if all(v is not None for v in limites) else "limites_indisponiveis"})
    return rows
