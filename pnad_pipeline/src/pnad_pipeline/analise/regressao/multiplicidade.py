"""Correção de Holm e testes conjuntos com a covariância amostral do M5."""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def ajustar_holm(p_valores):
    p = np.asarray(p_valores, dtype=float)
    ajustados = np.full(p.shape, np.nan)
    validos = np.isfinite(p)
    if validos.any():
        ajustados[validos] = multipletests(p[validos], method="holm")[1]
    return ajustados


def calcular_testes_blocos(nomes, beta, cov, ativos, gl):
    blocos = {
        "Escolaridade e interações": [n for n in nomes if n.startswith("Estudo_")],
        "Formalidade e interação": ["is_formal", "Estudo_x_is_formal"],
        "Setor público e interação": ["is_public", "Estudo_x_is_public"],
        "Sexo e interação": ["Sexo_Dummy", "Estudo_x_Sexo_Dummy"],
        "Cor/raça": [n for n in nomes if n.startswith("Cor_")],
        "Idade e idade ao quadrado": ["Idade_centrada", "Idade_centrada_2"],
        "Atividade econômica": [n for n in nomes if n.startswith("Setor_")],
        "Ocupação": [n for n in nomes if n.startswith("Ocup_")],
        "UF": [n for n in nomes if n.startswith("UF_")],
        "Ano e trimestre": [n for n in nomes if n.startswith(("Ano_", "Trimestre_"))],
    }
    rows = []
    for rotulo, termos in blocos.items():
        indices = [nomes.index(nome) for nome in termos]
        if not indices or not np.isin(indices, ativos).all():
            rows.append({"bloco": rotulo, "status": "nao_identificado", "n_termos": len(indices), "p_valor": None})
            continue
        bloco = cov[np.ix_(indices, indices)]
        escala = np.sqrt(np.diag(bloco))
        if not np.all(escala > 0):
            raise ValueError(f"Covariância inválida para o bloco {rotulo}.")
        correlacao = bloco / np.outer(escala, escala)
        posto = int(np.linalg.matrix_rank(correlacao, tol=1e-10))
        if posto != len(indices):
            rows.append({"bloco": rotulo, "status": "covariancia_singular", "n_termos": len(indices), "p_valor": None})
            continue
        padronizados = beta[indices] / escala
        f = max(float(padronizados @ np.linalg.solve(correlacao, padronizados) / posto), 0)
        rows.append({"bloco": rotulo, "status": "ok", "n_termos": len(indices),
                     "wald_f": f, "gl_numerador": posto, "gl_denominador": gl,
                     "p_valor": float(stats.f.sf(f, posto, gl))})
    ajustados = ajustar_holm([r["p_valor"] for r in rows])
    for row, p in zip(rows, ajustados):
        row["p_holm"] = float(p) if np.isfinite(p) else None
    return rows
