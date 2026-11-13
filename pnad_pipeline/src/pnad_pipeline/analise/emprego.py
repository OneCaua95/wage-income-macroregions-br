"""Indicadores do emprego com domínios explícitos e linearização de razões."""

import numpy as np
from scipy.stats import t

from .amostragem import pesos_amostrais
from .descritiva import ATIVIDADES_MAP

CRITERIO_INFORMALIDADE = (
    "Critério do projeto: carteira/categoria do emprego e contribuição previdenciária "
    "para empregadores e conta própria. Não equivale à taxa oficial do IBGE baseada em CNPJ."
)


def dominios_emprego(df):
    ocupado = df.Condicao_Ocupacao.eq(1).to_numpy(dtype=bool, na_value=False)
    categoria = df.Categoria_emprego
    classificado = ocupado & categoria.isin(range(1, 11)).to_numpy(dtype=bool, na_value=False)
    formal_valido = classificado & (
        ~categoria.isin([8, 9]) | df.Contribuicao_previdencia.isin([1, 2])
    ).to_numpy(dtype=bool, na_value=False)
    formal = (categoria.isin([1, 3, 5, 7]) |
              (categoria.isin([8, 9]) & df.Contribuicao_previdencia.eq(1))).to_numpy(dtype=bool, na_value=False)
    return ocupado, classificado, formal_valido, formal


def informalidade_descritiva(df, minimo=30):
    """Também usada no mapa, inclusive para ocupados sem salário observável."""
    _, _, valido, formal = dominios_emprego(df)
    w = pesos_amostrais(df) if "V1028" in df else np.ones(len(df))
    total = float(w[valido].sum())
    n = int(valido.sum())
    return {"informalidade_pct": float(100 * w[valido & ~formal].sum() / total)
            if n >= minimo and total > 0 else None,
            "n_formalidade": n, "denominador_formalidade": total,
            "n_sem_classificacao_formalidade": int((df.Condicao_Ocupacao.eq(1).to_numpy(dtype=bool, na_value=False) & ~valido).sum())}


def indicadores_emprego(df, desenho):
    ocupado, classificado, valido, formal = dominios_emprego(df)
    categoria = df.Categoria_emprego
    horas = df.Horas_hab_trabalhadas.to_numpy(dtype=float, na_value=np.nan)
    renda = df.Rendimento_hab_Trab_princ.to_numpy(dtype=float, na_value=np.nan)
    horas_validas = ocupado & np.isfinite(horas) & (horas > 0)
    renda_valida = ocupado & np.isfinite(renda) & (renda > 0)
    atividades = df.Grupo_atv_princ_empreedimento
    # Ausentes e códigos não reconhecidos são uma categoria explícita da composição.
    setores = [(str(k), v, ocupado & atividades.eq(k).to_numpy(dtype=bool, na_value=False)) for k, v in ATIVIDADES_MAP.items()]
    setores.append(("sem_informacao", "Sem informação", ocupado & ~atividades.isin(ATIVIDADES_MAP).to_numpy(dtype=bool, na_value=False)))
    especificacoes = [
        ("informalidade", valido & ~formal, valido, 100.),
        ("conta_propria", classificado & categoria.eq(9).to_numpy(dtype=bool, na_value=False), classificado, 100.),
        ("empregados_formais", classificado & categoria.isin([1, 3, 5, 7]).to_numpy(dtype=bool, na_value=False), classificado, 100.),
        ("setor_publico", classificado & categoria.isin([5, 6, 7]).to_numpy(dtype=bool, na_value=False), classificado, 100.),
        ("jornada", np.where(horas_validas, horas, 0.), horas_validas, 1.),
        ("rendimento_mensal", np.where(renda_valida, renda, 0.), renda_valida, 1.),
        *[("setor_" + k, mask, ocupado, 100.) for k, _, mask in setores],
    ]
    w = pesos_amostrais(df)
    numeradores = np.column_stack([s[1] for s in especificacoes]).astype(float)
    denominadores = np.column_stack([s[2] for s in especificacoes]).astype(float)
    totais = w @ denominadores
    estimativas = np.divide(w @ numeradores, totais, out=np.zeros(len(totais)), where=totais > 0)
    scores = np.divide(w[:, None] * (numeradores - denominadores * estimativas), totais,
                       out=np.zeros_like(numeradores), where=totais > 0)
    desenho.iniciar_scores(len(especificacoes))
    desenho.acumular(df, scores)
    erros = np.sqrt(np.maximum(np.diag(desenho.covariancia(np.eye(len(especificacoes)))), 0))
    critico = t.ppf(.975, desenho.graus_liberdade)
    saida, composicao = {}, []
    for i, (nome, _, denom, escala) in enumerate(especificacoes):
        n = int(denom.sum())
        suficiente = n >= 30 and totais[i] > 0
        inferior = max(0., estimativas[i] - critico * erros[i])
        superior = estimativas[i] + critico * erros[i]
        if escala == 100.:
            superior = min(1., superior)
        item = {"estimativa": float(escala * estimativas[i]) if suficiente else None,
                "ic_inferior": float(escala * inferior) if suficiente else None,
                "ic_superior": float(escala * superior) if suficiente else None,
                "n_denominador": n, "populacao_denominador": float(totais[i]),
                "status": "ok" if suficiente else "insufficient"}
        if nome.startswith("setor_") and nome != "setor_publico":
            codigo, rotulo, mask = setores[i - 6]
            composicao.append({"codigo": codigo, "atividade": rotulo, "n_ocupados_atividade": int(mask.sum()),
                               "populacao_atividade": float(w[mask].sum()), **item})
        else:
            saida.update({nome + "_" + k: v for k, v in item.items()})
    saida["n_sem_classificacao_formalidade"] = int((ocupado & ~valido).sum())
    saida["n_sem_categoria_emprego"] = int((ocupado & ~classificado).sum())
    return saida, composicao
