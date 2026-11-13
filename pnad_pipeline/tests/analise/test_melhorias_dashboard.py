"""Referências independentes para as novas medidas e famílias inferenciais."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from pnad_pipeline.analise.amostragem import LinearizacaoUPA
from pnad_pipeline.analise.regressao.multiplicidade import ajustar_holm, calcular_testes_blocos

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def carregar(nome):
    spec = importlib.util.spec_from_file_location("_melhorias_" + nome, SCRIPTS / (nome + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_holm_preserva_ausentes_e_usa_a_familia_finita():
    np.testing.assert_allclose(ajustar_holm([.001, .04, np.nan, .03]), [.003, .06, np.nan, .06], equal_nan=True)


def test_wald_conjunto_usa_covariancia_e_interacoes():
    m5 = carregar("08_inferencia_m5")
    nomes = ["Intercept", *m5.nomes_features(m5.UFS)]
    beta = np.linspace(-.1, .2, len(nomes))
    cov = np.eye(len(nomes)) * .03
    indices = [nomes.index(n) for n in ("Sexo_Dummy", "Estudo_x_Sexo_Dummy")]
    cov[indices[0], indices[1]] = cov[indices[1], indices[0]] = .01
    rows = calcular_testes_blocos(nomes, beta, cov, np.arange(len(nomes)), 100)
    resultado = next(r for r in rows if r["bloco"] == "Sexo e interação")
    esperado = beta[indices] @ np.linalg.inv(cov[np.ix_(indices, indices)]) @ beta[indices] / 2
    assert resultado["wald_f"] == pytest.approx(esperado)
    assert resultado["gl_numerador"] == 2
    assert resultado["p_holm"] >= resultado["p_valor"]
    omitidos = calcular_testes_blocos(nomes, beta, cov, np.setdiff1d(np.arange(len(nomes)), indices[0]), 100)
    assert next(r for r in omitidos if r["bloco"] == "Sexo e interação")["p_holm"] is None


def test_desigualdade_ponderada_concorda_com_replicacao_e_fracao_do_topo():
    modulo = carregar("19_desigualdade_quantis")
    values, weights = np.array([1., 2., 5.]), np.array([1., 3., 6.])
    row, _ = modulo.metricas_distribuicao(values, weights, ("2025T4", "geral", "Brasil"))
    expanded = np.repeat(values, weights.astype(int))
    relative = expanded / expanded.mean()
    assert row["theil_t"] == pytest.approx(np.mean(relative * np.log(relative)))
    assert row["participacao_top10_pct"] == pytest.approx(100 * 5 / 37)
    assert row["razao_P50_P10"] == 5
    assert row["razao_P90_P50"] == 1
    scaled, _ = modulo.metricas_distribuicao(values * 10, weights * 3, ("2025T4", "geral", "Brasil"))
    assert scaled["theil_t"] == pytest.approx(row["theil_t"])
    assert scaled["participacao_top10_pct"] == pytest.approx(row["participacao_top10_pct"])
    zeros, _ = modulo.metricas_distribuicao(np.zeros(3), weights, ("2025T4", "geral", "Brasil"))
    assert zeros["theil_t"] is None and zeros["participacao_top10_pct"] is None


def test_taxas_do_trabalho_usam_denominadores_e_variancia_de_razoes():
    modulo = carregar("37_mercado_trabalho")
    rng = np.random.default_rng(42)
    condicoes = np.tile([1, 1, 2, 0], 30)
    df = pd.DataFrame({"UF": 35, "Estrato": np.repeat([1, 2], 60),
                       "UPA": np.repeat([11, 12, 21, 22], 30), "V1028": rng.uniform(1, 5, 120),
                       "F_Trabalho": np.where(condicoes > 0, 1, 2),
                       "Condicao_Ocupacao": np.where(condicoes > 0, condicoes, np.nan)})
    desenho = LinearizacaoUPA()
    desenho.registrar(df)
    result = modulo.taxas_grupo(df, desenho)
    w = df.V1028.to_numpy()
    labor = condicoes > 0
    unemployment = condicoes == 2
    taxa = w[unemployment].sum() / w[labor].sum()
    assert result["taxa_desocupacao_pct"] == pytest.approx(100 * taxa)
    assert result["taxa_participacao_pct"] == pytest.approx(100 * w[labor].sum() / w.sum())
    scores = w * (unemployment - labor * taxa) / w[labor].sum()
    manual = df[["Estrato", "UPA"]].assign(score=scores).groupby(["Estrato", "UPA"]).score.sum()
    variance = sum(2 * np.square(sub - sub.mean()).sum() for _, sub in manual.groupby(level=0))
    expected = 100 * max(0, taxa - t.ppf(.975, 2) * np.sqrt(variance))
    assert result["desocupacao_ic_inferior_pct"] == pytest.approx(expected)


def test_oaxaca_regional_retira_ufs_que_identificam_o_grupo():
    modulo = carregar("21_decomposicao_oaxaca")
    modelos = modulo.preparar_modelos()
    regionais = [m for m in modelos if m["coluna"] == "Regiao"]
    assert len(regionais) == 4
    for modelo in regionais:
        assert modelo["referencia"] == "Sudeste"
        assert not any(n.startswith("UF_") for n in modelo["nomes"])
        assert "Cor_Preta" in modelo["nomes"] and "Sexo_Dummy" in modelo["nomes"]
    assert any(n.startswith("UF_") for n in modelos[0]["nomes"])
