"""Identidades distributivas e variâncias conferidas por álgebra independente."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from pnad_pipeline.analise.amostragem import LinearizacaoUPA
from pnad_pipeline.analise.incerteza import intervalos_salario
from pnad_pipeline.analise.theil import decompor_theil


def desenho_e_amostra():
    rng = np.random.default_rng(38)
    df = pd.DataFrame({"UF": 35, "Estrato": np.repeat([1, 2], 60),
                       "UPA": np.repeat([11, 12, 21, 22], 30),
                       "V1028": rng.integers(1, 5, 120).astype(float),
                       "Salario_Hora": rng.integers(1, 15, 120).astype(float)})
    desenho = LinearizacaoUPA()
    desenho.registrar(df)
    return desenho, df


def variancia_manual(df, scores):
    totais = df[["Estrato", "UPA"]].assign(score=scores).groupby(["Estrato", "UPA"]).score.sum()
    return sum(2 * np.square(grupo - grupo.mean()).sum() for _, grupo in totais.groupby(level=0))


def test_intervalos_da_media_e_woodruff_conferem_com_calculo_manual():
    desenho, df = desenho_e_amostra()
    resultado = intervalos_salario(df, desenho)
    w, y = df.V1028.to_numpy(), df.Salario_Hora.to_numpy()
    media = np.average(y, weights=w)
    ep = np.sqrt(variancia_manual(df, w * (y - media) / w.sum()))
    assert resultado[0]["erro_padrao"] == pytest.approx(ep)
    assert resultado[0]["ic_superior"] == pytest.approx(media + t.ppf(.975, 2) * ep)
    expanded = np.sort(np.repeat(y, w.astype(int)))
    for row, prob in zip(resultado[1:], [.1, .5, .9]):
        quantil = expanded[int(np.ceil(prob * len(expanded))) - 1]
        assert row["estimativa"] == quantil
        indicadores = y <= quantil
        cdf = np.average(indicadores, weights=w)
        ep_cdf = np.sqrt(variancia_manual(df, w * (indicadores - cdf) / w.sum()))
        probs = cdf + t.ppf(.975, 2) * ep_cdf * np.array([-1, 1])
        for campo, p in zip(["ic_inferior", "ic_superior"], probs):
            if 0 <= p <= 1:
                indice = max(int(np.ceil(p * len(expanded))) - 1, 0)
                assert row[campo] == expanded[indice]
            else:
                assert row[campo] is None


def test_intervalos_preservam_empates_escala_dos_pesos_e_dominio_pequeno():
    desenho, df = desenho_e_amostra()
    base = intervalos_salario(df, desenho)
    df.V1028 *= 11
    scaled = intervalos_salario(df, desenho)
    for a, b in zip(base, scaled):
        assert b["estimativa"] == pytest.approx(a["estimativa"])
        assert b["ic_inferior"] == pytest.approx(a["ic_inferior"])
        assert b["ic_superior"] == pytest.approx(a["ic_superior"])
    df.Salario_Hora = 5.
    iguais = intervalos_salario(df, desenho)
    assert all(r["estimativa"] == r["ic_inferior"] == r["ic_superior"] == 5 for r in iguais)
    assert all(r["status"] == "amostra_insuficiente" for r in intervalos_salario(df.iloc[:10], desenho))


def test_theil_decomposto_reproduz_distribuicao_ponderada_e_contribuicoes():
    x = np.array([1., 2., 3., 8.])
    w = np.array([1., 2., 3., 1.])
    regiao = np.array(["A", "A", "B", "B"])
    totais = pd.DataFrame({"regiao": regiao, "n_salarios": 1, "peso_v1028": w,
                           "soma_salario_ponderado": w * x,
                           "soma_salario_log_ponderado": w * x * np.log(x)}).groupby("regiao").sum().reset_index()
    summary, rows = decompor_theil(totais)
    mean = np.average(x, weights=w)
    expected = np.average((x / mean) * np.log(x / mean), weights=w)
    assert summary["theil_total"] == pytest.approx(expected)
    assert summary["theil_dentro"] + summary["theil_entre"] == pytest.approx(expected)
    assert sum(r["contribuicao_dentro"] for r in rows) == pytest.approx(summary["theil_dentro"])
    assert sum(r["contribuicao_entre"] for r in rows) == pytest.approx(summary["theil_entre"])
    assert rows[0]["contribuicao_entre"] < 0
    # Uma região única preserva toda a desigualdade na parcela interna.
    unica = totais.drop(columns="regiao").sum().to_frame().T.assign(regiao="Brasil")
    single, _ = decompor_theil(unica)
    assert single["theil_dentro"] == pytest.approx(expected)
    assert single["theil_entre"] == pytest.approx(0, abs=1e-12)


def test_media_da_coorte_tem_ep_do_desenho_trimestral():
    path = Path(__file__).resolve().parents[2] / "scripts/39_sensibilidade_coortes.py"
    spec = importlib.util.spec_from_file_location("_teste_coortes_incerteza", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    desenho, df = desenho_e_amostra()
    df = df.assign(Ano_Nascimento=1970, Ano=2025, Idade=55, Sexo=1,
                   y=np.log(df.Salario_Hora), Anos_de_Estudo=np.arange(len(df)) % 17)
    w = df.V1028.to_numpy()
    media_y = np.average(df.y, weights=w)
    cells = pd.DataFrame({"coorte_id": [100], "w": [w.sum()], "media_log_salario_hora": [media_y],
                          "media_anos_estudo": [np.average(df.Anos_de_Estudo, weights=w)]})
    resultado = module.incerteza_celulas(df, cells, [1946, 2003], desenho).iloc[0]
    ep = np.sqrt(variancia_manual(df, w * (df.y.to_numpy() - media_y) / w.sum()))
    assert resultado.erro_padrao_media_log == pytest.approx(ep)
    assert resultado.ic95_superior_media_geometrica == pytest.approx(np.exp(media_y + t.ppf(.975, 2) * ep))
