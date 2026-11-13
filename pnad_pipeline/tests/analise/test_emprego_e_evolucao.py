"""Referências independentes para os domínios e a agregação anual."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from pnad_pipeline.analise.amostragem import LinearizacaoUPA
from pnad_pipeline.analise.emprego import indicadores_emprego, informalidade_descritiva


def amostra():
    categorias = np.tile([1, 2, 9, 9, 9, 7, 10, 0], 20)
    df = pd.DataFrame({"UF": 35, "Estrato": np.repeat([1, 2], 80), "UPA": np.repeat(range(1, 9), 20),
                       "V1028": np.tile([1., 2., 3., 4., 5.], 32), "Condicao_Ocupacao": 1,
                       "Categoria_emprego": categorias, "Contribuicao_previdencia": np.tile([1, 2, 1, 2, np.nan, 1, 2, 2], 20),
                       "Horas_hab_trabalhadas": np.tile([40., 30., 20., 10., 0., 50., np.nan, 40.], 20),
                       "Rendimento_hab_Trab_princ": np.tile([1000., 2000., 3000., 0., 2000., np.nan, 0., 500.], 20),
                       "Grupo_atv_princ_empreedimento": np.tile([1, 2, 3, 4, 5, 10, 11, np.nan], 20)})
    df.loc[df.index % 11 == 0, "Condicao_Ocupacao"] = 2
    return df


def estimar(df):
    desenho = LinearizacaoUPA(politica_upa_unica="ajustar")
    # Inclui duas UPAs sem pessoas no domínio.
    cadastro = pd.concat([df, pd.DataFrame({"UF": [35, 35], "Estrato": [1, 2], "UPA": [9, 10], "V1028": [1., 1.]})])
    desenho.registrar(cadastro)
    return indicadores_emprego(df, desenho), cadastro, desenho


def test_emprego_ponderado_preserva_dominios_e_composicao():
    df = amostra()
    (row, setores), _, _ = estimar(df)
    ocupado = df.Condicao_Ocupacao.eq(1)
    valido = ocupado & df.Categoria_emprego.isin(range(1, 11)) & (~df.Categoria_emprego.isin([8, 9]) | df.Contribuicao_previdencia.isin([1, 2]))
    informal = valido & (df.Categoria_emprego.isin([2, 4, 6, 10]) | (df.Categoria_emprego.isin([8, 9]) & df.Contribuicao_previdencia.eq(2)))
    assert row["informalidade_estimativa"] == pytest.approx(100 * df.loc[informal, "V1028"].sum() / df.loc[valido, "V1028"].sum())
    assert row["informalidade_n_denominador"] == valido.sum()
    assert row["n_sem_classificacao_formalidade"] == (ocupado & ~valido).sum()
    assert informalidade_descritiva(df)["informalidade_pct"] == pytest.approx(row["informalidade_estimativa"])
    assert sum(s["estimativa"] for s in setores) == pytest.approx(100.)
    assert sum(s["n_ocupados_atividade"] for s in setores) == ocupado.sum()
    assert all(s["n_denominador"] == ocupado.sum() for s in setores)
    assert next(s for s in setores if s["codigo"] == "sem_informacao")["n_ocupados_atividade"] > 0
    # Rendimento não exige horas válidas; jornada não exige renda positiva.
    renda = ocupado & df.Rendimento_hab_Trab_princ.gt(0)
    assert row["rendimento_mensal_estimativa"] == pytest.approx(np.average(df.loc[renda, "Rendimento_hab_Trab_princ"], weights=df.loc[renda, "V1028"]))
    assert row["rendimento_mensal_n_denominador"] == renda.sum()


def test_jornada_variancia_manual_com_upas_fora_do_dominio_e_invariancia():
    df = amostra()
    (r, _), cadastro, desenho = estimar(df)
    v = df.Condicao_Ocupacao.eq(1) & df.Horas_hab_trabalhadas.gt(0)
    media = np.average(df.loc[v, "Horas_hab_trabalhadas"], weights=df.loc[v, "V1028"])
    scores = pd.Series(0., index=df.index)
    scores.loc[v] = df.loc[v, "V1028"] * (df.loc[v, "Horas_hab_trabalhadas"] - media) / df.loc[v, "V1028"].sum()
    agregado = df[["Estrato", "UPA"]].assign(score=scores).groupby(["Estrato", "UPA"]).score.sum()
    chaves = pd.MultiIndex.from_frame(cadastro[["Estrato", "UPA"]].drop_duplicates())
    agregado = agregado.reindex(chaves, fill_value=0.)
    variancia = sum(len(s) / (len(s) - 1) * np.square(s - s.mean()).sum() for _, s in agregado.groupby(level=0))
    esperado = media + t.ppf(.975, desenho.graus_liberdade) * np.sqrt(variancia)
    assert r["jornada_ic_superior"] == pytest.approx(esperado)
    escalada = df.copy()
    escalada.V1028 *= 17
    (s, _), _, _ = estimar(escalada)
    assert s["jornada_ic_superior"] == pytest.approx(r["jornada_ic_superior"])
    assert s["informalidade_estimativa"] == pytest.approx(r["informalidade_estimativa"])


def test_denominador_insuficiente_nao_inventa_taxa():
    df = amostra().iloc[:24]
    row = informalidade_descritiva(df)
    assert row["informalidade_pct"] is None and row["n_formalidade"] < 30
    (r, setores), _, _ = estimar(df)
    assert r["jornada_estimativa"] is None and r["jornada_ic_inferior"] is None
    assert all(s["estimativa"] is None for s in setores)


def test_mascaras_aceitam_tipos_nullable_dos_microdados():
    df = amostra()
    nullable = df.astype({"Condicao_Ocupacao": "Int64", "Categoria_emprego": "Int64", "Contribuicao_previdencia": "Float64"})
    nullable.loc[nullable.Categoria_emprego.eq(0), "Categoria_emprego"] = pd.NA
    assert informalidade_descritiva(nullable) == informalidade_descritiva(df)
    (r, setores), _, _ = estimar(nullable)
    (esperado, _), _, _ = estimar(df)
    assert r["informalidade_estimativa"] == esperado["informalidade_estimativa"]
    assert sum(s["estimativa"] for s in setores) == pytest.approx(100.)


def test_composicao_separa_codigos_8_a_12_do_dicionario_ibge():
    df = amostra()
    df["Grupo_atv_princ_empreedimento"] = np.tile([8, 9, 10, 11, 12], 32)
    (_, setores), _, _ = estimar(df)
    rotulos = {s["codigo"]: s for s in setores}
    assert rotulos["8"]["atividade"] == "Administração Pública, Defesa e Seguridade Social"
    assert rotulos["9"]["atividade"] == "Educação, Saúde e Serviços Sociais"
    assert rotulos["10"]["atividade"] == "Outros Serviços"
    assert rotulos["11"]["atividade"] == "Serviços Domésticos"
    assert rotulos["12"]["atividade"] == "Atividades Mal Definidas"
    assert rotulos["11"]["n_ocupados_atividade"] > 0
    assert rotulos["sem_informacao"]["estimativa"] == 0
    assert sum(s["estimativa"] for s in setores) == pytest.approx(100.)


def carregar_anual():
    spec = importlib.util.spec_from_file_location("evolucao_anual", Path(__file__).resolve().parents[2] / "scripts/44_evolucao_regional_anual.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_mediana_anual_e_da_distribuicao_empilhada_e_crescimento(tmp_path):
    modulo = carregar_anual()
    partes = []
    for ano, fator in [(2013, 1.), (2014, 1.2)]:
        for trimestre, salario, peso in [(1, 10., 8.), (2, 20., 1.), (3, 30., 1.), (4, 40., 1.)]:
            partes.append(pd.DataFrame({"Ano": ano, "Trimestre": trimestre, "UF": 35, "Estrato": 1,
                "UPA": np.arange(1, 41), "V1028": peso, "Idade": 30, "Condicao_Ocupacao": 1,
                "Rendimento_hab_Trab_princ": 1000., "Horas_hab_trabalhadas": 40.,
                "Salario_Hora": salario * fator, "Regiao": "Sudeste"}))
    df = pd.concat(partes, ignore_index=True)
    entrada = tmp_path / "analise.parquet"
    df.to_parquet(entrada, index=False)
    report = modulo.executar(entrada, tmp_path / "saida")
    brasil = [r for r in report["rows"] if r["grupo"] == "Brasil"]
    assert [r["mediana"] for r in brasil] == [10., 12.]
    assert brasil[1]["crescimento_mediana_pct"] == pytest.approx(20.)
    assert brasil[0]["populacao_media_salarios"] == 110.
    assert next(r for r in report["rows"] if r["grupo"] == "Sul")["mediana"] is None
    with pytest.raises(ValueError, match="quatro trimestres"):
        modulo.resumir_ano(df.loc[df.Trimestre.ne(4)], 2013)
