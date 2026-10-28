import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[2] / "scripts" / "08_inferencia_m5.py"
SPEC = importlib.util.spec_from_file_location("inferencia_m5_teste", PATH)
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)


def ajustar(escala=1, lote=3):
    df = pd.DataFrame({
        "UF": [35] * 8, "Estrato": [1] * 4 + [2] * 4,
        "UPA": [11, 11, 12, 12, 21, 21, 22, 22],
        "V1028": np.array([1, 2, 7, 3, 2, 9, 4, 5], dtype=float) * escala,
        "uf_id": [35] * 8, "coorte_id": [1, 2] * 4, "inter_id": [135, 235] * 4,
    })
    x = np.column_stack([np.ones(8), np.arange(8, dtype=float)])
    y = np.array([1, 1.8, 2.7, 3.2, 4.1, 4.3, 5.7, 7.0])
    modelo = M5.ResultadoModelo("teste", [35])
    for inicio in range(0, len(df), lote):
        sub = df.iloc[inicio:inicio + lote]
        modelo.desenho.registrar(sub)
        modelo.registrar_grupos(sub)
        modelo.acumular_normal(x[inicio:inicio + lote], y[inicio:inicio + lote], sub.V1028.to_numpy())
    modelo.resolver()
    for inicio in range(0, len(df), lote):
        modelo.acumular_sanduiche(x[inicio:inicio + lote], y[inicio:inicio + lote], df.iloc[inicio:inicio + lote])
    return modelo, df, x, y


def test_regressao_e_covariancia_concordam_com_algebra_independente():
    modelo, df, x, y = ajustar()
    w = df.V1028.to_numpy()
    bread = np.linalg.inv(x.T @ np.diag(w) @ x)
    beta = bread @ x.T @ np.diag(w) @ y
    np.testing.assert_allclose(modelo.beta, beta, rtol=1e-10)
    # Calcula separadamente as somas de scores de cada UPA e estrato.
    scores = x * (w * (y - x @ beta))[:, None]
    meat = np.zeros((2, 2))
    for _, estrato in df.groupby("Estrato"):
        totais = np.stack([scores[grupo.index].sum(axis=0) for _, grupo in estrato.groupby("UPA")])
        centrados = totais - totais.mean(axis=0)
        meat += 2 * centrados.T @ centrados
    np.testing.assert_allclose(M5.covariancias(modelo)["desenho_pnad"], bread @ meat @ bread, rtol=1e-9)


def test_escala_do_peso_e_tamanho_do_lote_nao_alteram_inferencia():
    a, *_ = ajustar(escala=1, lote=3)
    b, *_ = ajustar(escala=1000, lote=5)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    np.testing.assert_allclose(M5.covariancias(a)["desenho_pnad"], M5.covariancias(b)["desenho_pnad"], rtol=1e-9)


def test_base_local_antiga_falha_com_orientacao_antes_de_estimar(tmp_path):
    caminho = tmp_path / "antiga.parquet"
    pd.DataFrame({"UPA": [1]}).to_parquet(caminho)
    with pytest.raises(ValueError, match="01_coletar_pnad"):
        list(M5.iterar_lotes(caminho, 10))


def test_termos_constantes_de_uma_onda_sao_marcados_como_nao_identificados():
    x = np.column_stack([np.ones(6), np.arange(6), np.ones(6), np.zeros(6)])
    y = np.array([2.1, 3, 4.2, 4.9, 6.2, 7])
    modelo = M5.ResultadoModelo("uma_onda", [35])
    modelo.acumular_normal(x, y, np.ones(6))
    modelo.resolver()
    np.testing.assert_array_equal(modelo.indices_ativos, [0, 1])
    esperado = np.linalg.lstsq(x[:, :2], y, rcond=None)[0]
    np.testing.assert_allclose(modelo.beta[:2], esperado)
    np.testing.assert_array_equal(modelo.beta[2:], [0, 0])


def test_fluxo_completo_ponderado_em_lotes(tmp_path):
    rng = np.random.default_rng(2026)
    linhas = []
    for macro, ufs in M5.ESTADOS_POR_MACRO.items():
        for uf in ufs:
            for upa in range(4):
                for i in range(40):
                    ano = int(rng.integers(2013, 2026))
                    idade = int(rng.integers(25, 66))
                    estudo = int(rng.integers(0, 17))
                    sexo = int(rng.integers(1, 3))
                    linhas.append({
                        "Ano": ano, "Trimestre": int(rng.integers(1, 5)),
                        "Ano_Nascimento": ano - idade, "Idade": idade, "UF": uf,
                        "Sexo": sexo, "Cor": int(rng.integers(1, 6)),
                        "Anos_de_Estudo": estudo, "Categoria_emprego": int(rng.integers(1, 11)),
                        "Contribuicao_previdencia": int(rng.integers(1, 3)),
                        "Grupo_atv_princ_empreedimento": int(rng.integers(1, 11)),
                        "Grupo_ocupacional_no_emprego": int(rng.integers(1, 11)),
                        "Condicao_Ocupacao": 2 if i % 19 == 0 else 1,
                        "Rendimento_hab_Trab_princ": float(np.exp(2 + .08 * estudo + .1 * sexo + rng.normal(0, .25)) * 40 * 4.33),
                        "Horas_hab_trabalhadas": 40.0, "Regiao": macro,
                        "V1028": float(rng.uniform(100, 1500)),
                        "UPA": uf * 10000000 + upa + 1, "Estrato": uf * 100000 + upa // 2,
                    })
    df = pd.DataFrame(linhas)
    # Uma UPA inteira sem pessoas elegíveis continua no desenho.
    df.loc[df.UPA.eq(df.UPA.iloc[0]), "Idade"] = 20
    entrada = tmp_path / "completa.parquet"
    df.to_parquet(entrada, index=False)

    spec = importlib.util.spec_from_file_location("tratar_pnad_teste", PATH.parent / "02_tratar_pnad.py")
    tratamento = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tratamento)
    args = Namespace(entrada=entrada, saida=tmp_path / "analitica.parquet", min_idade=25, max_idade=65, batch_size=333)
    tratamento.tratar_base(args)
    tratamento.gerar_tabelas(args.saida, tmp_path / "tabelas")
    tratada = pd.read_parquet(args.saida)
    assert {"UPA", "Estrato", "V1028"}.issubset(tratada.columns)
    assert df.UPA.iloc[0] not in tratada.UPA.to_numpy()
    assert (tmp_path / "tabelas" / "representatividade_trimestral.csv").exists()
    with pytest.raises(ValueError, match="consolidada completa"):
        list(M5.iterar_lotes(args.saida, 100))

    saida = tmp_path / "inferencial"
    assert M5.executar(Namespace(entrada=entrada, saida_dir=saida, batch_size=333, nivel_confianca=.90)) == 0
    resumo = json.loads((saida / "resumo_inferencia_m5.json").read_text(encoding="utf-8"))
    assert resumo["ponderado"] is True
    assert resumo["modelos"][0]["n_upas_desenho"] == 108
    assert resumo["modelos"][0]["graus_liberdade_desenho"] == 54
    tabela = pd.read_csv(saida / "inferencial_m5_geral.csv")
    assert tabela.metodo_principal.eq("desenho_pnad").all()
    assert tabela.erro_padrao_desenho_pnad.notna().all()
    assert tabela.graus_liberdade_desenho_pnad.eq(54).all()
    comparacao = pd.read_csv(saida / "comparacao_especificacoes_inferenciais.csv")
    assert comparacao.loc[comparacao.criterio_principal, "estimador_variancia"].eq("desenho_pnad").all()
