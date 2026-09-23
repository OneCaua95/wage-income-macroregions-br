#!/usr/bin/env python3
"""Executa as análises complementares do modelo regional de salário-hora.

Saídas principais:

* regressões Ridge separadas por macroregião;
* comparação do modelo agrupado com as regressões separadas;
* teste conjunto das interações regionais;
* estabilidade dos coeficientes entre macroregiões;
* decomposição Oaxaca-Blinder para gênero e cor;
* correção de seleção de Heckman em dois passos, como sensibilidade;
* registro explícito da ausência de pesos amostrais e de instrumento válido.

As estimativas são associações condicionais e não substituem uma estratégia
causal. Os resultados não são ponderados porque a base consolidada não contém
uma variável de peso da PNAD.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_SCRIPT = SCRIPT_DIR / "06_modelar_m5_hierarquico_ridge.py"
SPEC = importlib.util.spec_from_file_location("m5_hierarquico", MODEL_SCRIPT)
M5 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(M5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=PROJECT_ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "modelos_salario_hora")
    parser.add_argument("--max-treino", type=int, default=150_000)
    parser.add_argument("--max-avaliacao", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[100.0, 300.0, 1000.0, 3000.0])
    parser.add_argument("--max-selecao-heckman", type=int, default=300_000)
    return parser.parse_args()


def metricas(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    yr, pr = np.exp(y), np.exp(np.clip(p, -20, 20))
    return {
        "mae_log": float(np.mean(np.abs(y - p))),
        "rmse_log": float(np.sqrt(np.mean((y - p) ** 2))),
        "r2_log": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "mae_salario_hora": float(np.mean(np.abs(yr - pr))),
        "rmse_salario_hora": float(np.sqrt(np.mean((yr - pr) ** 2))),
        "r2_salario_hora": float(1 - np.sum((yr - pr) ** 2) / np.sum((yr - yr.mean()) ** 2)),
    }


def ajustar_ridge(x_treino: pd.DataFrame, y_treino: pd.Series, alpha: float):
    scaler = StandardScaler().fit(x_treino)
    modelo = Ridge(alpha=alpha).fit(scaler.transform(x_treino), y_treino)
    beta = modelo.coef_ / scaler.scale_
    intercepto = modelo.intercept_ - np.sum(modelo.coef_ * scaler.mean_ / scaler.scale_)
    return modelo, scaler, beta, float(intercepto)


def avaliar(modelo, scaler, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return metricas(y, modelo.predict(scaler.transform(x)))


def preparar_amostras(args: argparse.Namespace):
    return M5.carregar(args.entrada, args)


def construir_matrizes(amostras: dict[str, pd.DataFrame]):
    estados_por_macro = {
        macro: sorted(amostras["treino"].loc[amostras["treino"]["Regiao"].eq(macro), "UF"].unique().tolist())
        for macro in M5.MACROREGIOES
    }
    matrizes = {
        nome: M5.criar_matriz(df, estados_por_macro)
        for nome, df in amostras.items()
    }
    return matrizes, estados_por_macro


def ajustar_pooled(amostras, matrizes, args):
    validacoes = []
    modelos = {}
    for alpha in args.alphas:
        modelo, scaler, beta, intercepto = ajustar_ridge(
            matrizes["treino"], amostras["treino"]["y"], alpha
        )
        resultado = avaliar(modelo, scaler, matrizes["validacao"], amostras["validacao"]["y"])
        resultado.update({"alpha": float(alpha), "modelo": "pooled", "tipo": "validacao"})
        validacoes.append(resultado)
        modelos[float(alpha)] = (modelo, scaler, beta, intercepto)
    escolha = min(validacoes, key=lambda row: row["rmse_log"])
    alpha = float(escolha["alpha"])
    modelo, scaler, beta, intercepto = modelos[alpha]
    teste = avaliar(modelo, scaler, matrizes["teste"], amostras["teste"]["y"])
    teste.update({"alpha": alpha, "modelo": "pooled", "tipo": "teste"})
    return escolha, teste, (modelo, scaler, beta, intercepto), validacoes


def regressao_por_macro(amostras, matrizes, estados_por_macro, args):
    resultados = []
    ajustes = {}
    estabilidade = []
    for macro in M5.MACROREGIOES:
        treino = amostras["treino"]["Regiao"].eq(macro)
        validacao = amostras["validacao"]["Regiao"].eq(macro)
        teste = amostras["teste"]["Regiao"].eq(macro)
        if treino.sum() < 500 or validacao.sum() < 100 or teste.sum() < 100:
            continue
        xtr = matrizes["treino"].loc[treino].copy()
        # Retira colunas constantes na macroregião, como dummies macroregionais.
        colunas = [col for col in xtr.columns if xtr[col].nunique(dropna=False) > 1]
        xtr = xtr[colunas]
        xv = matrizes["validacao"].loc[validacao, colunas]
        xt = matrizes["teste"].loc[teste, colunas]
        ytr = amostras["treino"].loc[treino, "y"]
        yv = amostras["validacao"].loc[validacao, "y"]
        yt = amostras["teste"].loc[teste, "y"]
        escolhas = []
        modelos = {}
        for alpha in args.alphas:
            modelo, scaler, beta, intercepto = ajustar_ridge(xtr, ytr, alpha)
            score = avaliar(modelo, scaler, xv, yv)
            score.update({"macroregiao": macro, "alpha": float(alpha), "tipo": "validacao"})
            escolhas.append(score)
            modelos[float(alpha)] = (modelo, scaler, beta, intercepto)
        escolha = min(escolhas, key=lambda row: row["rmse_log"])
        alpha = float(escolha["alpha"])
        modelo, scaler, beta, intercepto = modelos[alpha]
        score = avaliar(modelo, scaler, xt, yt)
        score.update({"macroregiao": macro, "alpha": alpha, "tipo": "teste", "n_treino": int(treino.sum()), "n_teste": int(teste.sum())})
        resultados.extend([escolha, score])
        ajustes[macro] = {"modelo": modelo, "scaler": scaler, "beta": beta, "intercepto": intercepto, "colunas": colunas, "alpha": alpha}
        for nome, valor in zip(colunas, beta):
            estabilidade.append({"macroregiao": macro, "variavel": nome, "coeficiente": float(valor), "alpha": alpha})
    return pd.DataFrame(resultados), ajustes, pd.DataFrame(estabilidade)


def comparar_por_macro(amostras, matrizes, pooled_ajuste, separados):
    modelo, scaler, _, _ = pooled_ajuste
    linhas = []
    for macro in M5.MACROREGIOES:
        mask = amostras["teste"]["Regiao"].eq(macro)
        if mask.sum() == 0 or macro not in separados:
            continue
        pooled = metricas(amostras["teste"].loc[mask, "y"], modelo.predict(scaler.transform(matrizes["teste"].loc[mask])))
        ajuste = separados[macro]
        separado = metricas(
            amostras["teste"].loc[mask, "y"],
            ajuste["modelo"].predict(ajuste["scaler"].transform(matrizes["teste"].loc[mask, ajuste["colunas"]])),
        )
        linhas.append({
            "macroregiao": macro,
            "rmse_pooled": pooled["rmse_log"],
            "rmse_separado": separado["rmse_log"],
            "delta_rmse_separado_menos_pooled": separado["rmse_log"] - pooled["rmse_log"],
            "r2_pooled": pooled["r2_log"],
            "r2_separado": separado["r2_log"],
            "n_teste": int(mask.sum()),
        })
    return pd.DataFrame(linhas)


def teste_interacoes(matrizes, amostras, pooled_ajuste):
    x_full = matrizes["treino"]
    interacoes = [col for col in x_full.columns if "_x_Macro_" in col]
    x_reduzido = x_full.drop(columns=interacoes)
    x = np.column_stack([np.ones(len(x_full)), x_full.to_numpy(dtype=float)])
    xr = np.column_stack([np.ones(len(x_reduzido)), x_reduzido.to_numpy(dtype=float)])
    y = amostras["treino"]["y"].to_numpy(dtype=float)
    beta_full = np.linalg.lstsq(x, y, rcond=None)[0]
    beta_reduzido = np.linalg.lstsq(xr, y, rcond=None)[0]
    rss_full = float(np.sum((y - x @ beta_full) ** 2))
    rss_reduzido = float(np.sum((y - xr @ beta_reduzido) ** 2))
    q = len(interacoes)
    df_resid = max(len(y) - x.shape[1], 1)
    f_stat = ((rss_reduzido - rss_full) / max(q, 1)) / (rss_full / df_resid)
    p_value = float(stats.f.sf(f_stat, q, df_resid))
    pooled_model, pooled_scaler, _, _ = pooled_ajuste
    valid_full = avaliar(pooled_model, pooled_scaler, matrizes["validacao"], amostras["validacao"]["y"])
    # Ajuste reduzido com o mesmo alpha do pooled completo.
    alpha = float(getattr(pooled_model, "alpha", 100.0))
    reduced_model, reduced_scaler, _, _ = ajustar_ridge(x_reduzido, y, alpha)
    valid_reduced = avaliar(reduced_model, reduced_scaler, matrizes["validacao"].drop(columns=interacoes), amostras["validacao"]["y"])
    test_full = avaliar(pooled_model, pooled_scaler, matrizes["teste"], amostras["teste"]["y"])
    test_reduced = avaliar(reduced_model, reduced_scaler, matrizes["teste"].drop(columns=interacoes), amostras["teste"]["y"])
    return {
        "n_interacoes": q, "f_stat_ols_nao_clusterizado": float(f_stat), "p_valor": p_value,
        "rmse_validacao_com_interacoes": valid_full["rmse_log"], "rmse_validacao_sem_interacoes": valid_reduced["rmse_log"],
        "rmse_teste_com_interacoes": test_full["rmse_log"], "rmse_teste_sem_interacoes": test_reduced["rmse_log"],
        "observacao": "O F-test é uma referência OLS não ponderada e não clusterizada; a decisão preditiva usa a validação Ridge.",
    }


def estabilidade_coeficientes(estabilidade: pd.DataFrame) -> pd.DataFrame:
    comum = estabilidade.groupby("variavel").agg(
        media_coeficiente=("coeficiente", "mean"), desvio_padrao=("coeficiente", "std"),
        minimo=("coeficiente", "min"), maximo=("coeficiente", "max"),
        n_macroregioes=("macroregiao", "nunique"),
    ).reset_index()
    sinais = estabilidade.assign(sinal=np.sign(estabilidade["coeficiente"]))
    consistencia = sinais.groupby("variavel")["sinal"].agg(lambda s: float(max((s > 0).mean(), (s < 0).mean()))).reset_index(name="consistencia_sinal")
    return comum.merge(consistencia, on="variavel", how="left").sort_values("consistencia_sinal")


def oaxaca(amostras, matrizes):
    df = amostras["treino"].copy()
    x_full = matrizes["treino"].copy()
    resultados = []
    especificacoes = {
        "genero_homem_mulher": (df["Sexo"].eq(1), df["Sexo"].eq(2), [c for c in x_full if c == "Sexo_Dummy" or c.startswith("Sexo_x_")]),
        "cor_branca_nao_branca": (df["Cor"].eq(1), ~df["Cor"].eq(1), [c for c in x_full if c.startswith("Cor_")]),
    }
    for nome, (grupo_a, grupo_b, remover) in especificacoes.items():
        cols = [c for c in x_full.columns if c not in remover and x_full.loc[grupo_a | grupo_b, c].nunique(dropna=False) > 1]
        xa, xb = x_full.loc[grupo_a, cols], x_full.loc[grupo_b, cols]
        ya, yb = df.loc[grupo_a, "y"], df.loc[grupo_b, "y"]
        modelo_a, scaler_a, beta_a, int_a = ajustar_ridge(xa, ya, 100.0)
        modelo_b, scaler_b, beta_b, int_b = ajustar_ridge(xb, yb, 100.0)
        ma, mb = xa.mean().to_numpy(), xb.mean().to_numpy()
        va = np.r_[int_a, beta_a]
        vb = np.r_[int_b, beta_b]
        mean_a, mean_b = np.r_[1.0, ma], np.r_[1.0, mb]
        gap = float(ya.mean() - yb.mean())
        gap_modelo = float(mean_a @ va - mean_b @ vb)
        explicado = float((mean_a - mean_b) @ vb)
        nao_explicado = float(mean_b @ (va - vb))
        resultados.append({
            "comparacao": nome, "grupo_a": int(grupo_a.sum()), "grupo_b": int(grupo_b.sum()),
            "gap_observado_log": gap, "gap_reproduzido_modelo": gap_modelo,
            "componente_explicado": explicado, "componente_nao_explicado": nao_explicado,
            "percentual_explicado": explicado / gap * 100 if gap else np.nan,
            "observacao": "Decomposição Ridge; a parcela não explicada não é sinônimo de discriminação causal.",
        })
    return pd.DataFrame(resultados)


def heckman_dois_passos(args):
    colunas = [
        "Ano", "Idade", "Anos_de_Estudo", "Sexo", "Cor", "Regiao", "Condicao_Ocupacao",
        "Categoria_emprego", "Contribuicao_previdencia", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas",
    ]
    try:
        pop = pd.read_parquet(args.entrada, columns=colunas, filters=[("Ano", "<=", 2022)])
    except Exception as exc:
        return {"status": "não estimado", "motivo": f"falha na leitura da população: {exc}"}, pd.DataFrame()
    pop["Idade"] = pd.to_numeric(pop["Idade"], errors="coerce")
    pop["Anos_de_Estudo"] = pd.to_numeric(pop["Anos_de_Estudo"], errors="coerce")
    pop = pop[pop["Idade"].between(25, 65) & pop["Anos_de_Estudo"].between(0, 16) & pop["Regiao"].isin(M5.MACROREGIOES)].dropna(subset=["Sexo", "Cor"])
    pop = M5.amostrar(pop, args.max_selecao_heckman, args.seed + 7)
    pop["ocupado"] = pop["Condicao_Ocupacao"].eq(1).astype(int)
    if pop["ocupado"].nunique() < 2:
        return {
            "status": "não estimado",
            "motivo": "A base consolidada contém apenas indivíduos ocupados; não há grupo de não ocupados para estimar a equação de seleção.",
        }, pd.DataFrame()
    pop["idade_c"] = pop["Idade"] - 45
    sel = pd.DataFrame({"Anos_de_Estudo": pop["Anos_de_Estudo"], "idade_c": pop["idade_c"], "idade_c_2": pop["idade_c"] ** 2, "Mulher": pop["Sexo"].eq(2).astype(int)})
    for code, name in M5.CORES.items():
        sel[name] = pop["Cor"].eq(code).astype(int)
    for macro in M5.MACROREGIOES[1:]:
        sel[f"Macro_{macro}"] = pop["Regiao"].eq(macro).astype(int)
    sel = sel.astype(float)
    try:
        # A seleção apresentou separação perfeita no Probit não regularizado.
        # Usa-se Logit regularizado apenas para construir a razão de Mills;
        # o resultado é uma sensibilidade, não uma estimação Heckman clássica.
        selecao = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        selecao.fit(sel, pop["ocupado"])
        z = np.asarray(selecao.decision_function(sel), dtype=float)
        lambda_mills = stats.norm.pdf(z) / np.clip(stats.norm.cdf(z), 1e-12, None)
        ocupados = pop["ocupado"].eq(1) & pop["Rendimento_hab_Trab_princ"].gt(0) & pop["Horas_hab_trabalhadas"].gt(0)
        y = np.log(pop.loc[ocupados, "Rendimento_hab_Trab_princ"] / (pop.loc[ocupados, "Horas_hab_trabalhadas"] * 4.33))
        wage_x = sel.loc[ocupados].copy()
        wage_x["lambda_mills"] = lambda_mills[ocupados.to_numpy()]
        ajuste = sm.OLS(y, sm.add_constant(wage_x, has_constant="add")).fit(cov_type="HC1")
        coef = pd.DataFrame({"variavel": ajuste.params.index, "coeficiente": ajuste.params.values, "p_valor_hc1": ajuste.pvalues.values})
        resumo = {"status": "estimado como sensibilidade", "n_populacao": int(len(pop)), "n_ocupados": int(ocupados.sum()), "lambda_coeficiente": float(ajuste.params["lambda_mills"]), "lambda_p_valor_hc1": float(ajuste.pvalues["lambda_mills"]), "observacao": "A seleção usa Logit regularizado porque o Probit apresentou separação perfeita. Não há variável de exclusão validada; a identificação depende da forma funcional e da regularização."}
        return resumo, coef
    except Exception as exc:
        return {"status": "não estimado", "motivo": str(exc)}, pd.DataFrame()


def limitações_base(args):
    colunas = pd.read_parquet(args.entrada, columns=None).columns.tolist()
    peso_candidatos = [c for c in colunas if any(t in c.lower() for t in ("peso", "weight", "v1028", "v1029"))]
    return {
        "pesos_amostrais": {"status": "não estimado", "colunas_candidatas_encontradas": peso_candidatos, "motivo": "A base consolidada não contém peso amostral; seria necessário reextrair e propagar a variável de peso dos microdados."},
        "variaveis_instrumentais": {"status": "não estimado", "motivo": "Não há instrumento exógeno documentado no projeto para escolaridade ou participação no mercado de trabalho; usar UF, ano ou características demográficas como instrumento seria indefensável sem hipótese adicional."},
    }


def salvar_relatorio(args, pooled_valid, pooled_test, separados, comparacao, interacoes, oaxaca_df, heckman, limitacoes):
    linhas = [
        "# Análises complementares do salário-hora", "",
        "A avaliação usa indivíduos de 25 a 65 anos, alvo `log(Salario_Hora)`, treino 2013–2022, validação 2023–2024 e teste 2025.", "",
        "## Modelo agrupado e regressões por macroregião", "",
        f"O modelo agrupado selecionado na validação teve RMSE log {pooled_valid['rmse_log']:.4f} e, no teste, RMSE log {pooled_test['rmse_log']:.4f}.", "",
        "A comparação detalhada está em `comparacao_pooled_separado.csv`.", "",
        "## Heterogeneidade regional", "",
        f"O teste conjunto das {interacoes['n_interacoes']} interações teve F = {interacoes['f_stat_ols_nao_clusterizado']:.2f} e p = {interacoes['p_valor']:.4g}. Este F-test é uma referência OLS não ponderada e não clusterizada.", "",
        "## Oaxaca-Blinder", "",
        "A decomposição Ridge para gênero e cor está em `oaxaca_decomposicao.csv`. A parcela não explicada não deve ser chamada de discriminação sem desenho causal.", "",
        "## Heckman", "",
        f"Status: {heckman.get('status')}. O resultado e o coeficiente da razão inversa de Mills estão em `heckman_resumo.json` e `heckman_coeficientes.csv`.", "",
        "## Limitações", "",
        "- A base consolidada não contém pesos amostrais; a análise ponderada não foi estimada.",
        "- Não há instrumento exógeno documentado; 2SLS/IV não foi estimado.",
        "- Ridge fornece coeficientes preditivos regularizados, sem p-valores OLS convencionais.",
        "- Todas as estimativas são associações condicionais e não efeitos causais.",
    ]
    (args.saida_dir / "analises_complementares.md").write_text("\n".join(linhas), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    amostras = preparar_amostras(args)
    matrizes, estados_por_macro = construir_matrizes(amostras)
    pooled_valid, pooled_test, pooled_ajuste, validacoes = ajustar_pooled(amostras, matrizes, args)
    separados_df, separados, estabilidade = regressao_por_macro(amostras, matrizes, estados_por_macro, args)
    comparacao = comparar_por_macro(amostras, matrizes, pooled_ajuste, separados)
    interacoes = teste_interacoes(matrizes, amostras, pooled_ajuste)
    estabilidade_resumo = estabilidade_coeficientes(estabilidade)
    oaxaca_df = oaxaca(amostras, matrizes)
    heckman, heckman_coef = heckman_dois_passos(args)
    limitacoes = limitações_base(args)

    separados_df.to_csv(args.saida_dir / "regressoes_por_macroregiao.csv", index=False)
    comparacao.to_csv(args.saida_dir / "comparacao_pooled_separado.csv", index=False)
    estabilidade.to_csv(args.saida_dir / "coeficientes_estabilidade_macroregiao.csv", index=False)
    estabilidade_resumo.to_csv(args.saida_dir / "resumo_estabilidade_coeficientes.csv", index=False)
    oaxaca_df.to_csv(args.saida_dir / "oaxaca_decomposicao.csv", index=False)
    heckman_coef.to_csv(args.saida_dir / "heckman_coeficientes.csv", index=False)
    (args.saida_dir / "heckman_resumo.json").write_text(json.dumps(heckman, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (args.saida_dir / "teste_interacoes_regionais.json").write_text(json.dumps(interacoes, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (args.saida_dir / "limitacoes_analises_complementares.json").write_text(json.dumps(limitacoes, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (args.saida_dir / "alphas_pooled_complementar.csv").write_text(pd.DataFrame(validacoes).to_csv(index=False), encoding="utf-8")
    salvar_relatorio(args, pooled_valid, pooled_test, separados_df, comparacao, interacoes, oaxaca_df, heckman, limitacoes)
    print(json.dumps({"pooled_validacao": pooled_valid, "pooled_teste": pooled_test, "interacoes": interacoes, "heckman": heckman, "arquivos": str(args.saida_dir)}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
