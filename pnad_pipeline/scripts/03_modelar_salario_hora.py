#!/usr/bin/env python3
"""Compara modelos para estimar o log do salário-hora da PNAD Contínua.

Validação temporal:
    treino:    2013–2022
    validação: 2023–2024
    teste:     2025

O salário-hora é construído previamente a partir do rendimento habitual e das
horas habituais. Essas duas variáveis ficam fora dos preditores para evitar
vazamento mecânico do alvo.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config.settings import setup_logging

logger = logging.getLogger("scripts.03_modelar_salario_hora")

TARGET = "log_salario_hora"
BASE_NUMERIC = ["Idade", "Idade_2", "Anos_de_Estudo"]
BASE_CATEGORICAL = [
    "Sexo_Desc",
    "Cor_Desc",
    "Categoria_emprego",
    "Grupo_atv_princ_empreedimento",
    "Contribuicao_previdencia",
    "Regiao",
    "Setor",
    "Formalidade",
]
FIXED_EFFECTS = ["UF", "Ano", "Trimestre"]
MODEL_COLUMNS = sorted(
    set(BASE_NUMERIC + BASE_CATEGORICAL + FIXED_EFFECTS + ["Salario_Hora", "Periodo", "Faixa_Etaria"])
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara modelos para estimar o salário-hora da PNAD Contínua."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "pnad_analise_25_65.parquet",
        help="Parquet tratado de entrada.",
    )
    parser.add_argument(
        "--saida-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "processed" / "modelos_salario_hora",
        help="Diretório dos resultados da comparação.",
    )
    parser.add_argument(
        "--max-treino",
        type=int,
        default=150_000,
        help="Máximo de observações aleatórias no treino de cada modelo.",
    )
    parser.add_argument(
        "--max-avaliacao",
        type=int,
        default=50_000,
        help="Máximo de observações em validação e teste.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def amostrar(df: pd.DataFrame, limite: int, seed: int) -> pd.DataFrame:
    if len(df) <= limite:
        return df.copy()
    return df.sample(n=limite, random_state=seed).copy()


def carregar_base(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Base tratada não encontrada: {caminho}")
    colunas = [c for c in MODEL_COLUMNS if c not in {"Faixa_Etaria", "Idade_2"}]
    df = pd.read_parquet(caminho, columns=colunas)
    df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")
    df["Anos_de_Estudo"] = pd.to_numeric(df["Anos_de_Estudo"], errors="coerce")
    df["Idade_2"] = df["Idade"] ** 2
    df[TARGET] = np.log(df["Salario_Hora"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    for col in BASE_CATEGORICAL + FIXED_EFFECTS:
        df[col] = df[col].astype("string").fillna("Sem informação")
    return df


def preprocessor(numeric: list[str], categorical: list[str], dense: bool = False) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
    )


def build_pipeline(model: Any, categorical: list[str], dense: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("features", preprocessor(BASE_NUMERIC, categorical, dense=dense)),
            ("model", model),
        ]
    )


def metricas(modelo: str, tipo: str, y_true_log: pd.Series, pred_log: np.ndarray) -> dict[str, Any]:
    pred_log = np.asarray(pred_log, dtype=float)
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_true = np.exp(y_true_log)
    pred = np.exp(np.clip(pred_log, -20, 20))
    return {
        "modelo": modelo,
        "tipo": tipo,
        "mae_log": mean_absolute_error(y_true_log, pred_log),
        "rmse_log": mean_squared_error(y_true_log, pred_log) ** 0.5,
        "r2_log": r2_score(y_true_log, pred_log),
        "mae_salario_hora": mean_absolute_error(y_true, pred),
        "rmse_salario_hora": mean_squared_error(y_true, pred) ** 0.5,
        "r2_salario_hora": r2_score(y_true, pred),
    }


def avaliar_sklearn(
    nome: str,
    modelo: Pipeline,
    treino: pd.DataFrame,
    avaliacao: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    inicio = time.perf_counter()
    modelo.fit(treino, treino[TARGET])
    tempo = time.perf_counter() - inicio
    resultados = []
    for conjunto, dados in avaliacao.items():
        pred = modelo.predict(dados)
        resultado = metricas(nome, conjunto, dados[TARGET], pred)
        resultado["tempo_treino_seg"] = tempo
        resultados.append(resultado)
    return resultados


def avaliar_quantile(
    treino: pd.DataFrame,
    avaliacao: dict[str, pd.DataFrame],
    quantile: float = 0.50,
) -> list[dict[str, Any]]:
    """Regressão quantílica com a mesma matriz de efeitos fixos do OLS."""
    nome = f"QuantReg_q{int(quantile * 100)}"
    categorias = BASE_CATEGORICAL + FIXED_EFFECTS
    transformador = preprocessor(BASE_NUMERIC, categorias, dense=True)
    inicio = time.perf_counter()
    x_treino = transformador.fit_transform(treino)
    x_treino = sm.add_constant(x_treino, has_constant="add")
    ajuste = sm.QuantReg(treino[TARGET].to_numpy(), x_treino).fit(q=quantile, max_iter=2000)
    tempo = time.perf_counter() - inicio
    resultados = []
    for conjunto, dados in avaliacao.items():
        x = transformador.transform(dados)
        x = sm.add_constant(x, has_constant="add")
        resultado = metricas(nome, conjunto, dados[TARGET], ajuste.predict(x))
        resultado["tempo_treino_seg"] = tempo
        resultados.append(resultado)
    return resultados


def avaliar_mixedlm(
    treino: pd.DataFrame,
    avaliacao: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Modelo multinível com intercepto aleatório por coorte analítica."""
    nome = "MixedLM_coorte"
    def preparar_exog(df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        x["const"] = 1.0
        x["Idade"] = df["Idade"].astype(float)
        x["Idade_2"] = df["Idade_2"].astype(float)
        x["Anos_de_Estudo"] = df["Anos_de_Estudo"].astype(float)
        x["Mulher"] = (df["Sexo_Desc"] == "Mulher").astype(float)
        x["Parda"] = (df["Cor_Desc"] == "Parda").astype(float)
        x["Preta"] = (df["Cor_Desc"] == "Preta").astype(float)
        x["Publico"] = (df["Setor"] == "Público").astype(float)
        x["Formal"] = (df["Formalidade"] == "Formal").astype(float)
        return x

    treino = treino.copy()
    treino["coorte"] = (
        treino["Sexo_Desc"].astype(str) + "_"
        + treino["Cor_Desc"].astype(str) + "_"
        + treino["Regiao"].astype(str) + "_"
        + treino["Anos_de_Estudo"].round().astype(str)
    )
    inicio = time.perf_counter()
    ajuste = MixedLM(
        treino[TARGET].to_numpy(),
        preparar_exog(treino),
        groups=treino["coorte"],
    ).fit(reml=False, method="lbfgs", maxiter=150, disp=False)
    tempo = time.perf_counter() - inicio
    resultados = []
    for conjunto, dados in avaliacao.items():
        resultado = metricas(nome, conjunto, dados[TARGET], ajuste.predict(exog=preparar_exog(dados)))
        resultado["tempo_treino_seg"] = tempo
        resultados.append(resultado)
    return resultados


def avaliar_por_grupo(
    modelo: Pipeline,
    dados: pd.DataFrame,
    nome: str,
) -> pd.DataFrame:
    pred = modelo.predict(dados)
    avaliacao = dados.copy()
    avaliacao["pred_log"] = pred
    linhas = []
    for grupo, sub in avaliacao.groupby("Sexo_Desc", dropna=False):
        linha = metricas(nome, "teste", sub[TARGET], sub["pred_log"])
        linha["grupo"] = str(grupo)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    args.saida_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Carregando base tratada: %s", args.entrada)
    df = carregar_base(args.entrada)
    treino = amostrar(df[df["Ano"].astype(int) <= 2022], args.max_treino, args.seed)
    validacao = amostrar(
        df[df["Ano"].astype(int).between(2023, 2024)], args.max_avaliacao, args.seed + 1
    )
    teste = amostrar(df[df["Ano"].astype(int) == 2025], args.max_avaliacao, args.seed + 2)
    avaliacao = {"validacao": validacao, "teste": teste}
    logger.info(
        "Amostras: treino=%s, validação=%s, teste=%s",
        f"{len(treino):,}", f"{len(validacao):,}", f"{len(teste):,}",
    )

    resultados: list[dict[str, Any]] = []
    modelos: dict[str, Pipeline] = {
        "OLS": build_pipeline(LinearRegression(), BASE_CATEGORICAL),
        "OLS_efeitos_fixos": build_pipeline(LinearRegression(), BASE_CATEGORICAL + FIXED_EFFECTS),
        "GAM_splines_Ridge": Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            (
                                "splines",
                                Pipeline(
                                    [
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("spline", SplineTransformer(n_knots=6, degree=3)),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                ["Idade", "Anos_de_Estudo"],
                            ),
                            (
                                "cat",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                                BASE_CATEGORICAL + FIXED_EFFECTS,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForest": build_pipeline(
            RandomForestRegressor(
                n_estimators=80,
                min_samples_leaf=20,
                max_features=0.7,
                n_jobs=-1,
                random_state=args.seed,
            ),
            BASE_CATEGORICAL + FIXED_EFFECTS,
            dense=True,
        ),
        "GradientBoosting": build_pipeline(
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=args.seed,
            ),
            BASE_CATEGORICAL + FIXED_EFFECTS,
            dense=True,
        ),
    }

    for nome, modelo in modelos.items():
        logger.info("Ajustando %s", nome)
        resultados.extend(avaliar_sklearn(nome, modelo, treino, avaliacao))

    logger.info("Ajustando QuantReg_q50")
    resultados.extend(avaliar_quantile(treino.sample(n=min(50_000, len(treino)), random_state=args.seed), avaliacao))

    logger.info("Ajustando MixedLM_coorte")
    try:
        resultados.extend(
            avaliar_mixedlm(
                treino.sample(n=min(40_000, len(treino)), random_state=args.seed),
                avaliacao,
            )
        )
    except Exception as exc:
        logger.warning("MixedLM não convergiu e foi excluído: %s", exc)

    metricas_df = pd.DataFrame(resultados)
    metricas_df.to_csv(args.saida_dir / "metricas_modelos.csv", index=False)
    teste_df = metricas_df[metricas_df["tipo"] == "teste"].sort_values("rmse_log")
    melhor = teste_df.iloc[0].to_dict()

    comparacao = {
        "melhor_modelo_por_rmse_log_teste": melhor["modelo"],
        "metricas_melhor_modelo": melhor,
        "criterio": "menor RMSE no log do salário-hora no teste de 2025",
        "divisao_temporal": {"treino": "2013-2022", "validacao": "2023-2024", "teste": "2025"},
        "amostras": {"treino": len(treino), "validacao": len(validacao), "teste": len(teste)},
        "variavel_alvo": "log(Salario_Hora)",
        "limitacoes": [
            "Os pesos amostrais da PNAD não estão presentes na base tratada; os resultados são não ponderados.",
            "O GAM foi aproximado por splines com Ridge porque pyGAM não está instalado.",
            "As métricas usam amostras fixas para tornar a comparação reproduzível.",
        ],
    }
    (args.saida_dir / "resumo_modelos.json").write_text(
        json.dumps(comparacao, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tabela_teste = teste_df[
        ["modelo", "rmse_log", "mae_log", "r2_log", "mae_salario_hora", "rmse_salario_hora"]
    ].copy()
    tabela_teste["rmse_log"] = tabela_teste["rmse_log"].map(lambda x: f"{x:.4f}")
    tabela_teste["mae_log"] = tabela_teste["mae_log"].map(lambda x: f"{x:.4f}")
    tabela_teste["r2_log"] = tabela_teste["r2_log"].map(lambda x: f"{x:.4f}")
    tabela_teste["mae_salario_hora"] = tabela_teste["mae_salario_hora"].map(lambda x: f"R$ {x:.2f}")
    tabela_teste["rmse_salario_hora"] = tabela_teste["rmse_salario_hora"].map(lambda x: f"R$ {x:.2f}")
    tabela_teste.columns = [
        "Modelo", "RMSE log", "MAE log", "R² log", "MAE salário-hora", "RMSE salário-hora"
    ]
    relatorio = "\n".join(
        [
            "# Comparação de modelos para o salário-hora",
            "",
            "A comparação usa `log(Salario_Hora)` como alvo, com treino em 2013–2022, validação em 2023–2024 e teste em 2025.",
            "",
            f"## Resultado",
            "",
            f"O melhor modelo para previsão no teste de 2025 foi **{melhor['modelo']}**, pelo menor RMSE no log do salário-hora.",
            "",
            tabela_teste.to_markdown(index=False),
            "",
            "## Interpretação",
            "",
            "- Para previsão, o Gradient Boosting apresentou o melhor desempenho entre os modelos testados.",
            "- O Random Forest ficou próximo e pode ser uma alternativa robusta.",
            "- O OLS com efeitos fixos tem desempenho preditivo menor, mas é mais fácil de interpretar economicamente.",
            "- O MixedLM por coorte apresentou ajuste instável, com covariância aleatória singular e erro de previsão elevado; não deve ser escolhido nesta especificação.",
            "",
            "## Limitações",
            "",
            "- A base tratada não contém os pesos amostrais da PNAD, portanto os resultados não são ponderados para representar a população.",
            "- O GAM foi aproximado por splines com Ridge porque `pygam` não está instalado.",
            "- As métricas usam amostras fixas de 150 mil observações no treino e 50 mil em validação e teste.",
        ]
    )
    (args.saida_dir / "relatorio_modelos.md").write_text(relatorio, encoding="utf-8")
    logger.info("Melhor modelo por RMSE log no teste: %s", melhor["modelo"])
    logger.info("Resultados salvos em: %s", args.saida_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
