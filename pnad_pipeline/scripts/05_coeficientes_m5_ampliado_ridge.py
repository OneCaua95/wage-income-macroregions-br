#!/usr/bin/env python3
"""Gera os coeficientes lineares do M5 ampliado com Ridge.

O modelo usa a mesma população e a mesma divisão temporal do benchmark:
treino 2013--2022, validação 2023--2024 e teste 2025. Os coeficientes são
convertidos da escala padronizada para a escala original das variáveis.

Como há regularização Ridge, o arquivo não calcula p-valores: os coeficientes
são úteis para direção e magnitude preditiva, mas não representam uma
inferência OLS convencional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SETORES = {
    2: "Setor_Industria_Geral",
    3: "Setor_Construcao",
    4: "Setor_Comercio",
    5: "Setor_Transporte_Armazenagem_Correio",
    6: "Setor_Alojamento_Alimentacao",
    7: "Setor_Info_Comunicacao_Financ_Imob_Prof_Adm",
    8: "Setor_Adm_Publica",
    9: "Setor_Educacao_Saude",
    10: "Setor_Outros_Servicos",
}
OCUPACOES = {
    1: "Ocup_Diretores_Gerentes",
    2: "Ocup_Profissionais_Ciencias_Intelectuais",
    3: "Ocup_Tecnicos_Nivel_Medio",
    4: "Ocup_Apoio_Administrativo",
    5: "Ocup_Servicos_Vendedores_Comercio_Mercado",
    6: "Ocup_Qualificados_Agropecuaria_Pesca",
    7: "Ocup_Qualificados_Construcao_Mecanica_Oficios",
    8: "Ocup_Operadores_Instalacoes_Maquinas",
    9: "Ocup_Forcas_Armadas_Policiais_Bombeiros",
}
CORES = {2: "Cor_Preta", 3: "Cor_Amarela", 4: "Cor_Parda", 5: "Cor_Indigena"}
BIRTH_BINS = [1947, 1952, 1957, 1962, 1967, 1972, 1977, 1982, 1988, 1993, 1998, 2003]
RAW_COLUMNS = [
    "Ano", "Trimestre", "Ano_Nascimento", "Idade", "UF", "Sexo", "Cor",
    "Anos_de_Estudo", "Categoria_emprego", "Contribuicao_previdencia",
    "Grupo_atv_princ_empreedimento", "Grupo_ocupacional_no_emprego",
    "Condicao_Ocupacao", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas",
    "Regiao",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrada", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet",
    )
    parser.add_argument(
        "--saida-dir", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "modelos_salario_hora",
    )
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--max-treino", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def amostrar(df: pd.DataFrame, limite: int, seed: int) -> pd.DataFrame:
    return df.sample(n=limite, random_state=seed).copy() if len(df) > limite else df.copy()


def carregar(caminho: Path, limite: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(caminho, columns=RAW_COLUMNS)
    df["Idade"] = pd.to_numeric(df["Idade"], errors="coerce")
    df["Anos_de_Estudo"] = pd.to_numeric(df["Anos_de_Estudo"], errors="coerce")
    mascara = (
        df["Idade"].between(25, 65)
        & df["Condicao_Ocupacao"].eq(1)
        & df["Rendimento_hab_Trab_princ"].gt(0)
        & df["Horas_hab_trabalhadas"].gt(0)
        & df["Anos_de_Estudo"].between(0, 16)
        & df["Regiao"].notna()
        & df["UF"].notna()
        & df["Cor"].notna()
    )
    nascimento = df["Ano_Nascimento"].where(
        df["Ano_Nascimento"].between(1947, 2003), df["Ano"] - df["Idade"]
    )
    mascara &= pd.cut(nascimento, bins=BIRTH_BINS, labels=False, include_lowest=True).notna()
    df = df.loc[mascara].copy()
    df["Salario_Hora"] = df["Rendimento_hab_Trab_princ"] / (df["Horas_hab_trabalhadas"] * 4.33)
    df["y"] = np.log(df["Salario_Hora"])
    treino = amostrar(df[df["Ano"].between(2013, 2022)], limite, seed)
    validacao = amostrar(df[df["Ano"].between(2023, 2024)], 50_000, seed + 1)
    teste = amostrar(df[df["Ano"].eq(2025)], 50_000, seed + 2)
    return treino, validacao, teste


def criar_matriz(df: pd.DataFrame, ufs: list, include_year: bool = True) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["is_formal"] = (
        df["Categoria_emprego"].isin([1, 3, 5, 7])
        | (df["Categoria_emprego"].isin([8, 9]) & df["Contribuicao_previdencia"].eq(1))
    ).astype(float)
    x["is_public"] = df["Categoria_emprego"].isin([5, 6, 7]).astype(float)
    x["Sexo_Dummy"] = df["Sexo"].eq(2).astype(float)
    for code, name in SETORES.items():
        x[name] = df["Grupo_atv_princ_empreedimento"].eq(code).astype(float)
    for code, name in OCUPACOES.items():
        x[name] = df["Grupo_ocupacional_no_emprego"].eq(code).astype(float)
    for code, name in CORES.items():
        x[name] = df["Cor"].eq(code).astype(float)
    if include_year:
        for year in range(2014, 2023):
            x[f"Ano_{year}"] = df["Ano"].eq(year).astype(float)
    x["Idade_centrada"] = df["Idade"].astype(float) - 45.0
    x["Idade_centrada_2"] = x["Idade_centrada"] ** 2
    for uf in ufs[1:]:
        x[f"UF_{uf}"] = df["UF"].eq(uf).astype(float)
    for trimestre in (2, 3, 4):
        x[f"Trimestre_{trimestre}"] = df["Trimestre"].eq(trimestre).astype(float)
    for nivel in range(1, 17):
        x[f"Estudo_{nivel}"] = df["Anos_de_Estudo"].eq(nivel).astype(float)
    for col in ("is_formal", "is_public", "Sexo_Dummy"):
        x[f"Estudo_x_{col}"] = df["Anos_de_Estudo"].astype(float) * x[col]
    return x


def classificar(nome: str) -> tuple[str, str]:
    if nome == "Intercept":
        return "intercepto", "nível de referência"
    if nome.startswith("Estudo_") and "_x_" not in nome:
        return "escolaridade categórica", "comparação com 0 anos de estudo"
    if nome.startswith("Estudo_x_"):
        return "interação", "variação da escolaridade condicionada ao grupo"
    if nome.startswith("UF_"):
        return "UF", "comparação com a UF de referência"
    if nome.startswith("Trimestre_"):
        return "trimestre", "comparação com o 1º trimestre"
    if nome.startswith("Ano_"):
        return "efeito fixo de ano", "comparação com 2013"
    if nome in {"is_formal", "is_public", "Sexo_Dummy"} or nome.startswith("Setor_") or nome.startswith("Ocup_") or nome.startswith("Cor_"):
        return "indicadora", "diferença em relação à categoria de referência"
    if nome == "Idade_centrada":
        return "contínua", "variação de um ano; com termo quadrático"
    if nome == "Idade_centrada_2":
        return "não linearidade", "curvatura da idade"
    return "variável", "interpretação condicionada às demais variáveis"


def main() -> int:
    args = parse_args()
    treino, validacao, teste = carregar(args.entrada, args.max_treino, args.seed)
    ufs = sorted(treino["UF"].dropna().unique())
    x_treino = criar_matriz(treino, ufs)
    x_validacao = criar_matriz(validacao, ufs)
    x_teste = criar_matriz(teste, ufs)
    scaler = StandardScaler().fit(x_treino)
    modelo = Ridge(alpha=args.alpha).fit(scaler.transform(x_treino), treino["y"])

    coeficientes = modelo.coef_ / scaler.scale_
    intercepto = modelo.intercept_ - np.sum(modelo.coef_ * scaler.mean_ / scaler.scale_)
    nomes = ["Intercept"] + list(x_treino.columns)
    valores = np.r_[intercepto, coeficientes]
    linhas = []
    for nome, valor in zip(nomes, valores):
        tipo, observacao = classificar(nome)
        efeito = np.expm1(valor) * 100 if tipo in {"indicadora", "UF", "trimestre", "escolaridade categórica"} else np.nan
        linhas.append({
            "variavel": nome,
            "tipo": tipo,
            "coeficiente_linear": float(valor),
            "coeficiente_padronizado": float(modelo.intercept_ if nome == "Intercept" else modelo.coef_[len(linhas) - 1]),
            "efeito_percentual_aprox": float(efeito) if np.isfinite(efeito) else np.nan,
            "observacao": observacao,
        })
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(args.saida_dir / "coeficientes_m5_ampliado_ridge.csv", index=False)

    resumo = {
        "modelo": "M5_ampliado_Ridge",
        "alpha": args.alpha,
        "escala_coeficientes": "original, após desfazer a padronização do Ridge",
        "treino": "2013-2022",
        "ufs_referencia": int(ufs[0]),
        "observacao_inferencia": "Ridge reduz variância e estabiliza a previsão; p-valores OLS não são reportados.",
        "rmse_log_validacao": float(np.sqrt(np.mean((modelo.predict(scaler.transform(x_validacao)) - validacao["y"]) ** 2))),
        "rmse_log_teste": float(np.sqrt(np.mean((modelo.predict(scaler.transform(x_teste)) - teste["y"]) ** 2))),
        "n_coeficientes": len(nomes),
    }
    (args.saida_dir / "resumo_m5_ampliado_ridge.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"Coeficientes salvos em {args.saida_dir / 'coeficientes_m5_ampliado_ridge.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
