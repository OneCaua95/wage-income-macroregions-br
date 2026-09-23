#!/usr/bin/env python3
"""Estima a versão inferencial do M5 ampliado em toda a base.

O objetivo é separar claramente previsão de inferência. O modelo é OLS para
log(Salario_Hora), com a mesma família de controles do M5 ampliado: idade,
idade ao quadrado, escolaridade categórica, interações da escolaridade,
formalidade, setor público, sexo, setor, ocupação, cor, UF e efeitos fixos de
ano. A estimação usa todas as observações elegíveis do Parquet, em lotes, sem
montar uma matriz com as 9 milhões de linhas em memória.

São comparadas três versões de variância:

* HC1: robusta a heterocedasticidade;
* cluster por UF;
* cluster em duas vias por coorte e UF, escolhida como inferência principal.

Os p-valores são válidos apenas sob as hipóteses do modelo e do estimador de
variância. Eles não provam causalidade. Como a PNAD consolidada disponível é
uma seção transversal repetida, sem pesos amostrais, instrumento exógeno ou
identificador de painel documentado, o script também grava um diagnóstico
explícito de identificação causal.

Memória: a leitura é feita com PyArrow em lotes, as estatísticas X'X e X'y
são acumuladas, e a matriz de desenho é descartada a cada lote. O padrão de
100 mil linhas foi escolhido para permanecer conservador em uma máquina com
16 GB de RAM; use --batch-size menor se necessário.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SETORES = {
    2: "Setor_Industria_Geral", 3: "Setor_Construcao", 4: "Setor_Comercio",
    5: "Setor_Transporte_Armazenagem_Correio", 6: "Setor_Alojamento_Alimentacao",
    7: "Setor_Info_Comunicacao_Financ_Imob_Prof_Adm", 8: "Setor_Adm_Publica",
    9: "Setor_Educacao_Saude", 10: "Setor_Outros_Servicos",
}
OCUPACOES = {
    1: "Ocup_Diretores_Gerentes", 2: "Ocup_Profissionais_Ciencias_Intelectuais",
    3: "Ocup_Tecnicos_Nivel_Medio", 4: "Ocup_Apoio_Administrativo",
    5: "Ocup_Servicos_Vendedores_Comercio_Mercado", 6: "Ocup_Qualificados_Agropecuaria_Pesca",
    7: "Ocup_Qualificados_Construcao_Mecanica_Oficios", 8: "Ocup_Operadores_Instalacoes_Maquinas",
    9: "Ocup_Forcas_Armadas_Policiais_Bombeiros",
}
CORES = {2: "Cor_Preta", 3: "Cor_Amarela", 4: "Cor_Parda", 5: "Cor_Indigena"}
MACROREGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
ESTADOS_POR_MACRO = {
    "Norte": [11, 12, 13, 14, 15, 16, 17],
    "Nordeste": [21, 22, 23, 24, 25, 26, 27, 28, 29],
    "Sudeste": [31, 32, 33, 35],
    "Sul": [41, 42, 43],
    "Centro-Oeste": [50, 51, 52, 53],
}
UFS = [uf for estados in ESTADOS_POR_MACRO.values() for uf in estados]
BIRTH_BINS = [1947, 1952, 1957, 1962, 1967, 1972, 1977, 1982, 1988, 1993, 1998, 2003]
RAW_COLUMNS = [
    "Ano", "Trimestre", "Ano_Nascimento", "Idade", "UF", "Sexo", "Cor",
    "Anos_de_Estudo", "Categoria_emprego", "Contribuicao_previdencia",
    "Grupo_atv_princ_empreedimento", "Grupo_ocupacional_no_emprego",
    "Condicao_Ocupacao", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas", "Regiao",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=PROJECT_ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "modelos_salario_hora")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--nivel-confianca", type=float, default=0.95)
    return parser.parse_args()


def normalizar_lote(batch) -> pd.DataFrame:
    df = batch.to_pandas()
    numericas = [
        "Ano", "Trimestre", "Ano_Nascimento", "Idade", "UF", "Sexo", "Cor",
        "Anos_de_Estudo", "Categoria_emprego", "Contribuicao_previdencia",
        "Grupo_atv_princ_empreedimento", "Grupo_ocupacional_no_emprego",
        "Condicao_Ocupacao", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas",
    ]
    for col in numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    nascimento = df["Ano_Nascimento"].where(
        df["Ano_Nascimento"].between(1947, 2003), df["Ano"] - df["Idade"]
    )
    valido = (
        df["Idade"].between(25, 65)
        & df["Condicao_Ocupacao"].eq(1)
        & df["Rendimento_hab_Trab_princ"].gt(0)
        & df["Horas_hab_trabalhadas"].gt(0)
        & df["Anos_de_Estudo"].between(0, 16)
        & df["Regiao"].isin(MACROREGIOES)
        & df["UF"].isin(UFS)
        & df["Sexo"].isin([1, 2])
        & df["Cor"].isin([1, 2, 3, 4, 5])
        & pd.cut(nascimento, bins=BIRTH_BINS, labels=False, include_lowest=True).notna()
    )
    df = df.loc[valido].copy()
    df["y"] = np.log(df["Rendimento_hab_Trab_princ"] / (df["Horas_hab_trabalhadas"] * 4.33))
    birth_band = pd.cut(df["Ano_Nascimento"].where(
        df["Ano_Nascimento"].between(1947, 2003), df["Ano"] - df["Idade"]
    ), bins=BIRTH_BINS, labels=False, include_lowest=True).astype(int)
    # Código inteiro estável para evitar objetos/string durante os acumuladores.
    df["coorte_id"] = df["Sexo"].astype(int) * 1000 + birth_band * 10 + df["Cor"].astype(int)
    df["uf_id"] = df["UF"].astype(int)
    df["inter_id"] = df["coorte_id"] * 100 + df["uf_id"]
    return df


def coluna_binaria(df: pd.DataFrame, serie: pd.Series) -> np.ndarray:
    return serie.to_numpy(dtype=float, na_value=np.nan)


def nomes_features(ufs: list[int]) -> list[str]:
    nomes = ["is_formal", "is_public", "Sexo_Dummy"]
    nomes += list(SETORES.values()) + list(OCUPACOES.values()) + list(CORES.values())
    nomes += [f"Ano_{ano}" for ano in range(2014, 2026)]
    nomes += ["Idade_centrada", "Idade_centrada_2"]
    nomes += [f"UF_{uf}" for uf in ufs[1:]]
    nomes += [f"Trimestre_{tri}" for tri in (2, 3, 4)]
    nomes += [f"Estudo_{nivel}" for nivel in range(1, 17)]
    nomes += ["Estudo_x_is_formal", "Estudo_x_is_public", "Estudo_x_Sexo_Dummy"]
    return nomes


def matriz_m5(df: pd.DataFrame, ufs: list[int]) -> np.ndarray:
    n = len(df)
    colunas: list[np.ndarray] = [np.ones(n, dtype=np.float64)]
    formal = (
        df["Categoria_emprego"].isin([1, 3, 5, 7])
        | (df["Categoria_emprego"].isin([8, 9]) & df["Contribuicao_previdencia"].eq(1))
    ).to_numpy(dtype=float)
    publico = df["Categoria_emprego"].isin([5, 6, 7]).to_numpy(dtype=float)
    sexo = df["Sexo"].eq(2).to_numpy(dtype=float)
    escolaridade = df["Anos_de_Estudo"].to_numpy(dtype=float)
    colunas.extend([formal, publico, sexo])
    for code in SETORES:
        colunas.append(df["Grupo_atv_princ_empreedimento"].eq(code).to_numpy(dtype=float))
    for code in OCUPACOES:
        colunas.append(df["Grupo_ocupacional_no_emprego"].eq(code).to_numpy(dtype=float))
    for code in CORES:
        colunas.append(df["Cor"].eq(code).to_numpy(dtype=float))
    for ano in range(2014, 2026):
        colunas.append(df["Ano"].eq(ano).to_numpy(dtype=float))
    idade = df["Idade"].to_numpy(dtype=float) - 45.0
    colunas.extend([idade, idade ** 2])
    for uf in ufs[1:]:
        colunas.append(df["UF"].eq(uf).to_numpy(dtype=float))
    for tri in (2, 3, 4):
        colunas.append(df["Trimestre"].eq(tri).to_numpy(dtype=float))
    for nivel in range(1, 17):
        colunas.append(df["Anos_de_Estudo"].eq(nivel).to_numpy(dtype=float))
    colunas.extend([escolaridade * formal, escolaridade * publico, escolaridade * sexo])
    return np.column_stack(colunas)


@dataclass
class ResultadoModelo:
    nome: str
    ufs: list[int]
    k: int = 0
    n: int = 0
    xtx: np.ndarray | None = None
    xty: np.ndarray | None = None
    sum_y: float = 0.0
    sum_y2: float = 0.0
    beta: np.ndarray | None = None
    sse: float = 0.0
    meat_hc1: np.ndarray | None = None
    score_uf: np.ndarray | None = None
    score_coorte: np.ndarray | None = None
    score_inter: np.ndarray | None = None
    grupos_uf: dict[int, int] = field(default_factory=dict)
    grupos_coorte: dict[int, int] = field(default_factory=dict)
    grupos_inter: dict[int, int] = field(default_factory=dict)

    def iniciar(self, k: int) -> None:
        self.k = k
        self.xtx = np.zeros((k, k), dtype=np.float64)
        self.xty = np.zeros(k, dtype=np.float64)

    def acumular_normal(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.xtx is None or self.xty is None:
            self.iniciar(x.shape[1])
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.n += len(y)
        self.sum_y += float(y.sum())
        self.sum_y2 += float(y @ y)

    def resolver(self) -> None:
        assert self.xtx is not None and self.xty is not None
        try:
            self.beta = np.linalg.solve(self.xtx, self.xty)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.pinv(self.xtx, rcond=1e-12) @ self.xty
        self.score_uf = np.zeros((len(self.grupos_uf), self.k), dtype=np.float64)
        self.score_coorte = np.zeros((len(self.grupos_coorte), self.k), dtype=np.float64)
        self.score_inter = np.zeros((len(self.grupos_inter), self.k), dtype=np.float64)
        self.meat_hc1 = np.zeros((self.k, self.k), dtype=np.float64)

    def registrar_grupos(self, df: pd.DataFrame) -> None:
        for atributo, destino in (("uf_id", self.grupos_uf), ("coorte_id", self.grupos_coorte), ("inter_id", self.grupos_inter)):
            for valor in pd.unique(df[atributo]):
                valor = int(valor)
                if valor not in destino:
                    destino[valor] = len(destino)

    def acumular_sanduiche(self, x: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> None:
        assert self.beta is not None and self.meat_hc1 is not None
        residual = y - x @ self.beta
        self.sse += float(residual @ residual)
        self.meat_hc1 += x.T @ (x * (residual ** 2)[:, None])
        for atributo, mapa, destino in (
            ("uf_id", self.grupos_uf, self.score_uf),
            ("coorte_id", self.grupos_coorte, self.score_coorte),
            ("inter_id", self.grupos_inter, self.score_inter),
        ):
            assert destino is not None
            # A soma por grupo evita guardar X e resíduos da base inteira.
            ids = df[atributo].to_numpy(dtype=np.int64)
            for valor in np.unique(ids):
                grupo = mapa[int(valor)]
                mask = ids == valor
                destino[grupo] += x[mask].T @ residual[mask]


def iterar_lotes(caminho: Path, batch_size: int):
    parquet = pq.ParquetFile(caminho)
    yield from parquet.iter_batches(columns=RAW_COLUMNS, batch_size=batch_size, use_threads=True)


def construir_modelos() -> dict[str, ResultadoModelo]:
    modelos = {"geral": ResultadoModelo("geral", UFS)}
    for macro, estados in ESTADOS_POR_MACRO.items():
        modelos[macro] = ResultadoModelo(macro, estados)
    return modelos


def subamostra_modelo(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    if nome == "geral":
        return df
    return df.loc[df["Regiao"].eq(nome)]


def cov_cluster(scores: np.ndarray, bread: np.ndarray, n: int, k: int) -> np.ndarray:
    g = scores.shape[0]
    meat = scores.T @ scores
    correcao = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return correcao * (bread @ meat @ bread)


def covariancias(modelo: ResultadoModelo) -> dict[str, np.ndarray]:
    assert modelo.xtx is not None and modelo.meat_hc1 is not None
    bread = np.linalg.pinv(modelo.xtx, rcond=1e-12)
    hc1 = (modelo.n / max(modelo.n - modelo.k, 1)) * (bread @ modelo.meat_hc1 @ bread)
    uf = cov_cluster(modelo.score_uf, bread, modelo.n, modelo.k)
    coorte = cov_cluster(modelo.score_coorte, bread, modelo.n, modelo.k)
    inter = cov_cluster(modelo.score_inter, bread, modelo.n, modelo.k)
    duas_vias = bread @ (modelo.score_uf.T @ modelo.score_uf + modelo.score_coorte.T @ modelo.score_coorte - modelo.score_inter.T @ modelo.score_inter) @ bread
    # Pequenos autovalores negativos podem surgir apenas por arredondamento.
    eigval, eigvec = np.linalg.eigh((duas_vias + duas_vias.T) / 2)
    duas_vias = eigvec @ np.diag(np.clip(eigval, 0, None)) @ eigvec.T
    return {"HC1": hc1, "cluster_UF": uf, "cluster_coorte": coorte, "cluster_2vias": duas_vias}


def pvalor_t(beta: np.ndarray, cov: np.ndarray, graus_liberdade: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t_stat = beta / np.where(se > 0, se, np.nan)
    p_valor = 2 * stats.t.sf(np.abs(t_stat), df=max(int(graus_liberdade), 1))
    critico = stats.t.ppf(0.975, df=max(int(graus_liberdade), 1))
    return se, t_stat, p_valor, critico


def ajustar_bh(p_valores: np.ndarray) -> np.ndarray:
    """Ajusta p-valores pelo procedimento de Benjamini-Hochberg."""
    p = np.asarray(p_valores, dtype=float)
    ajustados = np.full(p.shape, np.nan, dtype=float)
    validos = np.flatnonzero(np.isfinite(p))
    if not len(validos):
        return ajustados
    ordem = validos[np.argsort(p[validos])]
    fatores = p[ordem] * len(ordem) / np.arange(1, len(ordem) + 1)
    fatores = np.minimum.accumulate(fatores[::-1])[::-1]
    ajustados[ordem] = np.minimum(fatores, 1.0)
    return ajustados


def resultados_coeficientes(modelo: ResultadoModelo, nivel_confianca: float) -> pd.DataFrame:
    assert modelo.beta is not None
    covs = covariancias(modelo)
    gl = {
        "HC1": max(modelo.n - modelo.k, 1),
        "cluster_UF": max(len(modelo.grupos_uf) - 1, 1),
        "cluster_coorte": max(len(modelo.grupos_coorte) - 1, 1),
        "cluster_2vias": max(min(len(modelo.grupos_uf), len(modelo.grupos_coorte)) - 1, 1),
    }
    alpha = 1 - nivel_confianca
    nomes = ["Intercept"] + nomes_features(modelo.ufs)
    dados = {"variavel": nomes, "coeficiente": modelo.beta}
    prefixos = {
        "HC1": "hc1",
        "cluster_UF": "cluster_uf",
        "cluster_coorte": "cluster_coorte",
        "cluster_2vias": "cluster_2vias",
    }
    for metodo, cov in covs.items():
        se, t_stat, p_valor, critico = pvalor_t(modelo.beta, cov, gl[metodo])
        prefixo = prefixos[metodo]
        dados[f"erro_padrao_{prefixo}"] = se
        dados[f"t_{prefixo}"] = t_stat
        dados[f"p_valor_{prefixo}"] = p_valor
        dados[f"ic_inferior_{prefixo}"] = modelo.beta - stats.t.ppf(1 - alpha / 2, gl[metodo]) * se
        dados[f"ic_superior_{prefixo}"] = modelo.beta + stats.t.ppf(1 - alpha / 2, gl[metodo]) * se
        dados[f"graus_liberdade_{prefixo}"] = gl[metodo]
    principal = "cluster_2vias"
    dados["significante_5pct_2vias"] = dados["p_valor_cluster_2vias"] < 0.05
    dados["p_valor_cluster_2vias_bh"] = ajustar_bh(dados["p_valor_cluster_2vias"])
    dados["significante_5pct_2vias_bh"] = dados["p_valor_cluster_2vias_bh"] < 0.05
    dados["efeito_percentual_aprox"] = np.where(
        np.arange(len(modelo.beta)) == 0, np.nan, np.expm1(modelo.beta) * 100
    )
    dados["metodo_principal"] = principal
    return pd.DataFrame(dados)


def resumo_modelo(modelo: ResultadoModelo, nivel_confianca: float) -> dict:
    assert modelo.beta is not None
    media_y = modelo.sum_y / modelo.n
    sst = modelo.sum_y2 - 2 * media_y * modelo.sum_y + modelo.n * media_y ** 2
    r2 = 1 - modelo.sse / max(sst, np.finfo(float).eps)
    return {
        "modelo": modelo.nome,
        "n_observacoes_elegiveis": modelo.n,
        "k_regressores_incluindo_intercepto": modelo.k,
        "r2": float(r2),
        "rmse_residuo_log": float(np.sqrt(modelo.sse / max(modelo.n - modelo.k, 1))),
        "clusters_uf": len(modelo.grupos_uf),
        "clusters_coorte": len(modelo.grupos_coorte),
        "clusters_intersecao": len(modelo.grupos_inter),
        "inferência_principal": "OLS com erro-padrão clusterizado em duas vias por coorte e UF",
        "nivel_confianca": nivel_confianca,
        "observacao": "p-valores são associativos e dependem das hipóteses do modelo; não demonstram causalidade.",
    }


def slug(texto: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


def montar_comparacao(modelo: ResultadoModelo, nivel_confianca: float) -> pd.DataFrame:
    assert modelo.beta is not None
    covs = covariancias(modelo)
    linhas = []
    for metodo, cov in covs.items():
        gl = max(modelo.n - modelo.k, 1) if metodo == "HC1" else max(len(modelo.grupos_uf) - 1, 1) if metodo == "cluster_UF" else max(len(modelo.grupos_coorte) - 1, 1) if metodo == "cluster_coorte" else max(min(len(modelo.grupos_uf), len(modelo.grupos_coorte)) - 1, 1)
        se, _, p, _ = pvalor_t(modelo.beta, cov, gl)
        linhas.append({
            "modelo": modelo.nome,
            "estimador_variancia": metodo,
            "n": modelo.n,
            "graus_liberdade_p_valor": gl,
            "media_erro_padrao": float(np.nanmean(se)),
            "n_significantes_5pct": int(np.sum(p < 0.05)),
            "n_coeficientes": len(p),
            "proporcao_significante_5pct": float(np.mean(p < 0.05)),
            "criterio_principal": metodo == "cluster_2vias",
            "nota": "a comparação mostra sensibilidade da inferência; menor erro-padrão não é critério suficiente para escolher um estimador.",
        })
    return pd.DataFrame(linhas)


def executar(args: argparse.Namespace) -> int:
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    modelos = construir_modelos()
    for modelo in modelos.values():
        modelo.iniciar(len(nomes_features(modelo.ufs)) + 1)

    print(f"Primeira passada: acumulando X'X e X'y em lotes de {args.batch_size:,} linhas")
    for batch in iterar_lotes(args.entrada, args.batch_size):
        df = normalizar_lote(batch)
        if df.empty:
            continue
        for nome, modelo in modelos.items():
            parte = subamostra_modelo(df, nome)
            if parte.empty:
                continue
            x = matriz_m5(parte, modelo.ufs)
            modelo.acumular_normal(x, parte["y"].to_numpy(dtype=float))
            modelo.registrar_grupos(parte)
            del x, parte
        del df
    for modelo in modelos.values():
        modelo.resolver()
        print(f"{modelo.nome}: {modelo.n:,} observações elegíveis; {modelo.k} parâmetros")

    print("Segunda passada: resíduos e variâncias robustas, sem guardar a base")
    # Os mapas foram preenchidos na primeira passagem; agora os scores têm tamanho fixo.
    for modelo in modelos.values():
        modelo.score_uf = np.zeros((len(modelo.grupos_uf), modelo.k), dtype=np.float64)
        modelo.score_coorte = np.zeros((len(modelo.grupos_coorte), modelo.k), dtype=np.float64)
        modelo.score_inter = np.zeros((len(modelo.grupos_inter), modelo.k), dtype=np.float64)

    for batch in iterar_lotes(args.entrada, args.batch_size):
        df = normalizar_lote(batch)
        if df.empty:
            continue
        for nome, modelo in modelos.items():
            parte = subamostra_modelo(df, nome)
            if parte.empty:
                continue
            x = matriz_m5(parte, modelo.ufs)
            modelo.acumular_sanduiche(x, parte["y"].to_numpy(dtype=float), parte)
            del x, parte
        del df
        gc.collect()

    todos_resumos = []
    todas_comparacoes = []
    for nome, modelo in modelos.items():
        coef = resultados_coeficientes(modelo, args.nivel_confianca)
        coef.to_csv(args.saida_dir / f"inferencial_m5_{slug(nome)}.csv", index=False)
        resumo = resumo_modelo(modelo, args.nivel_confianca)
        todos_resumos.append(resumo)
        comparacao = montar_comparacao(modelo, args.nivel_confianca)
        todas_comparacoes.append(comparacao)

    comparacoes = pd.concat(todas_comparacoes, ignore_index=True)
    comparacoes.to_csv(args.saida_dir / "comparacao_especificacoes_inferenciais.csv", index=False)
    causalidade = {
        "status": "não identificada",
        "modelo_inferencial": "OLS em log(Salario_Hora) com controles, efeitos fixos de ano e UF e erros-padrão em duas vias",
        "motivos": [
            "a base é uma seção transversal repetida; não há identificador de painel individual documentado",
            "não há instrumento exógeno documentado para escolaridade, formalidade, sexo ou cor",
            "não há experimento natural ou regra de descontinuidade implementada",
            "a base consolidada não contém pesos amostrais PNAD",
            "p-valores medem associação condicional, não efeito causal",
        ],
        "proximo_desenho_necessario": "instrumento ou choque exógeno defensável, hipóteses explícitas e erros compatíveis com o desenho amostral",
    }
    (args.saida_dir / "auditoria_causalidade_m5.json").write_text(json.dumps(causalidade, ensure_ascii=False, indent=2), encoding="utf-8")
    resumo_final = {
        "entrada": str(args.entrada),
        "batch_size": args.batch_size,
        "nivel_confianca": args.nivel_confianca,
        "modelo_principal": "OLS com cluster em duas vias por coorte e UF",
        "modelos": todos_resumos,
        "arquivos_coeficientes": [f"inferencial_m5_{slug(nome)}.csv" for nome in modelos],
        "limitacao_causal": "A estimação fornece significância associativa; não prova causalidade.",
    }
    (args.saida_dir / "resumo_inferencia_m5.json").write_text(json.dumps(resumo_final, ensure_ascii=False, indent=2), encoding="utf-8")
    relatorio = [
        "# Inferência do M5 ampliado",
        "",
        "A estimação foi feita com todas as observações elegíveis da base consolidada, em lotes, para preservar a memória RAM.",
        "",
        "## Especificação escolhida",
        "",
        "A versão principal é OLS para `log(Salario_Hora)`, com escolaridade categórica, idade e idade², formalidade, setor público, sexo, setor, ocupação, cor, UF e efeitos fixos de ano. Os erros-padrão principais são clusterizados em duas vias por coorte e UF.",
        "",
        "HC1, cluster por UF e cluster por coorte foram calculados como análises de sensibilidade. A escolha da variância em duas vias não significa escolher o menor erro-padrão; ela busca refletir dependência simultânea dentro da coorte e da UF.",
        "",
        "## Interpretação",
        "",
        "Um coeficiente com p-valor abaixo de 0,05 rejeita a hipótese nula de coeficiente igual a zero sob as hipóteses do modelo e do estimador de variância. Isso não prova que a variável cause o salário-hora.",
        "",
        "## Causalidade",
        "",
        "A causalidade não foi identificada. A PNAD consolidada disponível não fornece, sozinha, instrumento exógeno, experimento natural ou painel individual identificável; além disso, não contém pesos amostrais. Os resultados devem ser descritos como associações condicionais.",
        "",
        "## Arquivos",
        "",
        "- `inferencial_m5_geral.csv`: estimação geral.",
        "- `inferencial_m5_norte.csv`, `inferencial_m5_nordeste.csv`, `inferencial_m5_sudeste.csv`, `inferencial_m5_sul.csv`, `inferencial_m5_centro_oeste.csv`: estimações por macroregião.",
        "- `comparacao_especificacoes_inferenciais.csv`: sensibilidade dos erros-padrão.",
        "- `auditoria_causalidade_m5.json`: limites para interpretação causal.",
    ]
    (args.saida_dir / "relatorio_inferencia_m5.md").write_text("\n".join(relatorio), encoding="utf-8")
    print(json.dumps(resumo_final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
