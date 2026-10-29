#!/usr/bin/env python3
"""Estima a versão inferencial do M5 ampliado em toda a base.

O objetivo é separar claramente previsão de inferência. O modelo é ponderado para
log(Salario_Hora), com a mesma família de controles do M5 ampliado: idade,
idade ao quadrado, escolaridade categórica, interações da escolaridade,
formalidade, setor público, sexo, setor, ocupação, cor, UF e efeitos fixos de
ano. A estimação usa todas as observações elegíveis do Parquet, em lotes, sem
montar uma matriz com as 9 milhões de linhas em memória.

São comparadas versões de variância:

* HC1: robusta a heterocedasticidade;
* cluster por UF;
* cluster em duas vias por coorte e UF, como sensibilidade;
* linearização por UPA dentro de estrato, escolhida como inferência principal.

Os p-valores são válidos apenas sob as hipóteses do modelo e do estimador de
variância. Eles não provam causalidade. Como a PNAD consolidada disponível é
uma seção transversal repetida, sem instrumento exógeno ou
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
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from scipy.linalg import qr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pnad_pipeline.analise.amostragem import (
    COLUNAS_DESENHO, LinearizacaoUPA, exigir_colunas_desenho, pesos_amostrais, validar_desenho,
)

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
    *COLUNAS_DESENHO,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=PROJECT_ROOT / "data" / "processed" / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "modelos_salario_hora" / "inferencia_amostral")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--nivel-confianca", type=float, default=0.95)
    parser.add_argument("--upa-unica", choices=["erro", "ajustar"], default="erro", help="Tratamento explícito de estratos com uma UPA; ajustar centra na média global dos scores.")
    parser.add_argument("--upa-multiestrato", choices=["erro", "separar"], default="erro", help="Tratamento explícito de UPA reclassificada; separar usa cada combinação UF/estrato/UPA como versão distinta.")
    return parser.parse_args()


def normalizar_lote(batch) -> pd.DataFrame:
    df = batch.to_pandas()
    validar_desenho(df)
    numericas = [
        "Ano", "Trimestre", "Ano_Nascimento", "Idade", "UF", "Sexo", "Cor",
        "Anos_de_Estudo", "Categoria_emprego", "Contribuicao_previdencia",
        "Grupo_atv_princ_empreedimento", "Grupo_ocupacional_no_emprego",
        "Condicao_Ocupacao", "Rendimento_hab_Trab_princ", "Horas_hab_trabalhadas",
        *COLUNAS_DESENHO,
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
        & np.isfinite(df["Rendimento_hab_Trab_princ"])
        & np.isfinite(df["Horas_hab_trabalhadas"])
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
    sum_abs_residuo_log: float = 0.0
    sum_abs_residuo_salario_hora: float = 0.0
    sum_sq_residuo_salario_hora: float = 0.0
    sum_salario_hora: float = 0.0
    sum_salario_hora2: float = 0.0
    meat_hc1: np.ndarray | None = None
    score_uf: np.ndarray | None = None
    score_coorte: np.ndarray | None = None
    score_inter: np.ndarray | None = None
    grupos_uf: dict[int, int] = field(default_factory=dict)
    grupos_coorte: dict[int, int] = field(default_factory=dict)
    grupos_inter: dict[int, int] = field(default_factory=dict)
    desenho: LinearizacaoUPA = field(default_factory=LinearizacaoUPA)
    periodos: set[tuple[int, int]] = field(default_factory=set)
    soma_pesos: float = 0.0
    soma_pesos2: float = 0.0
    bread: np.ndarray | None = None
    indices_ativos: np.ndarray | None = None

    def iniciar(self, k: int) -> None:
        self.k = k
        self.xtx = np.zeros((k, k), dtype=np.float64)
        self.xty = np.zeros(k, dtype=np.float64)

    def acumular_normal(self, x: np.ndarray, y: np.ndarray, pesos: np.ndarray) -> None:
        if self.xtx is None or self.xty is None:
            self.iniciar(x.shape[1])
        self.xtx += x.T @ (x * pesos[:, None])
        self.xty += x.T @ (y * pesos)
        self.n += len(y)
        self.soma_pesos += float(pesos.sum())
        self.soma_pesos2 += float(pesos @ pesos)
        self.sum_y += float(pesos @ y)
        self.sum_y2 += float(pesos @ (y ** 2))

    def resolver(self) -> None:
        assert self.xtx is not None and self.xty is not None
        if not self.n:
            raise ValueError(f"Modelo {self.nome}: nenhuma observação elegível.")
        # Seleciona colunas identificáveis preservando o intercepto. Em uma
        # única onda, dummies de ano/trimestre são constantes ou vazias.
        centrada = self.xtx[1:, 1:] - np.outer(self.xtx[1:, 0], self.xtx[0, 1:]) / self.xtx[0, 0]
        variancias = np.maximum(np.diag(centrada), 0)
        candidatas = np.flatnonzero(variancias > np.maximum(np.diag(self.xtx)[1:], 1) * 1e-12)
        if len(candidatas):
            escala = np.sqrt(variancias[candidatas])
            correlacao = centrada[np.ix_(candidatas, candidatas)] / np.outer(escala, escala)
            _, r, pivots = qr(correlacao, pivoting=True)
            rank = int(np.sum(np.abs(np.diag(r)) > 1e-10))
            selecionadas = candidatas[pivots[:rank]] + 1
        else:
            selecionadas = np.array([], dtype=int)
        self.indices_ativos = np.sort(np.r_[0, selecionadas]).astype(int)
        bloco = self.xtx[np.ix_(self.indices_ativos, self.indices_ativos)]
        escala = np.sqrt(np.diag(bloco))
        inversa = np.linalg.inv(bloco / np.outer(escala, escala)) / np.outer(escala, escala)
        self.bread = np.zeros_like(self.xtx)
        self.bread[np.ix_(self.indices_ativos, self.indices_ativos)] = inversa
        self.beta = self.bread @ self.xty
        self.score_uf = np.zeros((len(self.grupos_uf), self.k), dtype=np.float64)
        self.score_coorte = np.zeros((len(self.grupos_coorte), self.k), dtype=np.float64)
        self.score_inter = np.zeros((len(self.grupos_inter), self.k), dtype=np.float64)
        self.meat_hc1 = np.zeros((self.k, self.k), dtype=np.float64)
        self.desenho.iniciar_scores(self.k)

    def registrar_grupos(self, df: pd.DataFrame) -> None:
        for atributo, destino in (("uf_id", self.grupos_uf), ("coorte_id", self.grupos_coorte), ("inter_id", self.grupos_inter)):
            for valor in pd.unique(df[atributo]):
                valor = int(valor)
                if valor not in destino:
                    destino[valor] = len(destino)

    def acumular_sanduiche(self, x: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> None:
        assert self.beta is not None and self.meat_hc1 is not None
        pred_log = x @ self.beta
        residual = y - pred_log
        salario_hora = np.exp(np.clip(y, -50, 50))
        pred_salario_hora = np.exp(np.clip(pred_log, -50, 50))
        erro_salario_hora = salario_hora - pred_salario_hora
        w = pesos_amostrais(df)
        score_residual = w * residual
        self.sse += float(w @ (residual ** 2))
        self.sum_abs_residuo_log += float(w @ np.abs(residual))
        self.sum_abs_residuo_salario_hora += float(w @ np.abs(erro_salario_hora))
        self.sum_sq_residuo_salario_hora += float(w @ (erro_salario_hora ** 2))
        self.sum_salario_hora += float(w @ salario_hora)
        self.sum_salario_hora2 += float(w @ (salario_hora ** 2))
        self.meat_hc1 += x.T @ (x * (score_residual ** 2)[:, None])
        self.desenho.acumular(df, x * score_residual[:, None])
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
                destino[grupo] += x[mask].T @ score_residual[mask]


def iterar_lotes(caminho: Path, batch_size: int):
    parquet = pq.ParquetFile(caminho)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") == b"false":
        raise ValueError("A inferência requer a base consolidada completa, anterior ao filtro de idade; use a saída de 01_coletar_pnad.py.")
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
    assert modelo.bread is not None and modelo.indices_ativos is not None
    bread = modelo.bread
    k = len(modelo.indices_ativos)
    hc1 = (modelo.n / max(modelo.n - k, 1)) * (bread @ modelo.meat_hc1 @ bread)
    uf = cov_cluster(modelo.score_uf, bread, modelo.n, k)
    coorte = cov_cluster(modelo.score_coorte, bread, modelo.n, k)
    inter = cov_cluster(modelo.score_inter, bread, modelo.n, k)
    duas_vias = bread @ (modelo.score_uf.T @ modelo.score_uf + modelo.score_coorte.T @ modelo.score_coorte - modelo.score_inter.T @ modelo.score_inter) @ bread
    # Pequenos autovalores negativos podem surgir apenas por arredondamento.
    eigval, eigvec = np.linalg.eigh((duas_vias + duas_vias.T) / 2)
    duas_vias = eigvec @ np.diag(np.clip(eigval, 0, None)) @ eigvec.T
    return {"desenho_pnad": modelo.desenho.covariancia(bread), "HC1": hc1, "cluster_UF": uf, "cluster_coorte": coorte, "cluster_2vias": duas_vias}


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
        "desenho_pnad": modelo.desenho.graus_liberdade,
        "HC1": max(modelo.n - modelo.k, 1),
        "cluster_UF": max(len(modelo.grupos_uf) - 1, 1),
        "cluster_coorte": max(len(modelo.grupos_coorte) - 1, 1),
        "cluster_2vias": max(min(len(modelo.grupos_uf), len(modelo.grupos_coorte)) - 1, 1),
    }
    alpha = 1 - nivel_confianca
    nomes = ["Intercept"] + nomes_features(modelo.ufs)
    identificavel = np.isin(np.arange(modelo.k), modelo.indices_ativos)
    dados = {"variavel": nomes, "coeficiente": np.where(identificavel, modelo.beta, np.nan), "identificavel": identificavel}
    prefixos = {
        "desenho_pnad": "desenho_pnad",
        "HC1": "hc1",
        "cluster_UF": "cluster_uf",
        "cluster_coorte": "cluster_coorte",
        "cluster_2vias": "cluster_2vias",
    }
    for metodo, cov in covs.items():
        se, t_stat, p_valor, critico = pvalor_t(modelo.beta, cov, gl[metodo])
        se = np.where(identificavel, se, np.nan)
        t_stat = np.where(identificavel, t_stat, np.nan)
        p_valor = np.where(identificavel, p_valor, np.nan)
        prefixo = prefixos[metodo]
        dados[f"erro_padrao_{prefixo}"] = se
        dados[f"t_{prefixo}"] = t_stat
        dados[f"p_valor_{prefixo}"] = p_valor
        dados[f"ic_inferior_{prefixo}"] = np.where(identificavel, modelo.beta - stats.t.ppf(1 - alpha / 2, gl[metodo]) * se, np.nan)
        dados[f"ic_superior_{prefixo}"] = np.where(identificavel, modelo.beta + stats.t.ppf(1 - alpha / 2, gl[metodo]) * se, np.nan)
        dados[f"graus_liberdade_{prefixo}"] = gl[metodo]
    principal = "desenho_pnad"
    dados["significante_5pct_principal"] = dados["p_valor_desenho_pnad"] < 0.05
    dados["p_valor_principal_bh"] = ajustar_bh(dados["p_valor_desenho_pnad"])
    dados["significante_5pct_principal_bh"] = dados["p_valor_principal_bh"] < 0.05
    dados["significante_5pct_2vias"] = dados["p_valor_cluster_2vias"] < 0.05
    dados["p_valor_cluster_2vias_bh"] = ajustar_bh(dados["p_valor_cluster_2vias"])
    dados["significante_5pct_2vias_bh"] = dados["p_valor_cluster_2vias_bh"] < 0.05
    dados["efeito_percentual_aprox"] = np.where(
        (np.arange(len(modelo.beta)) == 0) | ~identificavel, np.nan, np.expm1(modelo.beta) * 100
    )
    dados["metodo_principal"] = principal
    return pd.DataFrame(dados)


def resultados_perfil_idade(modelo: ResultadoModelo, nivel_confianca: float) -> pd.DataFrame:
    """Predições por idade no perfil médio ponderado, com covariância do desenho."""
    assert modelo.xtx is not None and modelo.beta is not None
    assert modelo.indices_ativos is not None
    nomes = ["Intercept"] + nomes_features(modelo.ufs)
    idade_idx = nomes.index("Idade_centrada")
    idade2_idx = nomes.index("Idade_centrada_2")
    ativos = set(modelo.indices_ativos.tolist())
    if idade_idx not in ativos or idade2_idx not in ativos:
        raise ValueError(f"Modelo {modelo.nome}: os termos de idade não são identificáveis.")

    cov = covariancias(modelo)["desenho_pnad"]
    gl = max(int(modelo.desenho.graus_liberdade), 1)
    critico = float(stats.t.ppf((1 + nivel_confianca) / 2, df=gl))
    perfil_medio = modelo.xtx[0, :] / modelo.xtx[0, 0]
    linhas = []
    for idade in range(25, 66):
        x = perfil_medio.copy()
        x[idade_idx] = idade - 45.0
        x[idade2_idx] = (idade - 45.0) ** 2
        log_previsto = float(x @ modelo.beta)
        variancia = max(float(x @ cov @ x), 0.0)
        erro_padrao = float(np.sqrt(variancia))
        limite_log_inferior = log_previsto - critico * erro_padrao
        limite_log_superior = log_previsto + critico * erro_padrao
        linhas.append({
            "idade": idade,
            "log_previsto": log_previsto,
            "erro_padrao_log": erro_padrao,
            "salario_hora_previsto": float(np.exp(np.clip(log_previsto, -50, 50))),
            "ic_inferior": float(np.exp(np.clip(limite_log_inferior, -50, 50))),
            "ic_superior": float(np.exp(np.clip(limite_log_superior, -50, 50))),
            "nivel_confianca": nivel_confianca,
            "graus_liberdade_desenho": gl,
        })
    return pd.DataFrame(linhas)


def resumo_modelo(modelo: ResultadoModelo, nivel_confianca: float) -> dict:
    assert modelo.beta is not None
    media_y = modelo.sum_y / modelo.soma_pesos
    sst = modelo.sum_y2 - 2 * media_y * modelo.sum_y + modelo.soma_pesos * media_y ** 2
    r2 = 1 - modelo.sse / max(sst, np.finfo(float).eps)
    media_salario_hora = modelo.sum_salario_hora / modelo.soma_pesos
    sst_salario_hora = (
        modelo.sum_salario_hora2
        - 2 * media_salario_hora * modelo.sum_salario_hora
        + modelo.soma_pesos * media_salario_hora ** 2
    )
    return {
        "modelo": modelo.nome,
        "n_observacoes_elegiveis": modelo.n,
        "k_regressores_incluindo_intercepto": modelo.k,
        "k_parametros_identificaveis": len(modelo.indices_ativos),
        "r2": float(r2),
        "rmse_residuo_log": float(np.sqrt(modelo.sse / modelo.soma_pesos)),
        "mae_log": float(modelo.sum_abs_residuo_log / modelo.soma_pesos),
        "mae_salario_hora": float(modelo.sum_abs_residuo_salario_hora / modelo.soma_pesos),
        "rmse_salario_hora": float(np.sqrt(modelo.sum_sq_residuo_salario_hora / modelo.soma_pesos)),
        "r2_salario_hora": float(1 - modelo.sum_sq_residuo_salario_hora / max(sst_salario_hora, np.finfo(float).eps)),
        "clusters_uf": len(modelo.grupos_uf),
        "clusters_coorte": len(modelo.grupos_coorte),
        "clusters_intersecao": len(modelo.grupos_inter),
        "inferência_principal": "Regressão ponderada por V1028 com linearização por UPA dentro de estrato",
        "metodo_principal": "desenho_pnad",
        "peso_amostral": "V1028",
        "n_upas_desenho": len(modelo.desenho.grupos),
        "n_estratos_desenho": len(modelo.desenho.estratos),
        "graus_liberdade_desenho": modelo.desenho.graus_liberdade,
        "estratos_upa_unica": sum(len(ids) == 1 for ids in modelo.desenho.estratos.values()),
        "politica_upa_unica": modelo.desenho.politica_upa_unica,
        "politica_upa_multiestrato": modelo.desenho.politica_upa_multiestrato,
        "upas_multiestrato": [
            {"UF": uf, "UPA": upa, "estratos": estratos}
            for (uf, upa), estratos in sorted(modelo.desenho.upas_multiestrato.items())
        ],
        "n_periodos": len(modelo.periodos),
        "soma_pesos_pessoa_trimestre": modelo.soma_pesos,
        "n_efetivo_kish": modelo.soma_pesos ** 2 / modelo.soma_pesos2,
        "limitacao_temporal": "UPAs recorrentes permanecem agrupadas enquanto mantêm o estrato; reclassificações separadas não têm covariância entre versões estimada. A rotação de domicílios entre trimestres e a renovação gradual da Amostra Mestra iniciada em 2025T3 limitam a aproximação temporal. A soma dos pesos não conta pessoas únicas.",
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
        gl = modelo.desenho.graus_liberdade if metodo == "desenho_pnad" else max(modelo.n - modelo.k, 1) if metodo == "HC1" else max(len(modelo.grupos_uf) - 1, 1) if metodo == "cluster_UF" else max(len(modelo.grupos_coorte) - 1, 1) if metodo == "cluster_coorte" else max(min(len(modelo.grupos_uf), len(modelo.grupos_coorte)) - 1, 1)
        se, _, p, _ = pvalor_t(modelo.beta, cov, gl)
        linhas.append({
            "modelo": modelo.nome,
            "estimador_variancia": metodo,
            "n": modelo.n,
            "graus_liberdade_p_valor": gl,
            "media_erro_padrao": float(np.nanmean(se)),
            "n_significantes_5pct": int(np.sum(p < 0.05)),
            "n_coeficientes": int(np.isfinite(p).sum()),
            "proporcao_significante_5pct": float(np.sum(p < 0.05) / max(np.isfinite(p).sum(), 1)),
            "criterio_principal": metodo == "desenho_pnad",
            "nota": "a comparação mostra sensibilidade da inferência; menor erro-padrão não é critério suficiente para escolher um estimador.",
        })
    return pd.DataFrame(linhas)


def executar(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or not 0 < args.nivel_confianca < 1:
        raise ValueError("Lote deve ser positivo e nível de confiança deve estar entre 0 e 1.")
    exigir_colunas_desenho(pq.ParquetFile(args.entrada).schema_arrow.names)
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    modelos = construir_modelos()
    for modelo in modelos.values():
        modelo.desenho.politica_upa_unica = getattr(args, "upa_unica", "erro")
        modelo.desenho.politica_upa_multiestrato = getattr(args, "upa_multiestrato", "erro")
        modelo.iniciar(len(nomes_features(modelo.ufs)) + 1)

    print(f"Primeira passada: acumulando X'WX e X'Wy em lotes de {args.batch_size:,} linhas")
    for batch in iterar_lotes(args.entrada, args.batch_size):
        # O cadastro usa toda a amostra, antes dos filtros de renda e idade.
        amostra = batch.to_pandas()
        for nome, modelo in modelos.items():
            cadastro = subamostra_modelo(amostra, nome)
            modelo.desenho.registrar(cadastro)
            modelo.periodos.update((int(a), int(t)) for a, t in cadastro[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        del amostra, cadastro
        df = normalizar_lote(batch)
        if df.empty:
            continue
        for nome, modelo in modelos.items():
            parte = subamostra_modelo(df, nome)
            if parte.empty:
                continue
            x = matriz_m5(parte, modelo.ufs)
            modelo.acumular_normal(x, parte["y"].to_numpy(dtype=float), pesos_amostrais(parte))
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
        perfil = resultados_perfil_idade(modelo, args.nivel_confianca)
        perfil.to_csv(args.saida_dir / f"perfil_idade_m5_{slug(nome)}.csv", index=False)
        resumo = resumo_modelo(modelo, args.nivel_confianca)
        todos_resumos.append(resumo)
        comparacao = montar_comparacao(modelo, args.nivel_confianca)
        todas_comparacoes.append(comparacao)

    comparacoes = pd.concat(todas_comparacoes, ignore_index=True)
    comparacoes.to_csv(args.saida_dir / "comparacao_especificacoes_inferenciais.csv", index=False)
    causalidade = {
        "status": "não identificada",
        "modelo_inferencial": "Regressão ponderada por V1028 em log(Salario_Hora), com controles e linearização por UPA e estrato",
        "motivos": [
            "a base é uma seção transversal repetida; não há identificador de painel individual documentado",
            "não há instrumento exógeno documentado para escolaridade, formalidade, sexo ou cor",
            "não há experimento natural ou regra de descontinuidade implementada",
            "p-valores medem associação condicional, não efeito causal",
        ],
        "proximo_desenho_necessario": "instrumento ou choque exógeno defensável, hipóteses explícitas e erros compatíveis com o desenho amostral",
    }
    (args.saida_dir / "auditoria_causalidade_m5.json").write_text(json.dumps(causalidade, ensure_ascii=False, indent=2), encoding="utf-8")
    resumo_final = {
        "entrada": str(args.entrada),
        "batch_size": args.batch_size,
        "nivel_confianca": args.nivel_confianca,
        "modelo_principal": "Regressão ponderada por V1028 com variância por UPA e estrato",
        "metodo_principal": "desenho_pnad",
        "ponderado": True,
        "politica_upa_unica": getattr(args, "upa_unica", "erro"),
        "politica_upa_multiestrato": getattr(args, "upa_multiestrato", "erro"),
        "upas_multiestrato_desenho_geral": [
            {"UF": uf, "UPA": upa, "estratos": estratos}
            for (uf, upa), estratos in sorted(modelos["geral"].desenho.upas_multiestrato.items())
        ],
        "aproximacao_variancia": "Conglomerado primário com reposição, sem FPC e sem recalibrar os pesos finais. UPAs recorrentes permanecem agrupadas enquanto mantêm o estrato; com upa-multiestrato=separar, cada versão (UF, estrato, UPA) é separada e a covariância entre versões não é estimada. A rotação de domicílios entre trimestres e a renovação gradual da Amostra Mestra iniciada em 2025T3 limitam a aproximação temporal.",
        "modelos": todos_resumos,
        "arquivos_coeficientes": [f"inferencial_m5_{slug(nome)}.csv" for nome in modelos],
        "arquivos_perfil_idade": [f"perfil_idade_m5_{slug(nome)}.csv" for nome in modelos],
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
        "A versão principal é uma regressão ponderada por `V1028` para `log(Salario_Hora)`, com escolaridade categórica, idade e idade², formalidade, setor público, sexo, setor, ocupação, cor, UF e efeitos fixos de ano e trimestre. Os erros-padrão usam linearização por UPA dentro de estrato, com graus de liberdade iguais ao número de UPAs menos o número de estratos.",
        "",
        "HC1, cluster por UF, por coorte e em duas vias foram calculados como sensibilidades ponderadas. Eles não substituem UPA e estrato. O cadastro completo de UPAs é registrado antes da seleção de pessoas ocupadas de 25 a 65 anos com renda e horas positivas.",
        "",
        f"No empilhamento temporal, uma UPA recorrente permanece no mesmo conglomerado enquanto mantém o estrato. A variância é uma aproximação com reposição, sem FPC ou recalibração. Política de UPA reclassificada: {getattr(args, 'upa_multiestrato', 'erro')}. Quando 'separar' é usado, cada combinação (UF, estrato, UPA) é tratada como uma versão distinta; a covariância entre versões não é estimada. UPAs reclassificadas encontradas: {modelos['geral'].desenho.upas_multiestrato}. Política de UPA única nesta execução: {getattr(args, 'upa_unica', 'erro')}. A opção ajustar centra o score na média de todas as UPAs, sem presumir variância nula. A PNAD revisita domicílios por trimestres consecutivos e iniciou a renovação gradual da Amostra Mestra em 2025T3; essa dependência temporal limita a aproximação empilhada. A soma de V1028 representa pessoas-trimestres, não pessoas únicas; o tamanho efetivo de Kish é apenas um diagnóstico dos pesos.",
        "",
        "## Interpretação",
        "",
        "Um coeficiente com p-valor abaixo de 0,05 rejeita a hipótese nula de coeficiente igual a zero sob as hipóteses do modelo e do estimador de variância. Isso não prova que a variável cause o salário-hora.",
        "",
        "## Causalidade",
        "",
        "A causalidade não foi identificada. A ponderação melhora a representação do domínio analisado, mas não fornece instrumento exógeno, experimento natural ou identificação causal. Os resultados descrevem associações condicionais.",
        "",
        "## Arquivos",
        "",
        "- `inferencial_m5_geral.csv`: estimação geral.",
        "- `inferencial_m5_norte.csv`, `inferencial_m5_nordeste.csv`, `inferencial_m5_sudeste.csv`, `inferencial_m5_sul.csv`, `inferencial_m5_centro_oeste.csv`: estimações por macroregião.",
        "- `perfil_idade_m5_*.csv`: salário-hora previsto por idade no perfil médio ponderado, com IC95% pela covariância do desenho.",
        "- `comparacao_especificacoes_inferenciais.csv`: sensibilidade dos erros-padrão.",
        "- `auditoria_causalidade_m5.json`: limites para interpretação causal.",
    ]
    (args.saida_dir / "relatorio_inferencia_m5.md").write_text("\n".join(relatorio), encoding="utf-8")
    print(json.dumps(resumo_final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
