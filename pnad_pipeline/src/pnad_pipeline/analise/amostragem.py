"""Ponderação e linearização por UPA para os microdados trimestrais.

V1028 é o peso final calibrado, não uma frequência nem uma precisão individual.
A variância usa a aproximação com reposição do conglomerado primário, sem FPC
ou nova calibração. O cadastro de UPAs deve ser registrado ANTES de selecionar
o domínio: UPAs sem observações elegíveis contribuem com score zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUNAS_DESENHO = ["V1028", "Estrato", "UPA"]
COLUNAS_PERIODO = ["Ano", "Trimestre"]
ORIENTACAO_RECOLETA = (
    "Reexecute scripts/01_coletar_pnad.py para obter V1028, Estrato e UPA "
    "dos microdados, depois scripts/02_tratar_pnad.py. "
    "Pesos e estratos não podem ser reconstruídos a partir da base antiga."
)


def exigir_colunas_desenho(colunas) -> None:
    faltantes = sorted(set(COLUNAS_DESENHO).difference(colunas))
    if faltantes:
        raise ValueError(f"Desenho amostral incompleto: faltam {faltantes}. {ORIENTACAO_RECOLETA}")


def pesos_amostrais(df: pd.DataFrame) -> np.ndarray:
    if "V1028" not in df:
        raise ValueError(f"Peso final V1028 ausente. {ORIENTACAO_RECOLETA}")
    pesos = pd.to_numeric(df["V1028"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    invalidos = ~np.isfinite(pesos) | (pesos <= 0)
    if invalidos.any():
        raise ValueError(f"V1028 contém {int(invalidos.sum())} pesos ausentes, não finitos ou não positivos.")
    return pesos


def validar_desenho(df: pd.DataFrame) -> None:
    exigir_colunas_desenho(df.columns)
    if "UF" not in df:
        raise ValueError("UF ausente: necessária para identificar estratos e UPAs.")
    pesos_amostrais(df)
    for coluna in ["UF", "Estrato", "UPA"]:
        valores = pd.to_numeric(df[coluna], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if (~np.isfinite(valores) | (valores <= 0) | (valores != np.floor(valores))).any():
            raise ValueError(f"{coluna} contém identificadores amostrais inválidos.")


def media_ponderada(df: pd.DataFrame, coluna: str) -> float:
    pesos = pesos_amostrais(df)
    valores = pd.to_numeric(df[coluna], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    validos = np.isfinite(valores)
    return float(np.average(valores[validos], weights=pesos[validos])) if validos.any() else np.nan


def quantis_ponderados(df: pd.DataFrame, coluna: str, quantis) -> np.ndarray:
    """Inversa da distribuição empírica ponderada; não interpola rendimentos."""
    q = np.asarray(quantis, dtype=float)
    if (~np.isfinite(q) | (q < 0) | (q > 1)).any():
        raise ValueError("Quantis devem estar no intervalo [0, 1].")
    pesos = pesos_amostrais(df)
    valores = pd.to_numeric(df[coluna], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    validos = np.isfinite(valores)
    if not validos.any():
        return np.full(q.shape, np.nan)
    ordem = np.argsort(valores[validos], kind="stable")
    valores, pesos = valores[validos][ordem], pesos[validos][ordem]
    acumulados = np.cumsum(pesos)
    indices = np.searchsorted(acumulados, q * acumulados[-1], side="left")
    return valores[np.minimum(indices, len(valores) - 1)]


def diagnostico_representatividade(df: pd.DataFrame) -> pd.DataFrame:
    """Contagens do domínio analítico, separadas dos totais ponderados trimestrais."""
    validar_desenho(df)
    linhas = []
    for (ano, trimestre), sub in df.groupby(COLUNAS_PERIODO, observed=True, sort=True):
        w = pesos_amostrais(sub)
        linhas.append({
            "Ano": ano, "Trimestre": trimestre, "n_amostral": len(sub),
            "total_pessoas_dominio": float(w.sum()),
            "n_efetivo_kish": float(w.sum() ** 2 / (w @ w)),
            "n_upas_dominio": len(sub[["UF", "Estrato", "UPA"]].drop_duplicates()),
            "n_estratos_dominio": len(sub[["UF", "Estrato"]].drop_duplicates()),
        })
    return pd.DataFrame(linhas)


@dataclass
class LinearizacaoUPA:
    """Acumula scores ponderados; mantém a mesma UPA unida entre trimestres.

    Para bases empilhadas, esta é uma aproximação de sanduíche com estratos
    estáveis e dependência dentro da UPA ao longo do tempo. Não cria uma UPA
    independente a cada entrevista. Mudanças de estrato da mesma UPA provocam
    erro por padrão. A opção explícita 'separar' trata cada combinação
    (UF, estrato, UPA) como uma versão de conglomerado distinta; isso não estima
    covariância entre as versões. Estratos com uma única UPA também provocam
    erro por padrão. A opção 'ajustar' centra seu score na média de todas as
    UPAs, como lonely.psu=adjust do pacote survey. Não presume seleção com
    certeza nem variância nula.
    """

    grupos: dict[tuple[int, int, int], int] = field(default_factory=dict)
    estratos: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    estratos_por_upa: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    scores: np.ndarray | None = None
    politica_upa_unica: str = "erro"
    politica_upa_multiestrato: str = "erro"

    def __post_init__(self):
        if self.politica_upa_unica not in {"erro", "ajustar"}:
            raise ValueError("Política de UPA única deve ser 'erro' ou 'ajustar'.")
        if self.politica_upa_multiestrato not in {"erro", "separar"}:
            raise ValueError("Política de UPA em múltiplos estratos deve ser 'erro' ou 'separar'.")

    @property
    def upas_multiestrato(self) -> dict[tuple[int, int], list[int]]:
        """UPAs reclassificadas e seus códigos de estrato observados."""
        return {
            chave: sorted(estratos)
            for chave, estratos in self.estratos_por_upa.items()
            if len(estratos) > 1
        }

    def registrar(self, df: pd.DataFrame) -> None:
        validar_desenho(df)
        cadastro = df[["UF", "Estrato", "UPA"]].drop_duplicates().apply(pd.to_numeric).astype("int64")
        for uf, estrato, upa in cadastro.itertuples(index=False, name=None):
            uf, estrato, upa = int(uf), int(estrato), int(upa)
            estratos = self.estratos_por_upa.setdefault((uf, upa), set())
            estratos.add(estrato)
            if len(estratos) > 1 and self.politica_upa_multiestrato == "erro":
                raise ValueError(f"UPA {upa} da UF {uf} mudou de estrato; analise separadamente as versões do desenho.")
            chave = (uf, estrato, upa)
            if chave not in self.grupos:
                if self.scores is not None:
                    raise RuntimeError("O cadastro de UPAs deve ser concluído antes de acumular scores.")
                self.grupos[chave] = len(self.grupos)
                self.estratos.setdefault((uf, estrato), []).append(self.grupos[chave])

    def iniciar_scores(self, k: int) -> None:
        self.scores = np.zeros((len(self.grupos), k), dtype=np.float64)

    def acumular(self, df: pd.DataFrame, scores: np.ndarray) -> None:
        if self.scores is None:
            raise RuntimeError("Inicialize os scores depois de registrar todo o desenho.")
        # Agrupa apenas as chaves distintas do lote; não percorre a base por UPA.
        chaves = pd.MultiIndex.from_frame(df[["UF", "Estrato", "UPA"]])
        codigos, unicas = pd.factorize(chaves, sort=False)
        locais = np.zeros((len(unicas), self.scores.shape[1]), dtype=float)
        np.add.at(locais, codigos, scores)
        ids = [self.grupos[tuple(int(v) for v in chave)] for chave in unicas]
        self.scores[ids] += locais

    @property
    def graus_liberdade(self) -> int:
        return len(self.grupos) - len(self.estratos)

    def covariancia(self, bread: np.ndarray) -> np.ndarray:
        if self.scores is None:
            raise RuntimeError("Scores não inicializados.")
        meat = np.zeros_like(bread)
        media_global = self.scores.mean(axis=0) if len(self.scores) else np.zeros(bread.shape[0])
        for chave, ids in self.estratos.items():
            m = len(ids)
            if m < 2:
                if self.politica_upa_unica == "erro":
                    raise ValueError(f"Estrato {chave} tem uma única UPA; a variância não é identificável sem tratamento explícito.")
                centrado = self.scores[ids[0]] - media_global
                meat += np.outer(centrado, centrado)
                continue
            scores = self.scores[ids]
            centrados = scores - scores.mean(axis=0)
            meat += (m / (m - 1)) * (centrados.T @ centrados)
        if self.graus_liberdade <= 0:
            raise ValueError("Desenho sem graus de liberdade para inferência.")
        cov = bread @ meat @ bread
        return (cov + cov.T) / 2
