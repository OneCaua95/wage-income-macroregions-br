"""Estima retorno associado à escolaridade com o desenho da PNAD.

Compara escolaridade categórica, linear e spline linear por segmentos. Também
estima perfis flexíveis por sexo, cor/raça e macrorregião. Os modelos usam a
mesma população analítica, controles, V1028 e linearização por UPA/estrato.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import qr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

M5_SPEC = importlib.util.spec_from_file_location("_pnad_inferencia_m5_helpers", SCRIPT_DIR / "08_inferencia_m5.py")
if M5_SPEC is None or M5_SPEC.loader is None:
    raise RuntimeError("Não foi possível carregar os auxiliares da inferência M5.")
M5 = importlib.util.module_from_spec(M5_SPEC)
sys.modules[M5_SPEC.name] = M5
M5_SPEC.loader.exec_module(M5)

from pnad_pipeline.analise.amostragem import LinearizacaoUPA, exigir_colunas_desenho, pesos_amostrais


NOS_SPLINE = (4.0, 8.0, 11.0)
MODELOS = (
    ("categorias", "geral"),
    ("linear", "geral"),
    ("spline", "geral"),
    ("spline", "sexo"),
    ("spline", "cor"),
    ("spline", "regiao"),
)
GRUPOS = {
    "sexo": [(1, "Homem"), (2, "Mulher")],
    "cor": [(1, "Branca"), (2, "Preta"), (3, "Amarela"), (4, "Parda"), (5, "Indígena")],
    "regiao": [("Norte", "Norte"), ("Nordeste", "Nordeste"), ("Sudeste", "Sudeste"), ("Sul", "Sul"), ("Centro-Oeste", "Centro-Oeste")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/retorno_educacao")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--nivel-confianca", type=float, default=0.95)
    parser.add_argument("--upa-unica", choices=["erro", "ajustar"], default="erro")
    parser.add_argument("--upa-multiestrato", choices=["erro", "separar"], default="erro")
    return parser.parse_args()


def nomes_controles() -> list[str]:
    nomes = ["is_formal", "is_public", "Sexo_Dummy"]
    nomes += list(M5.SETORES.values()) + list(M5.OCUPACOES.values()) + list(M5.CORES.values())
    nomes += [f"Ano_{ano}" for ano in range(2014, 2026)]
    nomes += ["Idade_centrada", "Idade_centrada_2"]
    nomes += [f"UF_{uf}" for uf in M5.UFS[1:]]
    nomes += [f"Trimestre_{tri}" for tri in (2, 3, 4)]
    return nomes


def nomes_escolaridade(spec: str) -> list[str]:
    if spec == "categorias":
        return [f"Escolaridade_{nivel}" for nivel in range(1, 17)]
    if spec == "linear":
        return ["Escolaridade_anos"]
    return ["Escolaridade_anos", *(f"Escolaridade_apos_{int(no)}" for no in NOS_SPLINE)]


def nomes_modelo(spec: str, dimensao: str) -> list[str]:
    nomes = ["Intercept", *nomes_controles(), *nomes_escolaridade(spec)]
    if dimensao != "geral":
        grupos = GRUPOS[dimensao]
        for codigo, rotulo in grupos[1:]:
            sufixo = str(codigo) if dimensao != "regiao" else str(codigo).replace("-", "_")
            nomes += [f"{nome}_x_{dimensao}_{sufixo}" for nome in nomes_escolaridade(spec)]
    return nomes


def base_escolaridade(df: pd.DataFrame, spec: str) -> list[np.ndarray]:
    anos = df["Anos_de_Estudo"].to_numpy(dtype=np.float64)
    if spec == "categorias":
        return [(anos == nivel).astype(np.float64) for nivel in range(1, 17)]
    if spec == "linear":
        return [anos]
    return [anos, *(np.maximum(anos - no, 0.0) for no in NOS_SPLINE)]


def matriz_modelo(df: pd.DataFrame, spec: str, dimensao: str) -> np.ndarray:
    n = len(df)
    colunas: list[np.ndarray] = [np.ones(n, dtype=np.float64)]
    formal = (
        df["Categoria_emprego"].isin([1, 3, 5, 7])
        | (df["Categoria_emprego"].isin([8, 9]) & df["Contribuicao_previdencia"].eq(1))
    ).to_numpy(dtype=np.float64)
    publico = df["Categoria_emprego"].isin([5, 6, 7]).to_numpy(dtype=np.float64)
    sexo = df["Sexo"].eq(2).to_numpy(dtype=np.float64)
    colunas.extend([formal, publico, sexo])
    for code in M5.SETORES:
        colunas.append(df["Grupo_atv_princ_empreedimento"].eq(code).to_numpy(dtype=np.float64))
    for code in M5.OCUPACOES:
        colunas.append(df["Grupo_ocupacional_no_emprego"].eq(code).to_numpy(dtype=np.float64))
    for code in M5.CORES:
        colunas.append(df["Cor"].eq(code).to_numpy(dtype=np.float64))
    for ano in range(2014, 2026):
        colunas.append(df["Ano"].eq(ano).to_numpy(dtype=np.float64))
    idade = df["Idade"].to_numpy(dtype=np.float64) - 45.0
    colunas.extend([idade, idade ** 2])
    for uf in M5.UFS[1:]:
        colunas.append(df["UF"].eq(uf).to_numpy(dtype=np.float64))
    for trimestre in (2, 3, 4):
        colunas.append(df["Trimestre"].eq(trimestre).to_numpy(dtype=np.float64))

    edu = base_escolaridade(df, spec)
    colunas.extend(edu)
    if dimensao != "geral":
        for codigo, _ in GRUPOS[dimensao][1:]:
            if dimensao == "regiao":
                indicador = df["Regiao"].eq(codigo).to_numpy(dtype=np.float64)
            elif dimensao == "sexo":
                indicador = df["Sexo"].eq(codigo).to_numpy(dtype=np.float64)
            else:
                indicador = df["Cor"].eq(codigo).to_numpy(dtype=np.float64)
            colunas.extend([termo * indicador for termo in edu])
    return np.column_stack(colunas)


@dataclass
class ModeloEducacao:
    especificacao: str
    dimensao: str
    nomes: list[str]
    n: int = 0
    soma_pesos: float = 0.0
    soma_pesos2: float = 0.0
    xtx: np.ndarray = field(init=False)
    xty: np.ndarray = field(init=False)
    beta: np.ndarray | None = None
    bread: np.ndarray | None = None
    ativos: np.ndarray | None = None
    desenho: LinearizacaoUPA | None = None

    def __post_init__(self):
        k = len(self.nomes)
        self.xtx = np.zeros((k, k), dtype=np.float64)
        self.xty = np.zeros(k, dtype=np.float64)

    def acumular_normal(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.xtx += x.T @ (x * w[:, None])
        self.xty += x.T @ (y * w)
        self.n += len(y)
        self.soma_pesos += float(w.sum())
        self.soma_pesos2 += float(w @ w)

    def resolver(self) -> None:
        if not self.n:
            raise ValueError(f"Modelo {self.especificacao}/{self.dimensao}: nenhuma observação elegível.")
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
        self.ativos = np.sort(np.r_[0, selecionadas]).astype(int)
        bloco = self.xtx[np.ix_(self.ativos, self.ativos)]
        escala = np.sqrt(np.diag(bloco))
        inversa = np.linalg.inv(bloco / np.outer(escala, escala)) / np.outer(escala, escala)
        self.bread = np.zeros_like(self.xtx)
        self.bread[np.ix_(self.ativos, self.ativos)] = inversa
        self.beta = self.bread @ self.xty

    def covariancia(self) -> np.ndarray:
        assert self.bread is not None and self.desenho is not None
        return self.desenho.covariancia(self.bread)


def _wald(modelo: ModeloEducacao, restricoes: np.ndarray, cov: np.ndarray) -> dict:
    assert modelo.beta is not None and modelo.ativos is not None and modelo.desenho is not None
    invalidas = np.any(np.abs(restricoes[:, np.setdiff1d(np.arange(len(modelo.nomes)), modelo.ativos)]) > 1e-12, axis=1) if len(modelo.ativos) < len(modelo.nomes) else np.zeros(len(restricoes), dtype=bool)
    r = restricoes[~invalidas]
    if not len(r):
        return {"f": None, "p_valor": None, "gl_numerador": 0, "gl_denominador": modelo.desenho.graus_liberdade, "restricoes_solicitadas": len(restricoes), "restricoes_estimaveis": 0}
    ra = r[:, modelo.ativos]
    _, triangular, pivots = qr(ra.T, pivoting=True, mode="economic")
    rank = int(np.sum(np.abs(np.diag(triangular)) > 1e-10))
    if not rank:
        return {"f": None, "p_valor": None, "gl_numerador": 0, "gl_denominador": modelo.desenho.graus_liberdade, "restricoes_solicitadas": len(restricoes), "restricoes_estimaveis": 0}
    ra = ra[pivots[:rank]]
    v = cov[np.ix_(modelo.ativos, modelo.ativos)]
    meio = ra @ v @ ra.T
    delta = ra @ modelo.beta[modelo.ativos]
    fstat = float(delta @ np.linalg.pinv(meio, rcond=1e-10) @ delta / rank)
    gl2 = max(int(modelo.desenho.graus_liberdade), 1)
    return {"f": fstat, "p_valor": float(stats.f.sf(fstat, rank, gl2)), "gl_numerador": rank, "gl_denominador": gl2,
            "restricoes_solicitadas": len(restricoes), "restricoes_estimaveis": rank}


def restricoes_indices(modelo: ModeloEducacao, names: list[str]) -> np.ndarray:
    r = np.zeros((len(names), len(modelo.nomes)), dtype=np.float64)
    for i, name in enumerate(names):
        r[i, modelo.nomes.index(name)] = 1.0
    return r


def testes_modelo(modelo: ModeloEducacao, cov: np.ndarray) -> dict[str, dict]:
    if modelo.especificacao == "categorias" and modelo.dimensao == "geral":
        terms = [f"Escolaridade_{k}" for k in range(1, 17)]
        joint = _wald(modelo, restricoes_indices(modelo, terms), cov)
        linearidade = np.zeros((15, len(modelo.nomes)), dtype=np.float64)
        for i, nivel in enumerate(range(2, 17)):
            linearidade[i, modelo.nomes.index(f"Escolaridade_{nivel}")] = 1.0
            linearidade[i, modelo.nomes.index("Escolaridade_1")] = -float(nivel)
        return {"associacao_escolaridade": joint, "desvio_da_forma_linear": _wald(modelo, linearidade, cov)}
    if modelo.especificacao == "linear" and modelo.dimensao == "geral":
        term = restricoes_indices(modelo, ["Escolaridade_anos"])
        return {"associacao_escolaridade": _wald(modelo, term, cov)}
    if modelo.dimensao == "geral":
        all_edu = nomes_escolaridade("spline")
        hinge = [f"Escolaridade_apos_{int(no)}" for no in NOS_SPLINE]
        return {"associacao_escolaridade": _wald(modelo, restricoes_indices(modelo, all_edu), cov),
                "componentes_nao_lineares": _wald(modelo, restricoes_indices(modelo, hinge), cov)}
    interacoes = [name for name in modelo.nomes if "_x_" in name]
    return {f"heterogeneidade_{modelo.dimensao}": _wald(modelo, restricoes_indices(modelo, interacoes), cov)}


def _grupo_bases(dimensao: str) -> list[tuple[str, str, int | str]]:
    if dimensao == "geral":
        return [("geral", "Brasil", 0)]
    return [(str(codigo), rotulo, codigo) for codigo, rotulo in GRUPOS[dimensao]]


def _contraste_escolaridade(modelo: ModeloEducacao, anos: int, grupo_codigo: int | str, marginal: bool = False) -> np.ndarray:
    c = np.zeros(len(modelo.nomes), dtype=np.float64)
    if modelo.especificacao == "categorias":
        if anos:
            c[modelo.nomes.index(f"Escolaridade_{anos}")] = 1.0
        if marginal and anos > 1:
            c[modelo.nomes.index(f"Escolaridade_{anos - 1}")] -= 1.0
        return c
    def base(nivel: int) -> list[float]:
        return [float(nivel)] if modelo.especificacao == "linear" else [float(nivel), *(max(float(nivel) - no, 0.0) for no in NOS_SPLINE)]
    atual = base(anos)
    anterior = base(anos - 1) if marginal else [0.0] * len(atual)
    bases = [valor - base_anterior for valor, base_anterior in zip(atual, anterior)]
    for name, value in zip(nomes_escolaridade(modelo.especificacao), bases):
        c[modelo.nomes.index(name)] = value
        if modelo.dimensao != "geral":
            codigo_base = GRUPOS[modelo.dimensao][0][0]
            if grupo_codigo != codigo_base:
                suf = str(grupo_codigo) if modelo.dimensao != "regiao" else str(grupo_codigo).replace("-", "_")
                interaction = f"{name}_x_{modelo.dimensao}_{suf}"
                c[modelo.nomes.index(interaction)] = value
    return c


def _percentual(beta: float) -> float:
    return float(100.0 * np.expm1(beta))


def efeitos(modelos: dict[tuple[str, str], ModeloEducacao], nivel_confianca: float) -> pd.DataFrame:
    linhas: list[dict] = []
    alpha = 1 - nivel_confianca
    for (spec, dimensao), modelo in modelos.items():
        assert modelo.beta is not None and modelo.ativos is not None and modelo.desenho is not None
        cov = modelo.covariancia()
        df = max(int(modelo.desenho.graus_liberdade), 1)
        critico = float(stats.t.ppf(1 - alpha / 2, df))
        for grupo_codigo, grupo, raw_codigo in _grupo_bases(dimensao):
            for tipo in ("acumulado", "marginal"):
                for anos in range(1, 17):
                    marginal = tipo == "marginal"
                    c = _contraste_escolaridade(modelo, anos, raw_codigo, marginal=marginal)
                    precisa = np.flatnonzero(np.abs(c) > 1e-12)
                    estimavel = not np.any(~np.isin(precisa, modelo.ativos))
                    beta = float(c @ modelo.beta) if estimavel else np.nan
                    se = float(np.sqrt(max(float(c @ cov @ c), 0.0))) if estimavel else np.nan
                    lo, hi = (beta - critico * se, beta + critico * se) if estimavel else (np.nan, np.nan)
                    p = float(2 * stats.t.sf(abs(beta / se), df)) if estimavel and se > 0 else np.nan
                    nivel_rotulo = "16 anos ou mais" if anos == 16 else f"{anos} ano" if anos == 1 else f"{anos} anos"
                    contraste_rotulo = f"{nivel_rotulo} vs {anos - 1} anos" if marginal else f"{nivel_rotulo} vs código 0"
                    linhas.append({
                        "especificacao": spec if dimensao == "geral" else f"spline_{dimensao}",
                        "dimensao": dimensao, "grupo_codigo": grupo_codigo, "grupo": grupo,
                        "contraste": tipo, "contraste_rotulo": contraste_rotulo,
                        "anos_de_estudo_codigo": anos, "anos_de_estudo_rotulo": nivel_rotulo,
                        "referencia": f"Nível imediatamente anterior ({anos - 1} anos)" if marginal else "Código 0: sem instrução ou menos de 1 ano",
                        "beta_log": beta if estimavel else None, "erro_padrao": se if estimavel else None,
                        "ic_inferior_log": lo if estimavel else None, "ic_superior_log": hi if estimavel else None,
                        "associacao_pct": _percentual(beta) if estimavel else None,
                        "ic_inferior_pct": _percentual(lo) if estimavel else None,
                        "ic_superior_pct": _percentual(hi) if estimavel else None,
                        "p_valor": p if np.isfinite(p) else None, "graus_liberdade": df,
                        "identificavel": bool(estimavel),
                    })
    out = pd.DataFrame(linhas)
    out["p_valor_bh"] = np.nan
    for _, ids in out.groupby(["especificacao", "dimensao", "grupo_codigo", "contraste"], dropna=False).groups.items():
        indices = list(ids)
        p = out.loc[indices, "p_valor"].to_numpy(dtype=float)
        out.loc[indices, "p_valor_bh"] = M5.ajustar_bh(p)
    return out


def executar(args: argparse.Namespace) -> int:
    if args.batch_size <= 0 or not 0 < args.nivel_confianca < 1:
        raise ValueError("Lote deve ser positivo e nível de confiança deve estar entre 0 e 1.")
    parquet = M5.pq.ParquetFile(args.entrada)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") == b"false":
        raise ValueError("Use a base consolidada completa, anterior ao filtro de idade.")
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    modelos = {
        (spec, dimensao): ModeloEducacao(spec, dimensao, nomes_modelo(spec, dimensao))
        for spec, dimensao in MODELOS
    }
    desenho_base = LinearizacaoUPA(
        politica_upa_unica=args.upa_unica,
        politica_upa_multiestrato=args.upa_multiestrato,
    )
    periodos: set[tuple[int, int]] = set()

    print(f"Primeira passada: acumulando os modelos em lotes de {args.batch_size:,} linhas")
    for batch in M5.iterar_lotes(args.entrada, args.batch_size):
        amostra = batch.to_pandas()
        desenho_base.registrar(amostra)
        periodos.update((int(a), int(t)) for a, t in amostra[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        del amostra
        df = M5.normalizar_lote(batch)
        if not df.empty:
            y = df["y"].to_numpy(dtype=np.float64)
            w = pesos_amostrais(df)
            for (spec, dimensao), modelo in modelos.items():
                x = matriz_modelo(df, spec, dimensao)
                modelo.acumular_normal(x, y, w)
                del x
        del df
    for modelo in modelos.values():
        modelo.resolver()
        modelo.desenho = copy.deepcopy(desenho_base)
        modelo.desenho.iniciar_scores(len(modelo.nomes))
        print(f"{modelo.especificacao}/{modelo.dimensao}: {modelo.n:,} elegíveis; {len(modelo.ativos)} parâmetros identificáveis")
    del desenho_base
    gc.collect()

    print("Segunda passada: acumulando scores por UPA dentro de estrato")
    for batch in M5.iterar_lotes(args.entrada, args.batch_size):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        y = df["y"].to_numpy(dtype=np.float64)
        w = pesos_amostrais(df)
        for (spec, dimensao), modelo in modelos.items():
            assert modelo.beta is not None and modelo.desenho is not None
            x = matriz_modelo(df, spec, dimensao)
            residuo = y - x @ modelo.beta
            modelo.desenho.acumular(df, x * (w * residuo)[:, None])
            del x, residuo
        del df, y, w
        gc.collect()

    tests: dict[str, dict] = {}
    model_summaries: dict[str, dict] = {}
    heterogeneidade = []
    for key, modelo in modelos.items():
        cov = modelo.covariancia()
        model_key = key[0] if key[1] == "geral" else f"spline_{key[1]}"
        tests[model_key] = testes_modelo(modelo, cov)
        model_summaries[model_key] = {
            "especificacao": key[0], "dimensao": key[1], "n_observacoes_elegiveis": modelo.n,
            "k_regressores": len(modelo.nomes), "k_identificaveis": len(modelo.ativos),
            "n_upas": len(modelo.desenho.grupos), "n_estratos": len(modelo.desenho.estratos),
            "graus_liberdade_desenho": modelo.desenho.graus_liberdade,
            "soma_pesos_pessoa_trimestre": modelo.soma_pesos,
            "n_efetivo_kish": modelo.soma_pesos ** 2 / modelo.soma_pesos2,
        }
        if key[1] != "geral":
            name = f"heterogeneidade_{key[1]}"
            heterogeneidade.append((name, tests[model_key][name]["p_valor"]))

    # Ajuste BH para a família de três testes de heterogeneidade.
    p_h = M5.ajustar_bh(np.array([p for _, p in heterogeneidade], dtype=float))
    for (name, _), adjusted in zip(heterogeneidade, p_h):
        dim = name.removeprefix("heterogeneidade_")
        tests[f"spline_{dim}"][name]["p_valor_bh"] = float(adjusted) if np.isfinite(adjusted) else None

    efeitos_df = efeitos(modelos, args.nivel_confianca)
    efeitos_df.to_csv(args.saida_dir / "efeitos_retorno_educacao.csv", index=False)
    resumo = {
        "entrada": str(args.entrada), "nivel_confianca": args.nivel_confianca, "ponderacao": "V1028",
        "desenho": "Linearização por UPA dentro de estrato; conglomerado com reposição, sem FPC e sem recalibração.",
        "dominio": "Pessoas ocupadas de 25 a 65 anos, com rendimento e horas positivos e escolaridade válida; mesma amostra em todas as especificações.",
        "n_periodos": len(periodos), "periodos": [f"{a}T{t}" for a, t in sorted(periodos)],
        "politica_upa_unica": args.upa_unica, "politica_upa_multiestrato": args.upa_multiestrato,
        "upas_multiestrato": [{"UF": uf, "UPA": upa, "estratos": estratos} for (uf, upa), estratos in sorted(modelos[("categorias", "geral")].desenho.upas_multiestrato.items())],
        "nos_spline": list(NOS_SPLINE), "categorias_escolaridade": "VD3005 0–16; 16 representa 16 anos ou mais e é aproximado como 16 na tendência.",
        "modelos": model_summaries, "testes": tests,
        "ajuste_multiplo": "BH aplicado às 16 comparações vs código 0 e às 16 diferenças adjacentes por curva; também aplicado à família dos três testes de heterogeneidade.",
        "limitacao_temporal": "A PNAD revisita domicílios por trimestres consecutivos e iniciou a renovação gradual da Amostra Mestra em 2025T3; a variância empilhada é uma aproximação e não estima a covariância entre versões da UPA reclassificada.",
        "interpretacao": "Associações condicionais no salário-hora entre pessoas ocupadas; os resultados não identificam efeitos causais.",
        "arquivo_efeitos": "efeitos_retorno_educacao.csv",
    }
    (args.saida_dir / "resumo_retorno_educacao.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    relatorio = [
        "# Retorno associado à escolaridade",
        "",
        "A análise usa regressões ponderadas por `V1028` e a mesma amostra elegível em todas as especificações: pessoas ocupadas de 25 a 65 anos, com rendimento e horas positivas. O desfecho é o log do salário-hora real, calculado como rendimento habitual / (horas habituais × 4,33).",
        "",
        "## Modelos",
        "",
        "- **Categorias:** cada nível de VD3005 é comparado ao código 0 (sem instrução ou menos de um ano). O código 16 representa 16 anos ou mais.",
        "- **Linear:** anos de estudo entram como tendência linear; o código 16 é tratado como 16.",
        f"- **Spline linear por segmentos:** anos e termos `max(0, anos − nó)` com nós pré-definidos em {', '.join(map(str, NOS_SPLINE))} anos.",
        "- **Heterogeneidade:** a spline varia por sexo, cor/raça ou macrorregião. Os controles incluem formalidade, setor público, sexo, atividade, ocupação, cor/raça, idade e idade², UF, ano e trimestre.",
        "",
        "Os efeitos percentuais são `100 × (exp(Δ log salário-hora) − 1)`. A saída apresenta a diferença acumulada contra o código 0 e a diferença marginal entre níveis adjacentes. Os intervalos são de 95% e pontuais, transformados a partir da escala log. O teste de linearidade compara conjuntamente as categorias com uma tendência linear; o teste da spline avalia os três componentes não lineares. Os testes de heterogeneidade comparam os perfis educacionais em conjunto.",
        "",
        "## Amostra e inferência",
        "",
        f"A amostra tem {model_summaries['categorias']['n_observacoes_elegiveis']:,} observações elegíveis, {model_summaries['categorias']['n_upas']:,} versões de UPA e {model_summaries['categorias']['n_estratos']:,} estratos, nos {len(periodos)} trimestres de 2013–2025.",
        "",
        "Os erros-padrão usam linearização por UPA dentro de estrato, aproximação com reposição, sem correção de população finita e sem recalibração. UPAs reclassificadas são separadas por (UF, estrato, UPA), sem covariância estimada entre versões. Rotação de domicílios e a renovação gradual da Amostra Mestra iniciada em 2025T3 limitam a variância temporal empilhada. A inferência não reproduz integralmente a variância oficial do IBGE.",
        "",
        "## Interpretação",
        "",
        "Os resultados são associações condicionais entre escolaridade e salário-hora entre pessoas ocupadas. Não identificam o efeito causal da educação e não descrevem salários de pessoas fora da ocupação. Seleção ocupacional permanece uma extensão separada. Os testes de Wald-F são aproximações de desenho; os intervalos por nível são pontuais, não simultâneos.",
        "",
        "## Resultados dos testes de Wald-F",
        "",
        "| Teste | F | gl numerador | gl denominador | p-valor |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_key, group in tests.items():
        for name, result in group.items():
            relatorio.append(f"| `{name}` · `{model_key}` | {result['f'] if result['f'] is not None else '—'} | {result['gl_numerador']} | {result['gl_denominador']} | {result['p_valor'] if result['p_valor'] is not None else '—'} |")
    relatorio.extend(["", "## Arquivos", "", "- `efeitos_retorno_educacao.csv`: diferenças acumuladas e marginais, percentuais e intervalos por especificação e grupo.", "- `resumo_retorno_educacao.json`: amostra, desenho, especificações e testes conjuntos."])
    (args.saida_dir / "relatorio_retorno_educacao.md").write_text("\n".join(relatorio), encoding="utf-8")
    print(json.dumps({"n": model_summaries["categorias"]["n_observacoes_elegiveis"], "testes": tests, "saida": str(args.saida_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(executar(parse_args()))
