#!/usr/bin/env python3
"""Indicadores trimestrais e composição do emprego de pessoas de 25 a 65 anos."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.stats import t

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
from pnad_pipeline.analise.amostragem import LinearizacaoUPA, exigir_colunas_desenho, pesos_amostrais
from pnad_pipeline.analise.emprego import CRITERIO_INFORMALIDADE, indicadores_emprego
from pnad_pipeline.analise.regressao.comparacao_m5 import exportar_comparacao

HISTORICO = SCRIPT_DIR.parent / "data/processed/recoleta_ponderada"
DIMENSOES = {"geral": None, "regiao": "Regiao", "sexo": "Sexo", "cor": "Cor"}
ROTULOS = {"sexo": {1: "Homens", 2: "Mulheres"}, "cor": {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena"}}


def taxas_grupo(df, desenho):
    w = pesos_amostrais(df)
    trabalho = df.F_Trabalho.eq(1).to_numpy()
    ocupado = trabalho & df.Condicao_Ocupacao.eq(1).to_numpy()
    desocupado = trabalho & df.Condicao_Ocupacao.eq(2).to_numpy()
    numeradores = np.column_stack((trabalho, ocupado, desocupado)).astype(float)
    denominadores = np.column_stack((np.ones(len(df)), np.ones(len(df)), trabalho)).astype(float)
    totais = w @ denominadores
    taxas = np.divide(w @ numeradores, totais, out=np.zeros(3), where=totais > 0)
    scores = np.divide(w[:, None] * (numeradores - denominadores * taxas), totais,
                       out=np.zeros_like(numeradores), where=totais > 0)
    desenho.iniciar_scores(3)
    desenho.acumular(df, scores)
    erros = np.sqrt(np.maximum(np.diag(desenho.covariancia(np.eye(3))), 0))
    critico = t.ppf(.975, desenho.graus_liberdade)
    resultado = {"n_populacao": len(df), "n_forca_trabalho": int(trabalho.sum()),
                 "n_ocupados": int(ocupado.sum()), "n_desocupados": int(desocupado.sum()),
                 "populacao_ponderada": float(totais[0]), "forca_trabalho_ponderada": float(totais[2])}
    for i, nome in enumerate(("participacao", "ocupacao", "desocupacao")):
        suficiente = denominadores[:, i].sum() >= 30 and totais[i] > 0
        resultado[f"taxa_{nome}_pct"] = float(100 * taxas[i]) if suficiente else None
        resultado[f"{nome}_ic_inferior_pct"] = float(100 * max(0, taxas[i] - critico * erros[i])) if suficiente else None
        resultado[f"{nome}_ic_superior_pct"] = float(100 * min(1, taxas[i] + critico * erros[i])) if suficiente else None
    return resultado


def executar(entrada, saida, batch_size=100_000):
    if batch_size <= 0:
        raise ValueError("O tamanho do lote deve ser positivo.")
    origem = entrada.stat()
    parquet = pq.ParquetFile(entrada)
    exigir_colunas_desenho(parquet.schema_arrow.names)
    if (parquet.schema_arrow.metadata or {}).get(b"pnad_amostra_completa") != b"true":
        raise ValueError("Use a base completa anterior aos filtros de idade e ocupação.")
    colunas = ["Ano", "Trimestre", "Idade", "F_Trabalho", "Condicao_Ocupacao", "Regiao", "Sexo", "Cor", "V1028", "UF", "Estrato", "UPA",
               "Categoria_emprego", "Contribuicao_previdencia", "Grupo_atv_princ_empreedimento", "Horas_hab_trabalhadas", "Rendimento_hab_Trab_princ"]
    ausentes = set(colunas).difference(parquet.schema_arrow.names)
    if ausentes:
        raise ValueError(f"Reextraia as variáveis ausentes nos microdados: {sorted(ausentes)}.")
    periodos = set()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["Ano", "Trimestre"]):
        periodos.update(batch.to_pandas().drop_duplicates().itertuples(index=False, name=None))
    dataset = ds.dataset(entrada, format="parquet")
    rows, composicao = [], []
    n_alvo, n_excluidos = 0, 0
    for ano, trimestre in sorted(periodos):
        filtro = (ds.field("Ano") == ano) & (ds.field("Trimestre") == trimestre)
        raw = dataset.to_table(columns=colunas, filter=filtro).to_pandas()
        desenho = LinearizacaoUPA(politica_upa_unica="ajustar", politica_upa_multiestrato="separar")
        desenho.registrar(raw)
        alvo = raw.loc[raw.Idade.between(25, 65)].copy()
        fora = alvo.F_Trabalho.eq(2) & alvo.Condicao_Ocupacao.isna()
        validos = fora | (alvo.F_Trabalho.eq(1) & alvo.Condicao_Ocupacao.isin([1, 2]))
        n_excluidos += int((~validos).sum())
        alvo = alvo.loc[validos]
        n_alvo += len(alvo)
        for dimensao, coluna in DIMENSOES.items():
            grupos = [("Brasil", alvo)] if coluna is None else alvo.groupby(coluna, observed=True)
            for grupo, df in grupos:
                if dimensao in ROTULOS and grupo not in ROTULOS[dimensao]:
                    continue
                rotulo = ROTULOS.get(dimensao, {}).get(grupo, str(grupo))
                chave = {"trimestre": f"{ano}T{trimestre}", "dimensao": dimensao, "grupo": rotulo}
                emprego, setores = indicadores_emprego(df, desenho)
                composicao.extend({**chave, **s} for s in setores)
                rows.append({**chave,
                             "graus_liberdade": desenho.graus_liberdade,
                             "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
                             **taxas_grupo(df, desenho), **emprego})
        print(f"{ano}T{trimestre}: {len(alvo):,} registros no domínio", flush=True)
    fim = entrada.stat()
    if (origem.st_size, origem.st_mtime_ns) != (fim.st_size, fim.st_mtime_ns):
        raise ValueError("A base mudou durante o processamento.")
    inicio, ultimo = min(periodos), max(periodos)
    relatorio = {
        "status": "ok", "versao": "mercado-trabalho-2.0", "processado_em": datetime.now(timezone.utc).isoformat(),
        "periodo": f"{inicio[0]}T{inicio[1]} a {ultimo[0]}T{ultimo[1]}", "n_periodos": len(periodos),
        "ponderacao": "V1028", "n_observacoes": n_alvo, "n_condicao_ausente_ou_inconsistente": n_excluidos,
        "entrada": {"arquivo": entrada.name, "bytes": origem.st_size}, "rows": rows, "composicao_setorial": composicao,
        "criterio_informalidade": CRITERIO_INFORMALIDADE,
        "notas": [
            "Domínio de 25 a 65 anos. Participação = força de trabalho / população do domínio. Ocupação = ocupados / população do domínio. Desocupação = desocupados / força de trabalho.",
            "VD4001 identifica a força de trabalho e VD4002 distingue ocupados e desocupados. Condições ausentes ou inconsistentes são excluídas e contadas no resumo. Os denominadores são divulgados junto das taxas.",
            "As taxas e os IC95% pontuais usam V1028 e linearização de razões por UPA dentro de estrato, com cadastro completo antes do domínio. Os limites t são restritos a 0–100%. Denominadores com menos de 30 registros têm as taxas suprimidas.",
            "Cada trimestre tem seu desenho e seu intervalo. O recorte etário difere do indicador nacional de 14 anos ou mais. Os grupos são consultas pré-calculadas e os filtros demográficos globais não são aplicados.",
            CRITERIO_INFORMALIDADE,
            "Conta própria (VD4009=9), empregados formais (1,3,5,7) e setor público (5,6,7) usam ocupados com categoria válida; informalidade exclui conta própria/empregadores sem contribuição informada. As categorias não são uma partição entre si.",
            "Jornada: horas habituais semanais em todos os trabalhos, positivas e válidas (VD4031). Rendimento: média habitual mensal real positiva do trabalho principal (VD4016), a preços de janeiro/2026. Os dois domínios são independentes e não exigem salário-hora válido.",
            "A proxy salarial histórica combina rendimento do trabalho principal com horas de todos os trabalhos (VD4016 / (VD4031 × 4,33)); a interpretação é limitada para múltiplos vínculos. Os modelos anteriores foram preservados.",
            "Composição setorial: participação no total de ocupados, incluindo Sem informação para atividade ausente/não reconhecida. As treze categorias somam 100%; os ICs são pontuais. Denominadores e contagens constam no CSV.",
        ],
    }
    exportar_comparacao(saida, "mercado_trabalho", relatorio)
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=HISTORICO / "pnad_consolidada_2013_2025.parquet")
    parser.add_argument("--saida-dir", type=Path, default=HISTORICO / "inferencia_amostral/mercado_trabalho")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    executar(args.entrada, args.saida_dir, args.batch_size)
