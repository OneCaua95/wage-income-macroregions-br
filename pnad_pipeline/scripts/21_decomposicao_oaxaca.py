#!/usr/bin/env python3
"""Decomposição Oaxaca–Blinder ponderada por sexo, cor/raça e macroregião.

Referência: coeficientes pooled sobre cada par, incluindo indicador do grupo
como controle adicional (Jann, 2008). Usa o domínio elegível do M5 histórico.
A diferença é grupo menos referência. As parcelas são aditivas em log;
transformações percentuais separadas não são somáveis.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("_pnad_m5_oaxaca", SCRIPT_DIR / "08_inferencia_m5.py")
M5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M5
SPEC.loader.exec_module(M5)

from pnad_pipeline.analise.amostragem import LinearizacaoUPA, pesos_amostrais
from pnad_pipeline.analise.regressao.oaxaca import decompor, influencias

COMPARACOES = [
    {"id": "sexo_2_1", "coluna": "Sexo", "codigo": 2, "referencia_codigo": 1, "grupo": "Mulheres", "referencia": "Homens", "rotulo": "Mulheres vs homens"},
    *[{"id": f"cor_{codigo}_1", "coluna": "Cor", "codigo": codigo, "referencia_codigo": 1, "grupo": nome, "referencia": "Branca", "rotulo": f"{nome} vs branca"}
      for codigo, nome in [(2, "Preta"), (3, "Amarela"), (4, "Parda"), (5, "Indígena")]],
    *[{"id": f"regiao_{M5.slug(regiao)}_sudeste", "coluna": "Regiao", "codigo": regiao,
       "referencia_codigo": "Sudeste", "grupo": regiao, "referencia": "Sudeste", "rotulo": f"{regiao} vs Sudeste"}
      for regiao in M5.MACROREGIOES if regiao != "Sudeste"],
]
COMPONENTES = [("total", "Diferença total"), ("explicada", "Parcela explicada"), ("nao_explicada", "Parcela não explicada")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada", type=Path, default=PROJECT_ROOT / "data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet")
    p.add_argument("--saida-dir", type=Path, default=PROJECT_ROOT / "data/processed/recoleta_ponderada/inferencia_amostral/oaxaca")
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--nivel-confianca", type=float, default=.95)
    p.add_argument("--upa-unica", choices=["erro", "ajustar"], default="ajustar")
    p.add_argument("--upa-multiestrato", choices=["erro", "separar"], default="separar")
    return p.parse_args()


def preparar_modelos():
    nomes = ["Intercept", *M5.nomes_features(M5.UFS)]
    modelos = []
    for item in COMPARACOES:
        # A variável definidora dos grupos não é uma característica explicativa.
        # Também se retira sua interação para obter inclinações pooled comuns.
        excluir = {"Sexo_Dummy", "Estudo_x_Sexo_Dummy"} if item["coluna"] == "Sexo" else set(M5.CORES.values())
        if item["coluna"] == "Regiao":
            excluir = {nome for nome in nomes if nome.startswith("UF_")}
        indices = np.array([i for i, nome in enumerate(nomes) if nome not in excluir])
        k = len(indices)
        modelos.append({
            **item, "indices": indices, "nomes": [nomes[i] for i in indices] + ["Indicador_grupo"],
            "modelo": M5.ResultadoModelo(item["id"], M5.UFS),
            "grupos": [{"n": 0, "peso": 0., "soma_x": np.zeros(k), "soma_y": 0.} for _ in range(2)],
        })
    return modelos


def amostra_par(df, x_m5, item):
    mask = df[item["coluna"]].isin([item["codigo"], item["referencia_codigo"]]).to_numpy()
    sub = df.loc[mask]
    x = x_m5[np.ix_(mask, item["indices"])]
    grupo = sub[item["coluna"]].eq(item["codigo"]).to_numpy()
    return sub, x, grupo


def executar(args):
    if args.batch_size <= 0 or not 0 < args.nivel_confianca < 1:
        raise ValueError("Use lote positivo e confiança entre 0 e 1.")
    desenho = LinearizacaoUPA(politica_upa_unica=args.upa_unica, politica_upa_multiestrato=args.upa_multiestrato)
    modelos = preparar_modelos()
    periodos = set()
    n_elegiveis = 0
    print(f"Primeira passada: médias e {len(modelos)} modelos pooled ponderados", flush=True)
    for n_lote, batch in enumerate(M5.iterar_lotes(args.entrada, args.batch_size), 1):
        desenho.registrar(batch.to_pandas())
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        n_elegiveis += len(df)
        periodos.update(df[["Ano", "Trimestre"]].drop_duplicates().itertuples(index=False, name=None))
        x_m5 = M5.matriz_m5(df, M5.UFS)
        for item in modelos:
            sub, x, grupo = amostra_par(df, x_m5, item)
            if sub.empty:
                continue
            y, w = sub.y.to_numpy(dtype=float), pesos_amostrais(sub)
            item["modelo"].acumular_normal(np.column_stack([x, grupo.astype(float)]), y, w)
            for g in (0, 1):
                mask = grupo == g
                acc = item["grupos"][g]
                acc["n"] += int(mask.sum())
                acc["peso"] += float(w[mask].sum())
                acc["soma_x"] += x[mask].T @ w[mask]
                acc["soma_y"] += float(w[mask] @ y[mask])
        if n_lote % 10 == 0:
            print(f"  {n_lote} lotes; {n_elegiveis:,} observações elegíveis", flush=True)

    for item in modelos:
        for acc in item["grupos"]:
            if acc["n"] < 30 or acc["peso"] <= 0:
                raise ValueError(f"Amostra insuficiente em {item['rotulo']}.")
            acc["media_x"] = acc["soma_x"] / acc["peso"]
            acc["media_y"] = acc["soma_y"] / acc["peso"]
        modelo = item["modelo"]
        modelo.resolver()
        ref, g = item["grupos"]
        c = np.r_[g["media_x"] - ref["media_x"], 0.]
        if modelo.k - 1 not in modelo.indices_ativos or not np.allclose(modelo.xtx @ (modelo.bread @ c), c, atol=1e-7, rtol=1e-6):
            raise ValueError(f"Contraste não identificável em {item['rotulo']}; reveja o suporte comum dos controles.")
        item["estimativas"] = decompor(modelo.beta, g["media_x"], ref["media_x"], g["media_y"], ref["media_y"])
        # Com intercepto e indicador de grupo, U deve coincidir com esse coeficiente.
        if not np.isclose(item["estimativas"][2], modelo.beta[-1], atol=1e-7):
            raise ValueError("A identidade Oaxaca não fechou com o indicador do modelo pooled.")

    desenho.iniciar_scores(len(modelos) * 3)
    print("Segunda passada: linearização conjunta das médias e dos coeficientes", flush=True)
    for n_lote, batch in enumerate(M5.iterar_lotes(args.entrada, args.batch_size), 1):
        df = M5.normalizar_lote(batch)
        if df.empty:
            continue
        x_m5 = M5.matriz_m5(df, M5.UFS)
        scores = np.zeros((len(df), len(modelos) * 3))
        for i, item in enumerate(modelos):
            sub, x, grupo = amostra_par(df, x_m5, item)
            if sub.empty:
                continue
            modelo = item["modelo"]
            ref, g = item["grupos"]
            mask = df[item["coluna"]].isin([item["codigo"], item["referencia_codigo"]]).to_numpy()
            scores[mask, 3*i:3*i+3] = influencias(
                x, sub.y.to_numpy(dtype=float), grupo, pesos_amostrais(sub), modelo.beta, modelo.bread,
                g["media_x"], ref["media_x"], g["media_y"], ref["media_y"], g["peso"], ref["peso"],
            )
        desenho.acumular(df, scores)
        if n_lote % 10 == 0:
            print(f"  {n_lote} lotes de scores", flush=True)

    cov = desenho.covariancia(np.eye(len(modelos) * 3))
    critico = stats.t.ppf(1 - (1 - args.nivel_confianca) / 2, desenho.graus_liberdade)
    linhas, comparacoes = [], []
    for i, item in enumerate(modelos):
        ref, g = item["grupos"]
        comparacoes.append({
            **{k: item[k] for k in ("id", "rotulo", "grupo", "referencia")},
            "n_grupo": g["n"], "n_referencia": ref["n"], "peso_grupo_v1028": g["peso"], "peso_referencia_v1028": ref["peso"],
            "media_log_grupo": g["media_y"], "media_log_referencia": ref["media_y"],
            "diferenca_total_pct": float(100 * np.expm1(item["estimativas"][0])),
            "termos_omitidos": [nome for j, nome in enumerate(item["nomes"]) if j not in item["modelo"].indices_ativos],
        })
        for j, (codigo, rotulo) in enumerate(COMPONENTES):
            valor = float(item["estimativas"][j])
            se = float(np.sqrt(max(cov[3*i+j, 3*i+j], 0)))
            linhas.append({"comparacao_id": item["id"], "comparacao": item["rotulo"], "componente": codigo, "rotulo": rotulo,
                           "estimativa_log": valor, "erro_padrao_log": se,
                           "ic_inferior_log": float(valor - critico * se), "ic_superior_log": float(valor + critico * se)})
    resumo = {
        "status": "ok", "versao": "oaxaca-1.1", "periodo": "2013–2025 (empilhado)", "n_periodos": len(periodos),
        "primeiro_periodo": "%dT%d" % min(periodos), "ultimo_periodo": "%dT%d" % max(periodos),
        "processado_em": datetime.now(timezone.utc).isoformat(),
        "entrada": {"arquivo": args.entrada.name, "bytes": args.entrada.stat().st_size},
        "n_observacoes_elegiveis": n_elegiveis, "n_upas_desenho": len(desenho.grupos), "n_estratos_desenho": len(desenho.estratos),
        "graus_liberdade_desenho": desenho.graus_liberdade, "nivel_confianca": args.nivel_confianca,
        "politica_upa_unica": args.upa_unica, "politica_upa_multiestrato": args.upa_multiestrato,
        "n_upas_multiestrato": len(desenho.upas_multiestrato), "n_estratos_upa_unica": sum(len(ids) == 1 for ids in desenho.estratos.values()),
        "metodo": "Oaxaca–Blinder em duas parcelas; referência pooled com indicador de grupo em cada par (Jann, 2008).",
        "formula": "D = média(log salário)_grupo − média(log salário)_referência; E = (média(X)_grupo − média(X)_referência)' beta_pooled; U = D − E.",
        "controles": "Família M5: idade, idade², escolaridade categórica, formalidade, setor público, setor, ocupação, UF, ano, trimestre e interações de estudo com formalidade e setor público. Em sexo, excluem-se sexo e sua interação dos controles comuns. Em cor, retiram-se as indicadoras de cor. Nas comparações regionais, o Sudeste é a referência e retiram-se todas as indicadoras de UF, pois a UF determina a macroregião. Cor e sexo com sua interação permanecem nesses ajustes. O indicador do grupo entra apenas no ajuste pooled.",
        "variancia": "Scores conjuntos de médias e coeficientes, linearização por UPA dentro de estrato; IC pontual t com gl = UPAs − estratos. Cadastro completo antes do domínio, aproximação com reposição, sem FPC ou recalibração. A covariância entre versões da UPA reclassificada não é estimada; rotação e renovação amostral limitam a aproximação temporal.",
        "interpretacao": "A parcela não explicada não é uma medida automática de discriminação nem de causalidade. Inclui fatores não observados e depende dos controles e da referência. Formalidade, setor e ocupação podem mediar diferenças; a parcela explicada não é uma atribuição causal.",
        "unidade": "Log do salário-hora real, a preços de janeiro/2026. Componentes somam em log; seus percentuais exponenciados não somam.",
        "amostra": "Domínio do M5: ocupados de 25–65 anos com renda e horas positivas e covariáveis válidas; V1028. Pessoas-trimestre, não pessoas únicas.",
        "referencia_metodologica": "https://repec.ethz.ch/ets/papers/jann_oaxaca.pdf",
        "comparacoes": comparacoes,
    }
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(args.saida_dir / "decomposicao_oaxaca.csv", index=False, float_format="%.12g")
    np.savez_compressed(args.saida_dir / "covariancia_oaxaca.npz", covariancia=cov,
                        ordem=np.array([f"{item['id']}:{codigo}" for item in modelos for codigo, _ in COMPONENTES]))
    report = ["# Decomposição Oaxaca–Blinder ponderada", "", resumo["metodo"], "", resumo["amostra"], "", resumo["formula"], "",
              "| Comparação | Parcela | Estimativa log | IC inferior | IC superior |", "| --- | --- | ---: | ---: | ---: |"]
    report += [f"| {r['comparacao']} | {r['rotulo']} | {r['estimativa_log']:.5f} | {r['ic_inferior_log']:.5f} | {r['ic_superior_log']:.5f} |" for r in linhas]
    report += ["", resumo["controles"], "", resumo["variancia"], "", resumo["unidade"], "", resumo["interpretacao"], "", resumo["referencia_metodologica"], ""]
    (args.saida_dir / "relatorio_oaxaca.md").write_text("\n".join(report), encoding="utf-8")
    # Publica o marcador de conclusão por último para a API.
    temp = args.saida_dir / "resumo_oaxaca.tmp"
    temp.write_text(json.dumps(resumo, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(args.saida_dir / "resumo_oaxaca.json")
    print(f"Concluído: {len(modelos)} comparações e {len(linhas)} parcelas; saída em {args.saida_dir}", flush=True)
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        raise SystemExit(executar(parse_args()))
