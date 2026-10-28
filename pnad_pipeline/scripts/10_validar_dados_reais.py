#!/usr/bin/env python3
"""Valida o fluxo ponderado em uma onda real, preservando a base histórica.

Baixa o ZIP oficial por HTTPS, usa as posições do dicionário local, executa a
coleta/tratamento existentes e grava proveniência e verificações observadas.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request
import zipfile
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pnad_pipeline.analise.amostragem import validar_desenho
from pnad_pipeline.extracao.coleta import processar_periodo
from pnad_pipeline.extracao.io import ParquetStreamWriter
from pnad_pipeline.extracao.ipca import get_ipca_bc
from pnad_pipeline.extracao.ipca import calcular_fator_acumulado


def carregar_script(nome, arquivo):
    spec = importlib.util.spec_from_file_location(nome, ROOT / "scripts" / arquivo)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


def baixar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and zipfile.is_zipfile(destino):
        print(f"Arquivo oficial em cache: {destino.name}", flush=True)
        return
    parcial = destino.with_suffix(".zip.part")
    print(f"Baixando {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resposta, parcial.open("wb") as handle:
        total = int(resposta.headers.get("Content-Length", 0))
        recebidos = 0
        marca = 0
        while True:
            bloco = resposta.read(1024 * 1024)
            if not bloco:
                break
            handle.write(bloco)
            recebidos += len(bloco)
            if recebidos - marca >= 32 * 1024 * 1024:
                print(f"Recebidos {recebidos / 1024**2:.0f} MiB / {total / 1024**2:.0f} MiB", flush=True)
                marca = recebidos
    if not zipfile.is_zipfile(parcial):
        raise ValueError("O download não é um ZIP válido.")
    parcial.replace(destino)


class ExtratorZipOficial:
    def __init__(self, arquivo, dicionario):
        self.arquivo = arquivo
        self.dicionario = dicionario

    def download_data(self, year, quarter, cols):
        doc = pd.read_excel(self.dicionario, header=1)
        selecionadas = doc.loc[doc.iloc[:, 2].isin(cols)]
        names = selecionadas.iloc[:, 2].tolist()
        faltantes = set(cols) - set(names)
        if faltantes:
            raise ValueError(f"Variáveis ausentes no dicionário: {faltantes}")
        starts = pd.to_numeric(selecionadas.iloc[:, 0]).astype(int).to_numpy() - 1
        widths = pd.to_numeric(selecionadas.iloc[:, 1]).astype(int).to_numpy()
        colspecs = list(zip(starts, starts + widths))
        partes = []
        with zipfile.ZipFile(self.arquivo) as pacote:
            membro = next(n for n in pacote.namelist() if n.lower().endswith(".txt"))
            with pacote.open(membro) as handle:
                for i, chunk in enumerate(pd.read_fwf(handle, colspecs=colspecs, names=names, chunksize=100_000), 1):
                    partes.append(chunk)
                    print(f"Lote oficial {i}: {len(chunk):,} registros lidos", flush=True)
        return pd.concat(partes, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ano", type=int, default=2025)
    parser.add_argument("--trimestre", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--reprocessar", action="store_true", help="Releia o TXT oficial mesmo se a base completa estiver em cache.")
    args = parser.parse_args()
    periodo = f"{args.ano}T{args.trimestre}"
    saida = ROOT / "data" / "processed" / "validacao_real" / periodo
    saida.mkdir(parents=True, exist_ok=True)
    nome_zip = f"PNADC_{args.trimestre:02}{args.ano}.zip"
    url = f"https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/{args.ano}/{nome_zip}"
    arquivo = ROOT / "data" / "raw" / nome_zip
    inicio = time.perf_counter()
    baixar(url, arquivo)
    sha = hashlib.sha256()
    with arquivo.open("rb") as handle:
        for bloco in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(bloco)
    url_ipca = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"
        f"&dataInicial=01/{args.trimestre * 3:02}/{args.ano}&dataFinal=31/01/2026"
    )
    completa = saida / f"pnad_completa_{periodo}.parquet"
    if completa.exists() and (saida / "ipca_observado.csv").exists() and not args.reprocessar:
        print("Reutilizando base completa oficial processada nesta validação.", flush=True)
        base = pd.read_parquet(completa)
        ipca = pd.read_csv(saida / "ipca_observado.csv", parse_dates=["data"])
        fator = calcular_fator_acumulado(ipca, f"01/{args.trimestre * 3:02}/{args.ano}", "01/01/2026")
    else:
        ipca = get_ipca_bc(url_ipca)
        ipca.to_csv(saida / "ipca_observado.csv", index=False)
        base, _, fator = processar_periodo(
            args.ano, args.trimestre, ipca,
            extractor=ExtratorZipOficial(arquivo, ROOT / "data" / "raw" / "dicionario_PNADC_microdados_trimestral.xls"),
        )
        with ParquetStreamWriter(completa, metadata={"pnad_amostra_completa": "true"}) as writer:
            writer.append(base)
    validar_desenho(base)
    checks = []

    def verificar(nome, condicao, observado):
        checks.append({"verificacao": nome, "status": "aprovado" if condicao else "reprovado", "observado": observado})

    verificar("Período do arquivo", bool(base.Ano.eq(args.ano).all() and base.Trimestre.eq(args.trimestre).all()), periodo)
    verificar("Peso final positivo e finito", bool(np.isfinite(base.V1028).all() and base.V1028.gt(0).all()), f"{len(base):,} registros")
    verificar("Precisão do peso preservada", str(base.V1028.dtype) == "float64", str(base.V1028.dtype))
    verificar("Cobertura das UFs", base.UF.nunique() == 27, int(base.UF.nunique()))
    verificar("Regiões identificadas", base.Regiao.notna().all(), int(base.Regiao.nunique()))
    verificar("Amostra anterior ao filtro de idade", base.Idade.lt(25).any() and base.Idade.gt(65).any(), {"idade_minima": int(base.Idade.min()), "idade_maxima": int(base.Idade.max())})
    pares = base[["UF", "Estrato", "UPA"]].drop_duplicates()
    tamanhos = pares.groupby(["UF", "Estrato"]).size()
    unicas = [{"UF": int(uf), "Estrato": int(estrato)} for uf, estrato in tamanhos[tamanhos.eq(1)].index]
    checks.append({"verificacao": "Estratos com uma única UPA", "status": "atenção" if unicas else "aprovado", "observado": {"estratos": len(tamanhos), "upas": len(pares), "estratos_upa_unica": unicas}, "tratamento": "Ajuste explícito: score centrado na média global das UPAs, conforme survey.lonely.psu=adjust."})
    verificar("Peso associado à população coberta", base.V1028.sum() > len(base), {"n_amostral": len(base), "soma_pesos": float(base.V1028.sum())})
    del base
    tratamento = carregar_script("tratamento_real", "02_tratar_pnad.py")
    analitica = saida / f"pnad_analise_{periodo}.parquet"
    tratamento.tratar_base(Namespace(entrada=completa, saida=analitica, min_idade=25, max_idade=65, batch_size=100_000))
    tratamento.gerar_tabelas(analitica, saida / "tabelas")
    verificar("Tratamento real em lotes", pq.ParquetFile(analitica).metadata.num_rows > 0, pq.ParquetFile(analitica).metadata.num_rows)
    m5 = carregar_script("inferencia_real", "08_inferencia_m5.py")
    m5.executar(Namespace(entrada=completa, saida_dir=saida / "inferencia", batch_size=100_000, nivel_confianca=.95, upa_unica="ajustar"))
    verificar("Inferência ponderada real", True, "Brasil e cinco macroregiões")
    import statsmodels.api as sm
    lote = next(m5.iterar_lotes(completa, 100_000))
    amostra_real = m5.normalizar_lote(lote).sample(n=12_000, random_state=2026)
    x = m5.matriz_m5(amostra_real, m5.UFS)
    y = amostra_real.y.to_numpy(dtype=float)
    w = amostra_real.V1028.to_numpy(dtype=float)
    modelo = m5.ResultadoModelo("verificacao_numerica", m5.UFS)
    modelo.acumular_normal(x, y, w)
    modelo.resolver()
    ref = sm.WLS(y, x[:, modelo.indices_ativos], weights=w).fit()
    erro_beta = float(np.max(np.abs(modelo.beta[modelo.indices_ativos] - ref.params)))
    verificar("Coeficientes conferidos com statsmodels.WLS", erro_beta < 1e-8, {"registros_reais_verificados": len(amostra_real), "maior_diferenca_coeficiente": erro_beta, "tolerancia": 1e-8})
    for macro in ["geral", "norte", "nordeste", "sudeste", "sul", "centro_oeste"]:
        tabela = pd.read_csv(saida / "inferencia" / f"inferencial_m5_{macro}.csv")
        ativos = tabela.loc[tabela.identificavel]
        verificar(f"Coeficientes e intervalos finitos: {macro}", bool(np.isfinite(ativos[["coeficiente", "erro_padrao_desenho_pnad", "ic_inferior_desenho_pnad", "ic_superior_desenho_pnad"]]).all().all()), {"parametros_identificaveis": len(ativos), "termos_constantes_ou_redundantes": len(tabela) - len(ativos)})
    registro = {
        "executado_em": datetime.now(timezone(timedelta(hours=-3))).isoformat(),
        "periodo": periodo, "tipo_dados": "microdados oficiais reais do IBGE", "url_fonte": url,
        "arquivo_fonte": nome_zip, "sha256": sha.hexdigest(), "bytes_fonte": arquivo.stat().st_size,
        "fator_ipca": fator, "data_base_rendimento": "Janeiro/2026", "url_ipca": url_ipca,
        "duracao_segundos": time.perf_counter() - inicio, "verificacoes": checks,
        "politica_upa_unica": "ajustar",
        "incidente_resolvido": "A primeira execução foi interrompida no estrato 1620020 por conter uma única UPA. O tratamento ajustar foi ativado explicitamente e documentado; o padrão do script principal continua sendo erro.",
        "referencia_ajuste": "https://github.com/cran/survey/blob/master/R/multistage.R",
        "escopo": "Validação em uma onda completa; não substitui a reestimação ponderada da série 2013–2025.",
    }
    (saida / "validacao_observada.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Validação real concluída: {saida}", flush=True)


if __name__ == "__main__":
    main()
