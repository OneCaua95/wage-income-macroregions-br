#!/usr/bin/env python3
"""Gera uma versão independente do dashboard inferencial do M5.

O arquivo resultante incorpora os CSVs e JSONs de resultados no próprio HTML.
Assim, o dashboard pode ser compartilhado como um único arquivo e aberto
diretamente no navegador, sem depender de um servidor ou de dados externos.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "pnad_pipeline" / "data" / "processed" / "modelos_salario_hora"
TEMPLATE = PROJECT_ROOT / "frontend" / "inferencia_m5.html"
OUTPUT = PROJECT_ROOT / "frontend" / "inferencia_m5_autonomo.html"

MODEL_FILES = {
    "geral": "inferencial_m5_geral.csv",
    "Norte": "inferencial_m5_norte.csv",
    "Nordeste": "inferencial_m5_nordeste.csv",
    "Sudeste": "inferencial_m5_sudeste.csv",
    "Sul": "inferencial_m5_sul.csv",
    "Centro-Oeste": "inferencial_m5_centro_oeste.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_embedded_data() -> dict:
    return {
        "summary": read_json(DATA_DIR / "resumo_inferencia_m5.json"),
        "datasets": {
            model: read_csv(DATA_DIR / filename)
            for model, filename in MODEL_FILES.items()
        },
        "comparisons": read_csv(DATA_DIR / "comparacao_especificacoes_inferenciais.csv"),
    }


def replace_script(template: str, embedded: dict) -> str:
    start = template.rfind("<script>")
    end = template.rfind("</script>")
    if start < 0 or end < start:
        raise ValueError("O template não contém um bloco de script válido.")

    script = template[start:end + len("</script>")]
    literal = json.dumps(embedded, ensure_ascii=False, separators=(",", ":"))
    script = re.sub(
        r"    const basePath = .*?\n    const files = \{.*?\n    \};",
        "    const embeddedData = " + literal + ";\n"
        "    const files = { geral: '', Norte: '', Nordeste: '', Sudeste: '', Sul: '', 'Centro-Oeste': '' };",
        script,
        count=1,
        flags=re.DOTALL,
    )
    script = re.sub(
        r"\n    async function fetchText\(path\) \{.*?\n    \}\n",
        "\n",
        script,
        count=1,
        flags=re.DOTALL,
    )
    standalone_init = """    function init() {
      try {
        state.summary = embeddedData.summary;
        state.datasets = embeddedData.datasets;
        state.comparisons = embeddedData.comparisons;
        renderSelectors();
        renderAll();
      } catch (error) {
        $('#error-box').classList.remove('hidden');
        $('#error-box').innerHTML = `<strong>Dados embutidos não puderam ser exibidos.</strong> ${escapeHtml(error.message)}`;
      }
    }
"""
    script = re.sub(
        r"    async function init\(\) \{.*?\n    \}\n\n    \$\('#model-select'\)",
        standalone_init + "\n    $('#model-select')",
        script,
        count=1,
        flags=re.DOTALL,
    )
    return template[:start] + script + template[end + len("</script>"):]


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    embedded = build_embedded_data()
    standalone = replace_script(template, embedded)
    standalone = standalone.replace(
        "<title>PNAD / Inferência do salário-hora</title>",
        "<title>PNAD / Inferência do salário-hora · arquivo único</title>",
    )
    standalone = standalone.replace(
        '<div class="run-badge">95% confiança · 2 vias</div>',
        '<div class="run-badge">95% confiança · arquivo único</div>',
    )
    standalone = re.sub(
        r'      <div class="download-links">.*?</div>',
        '      <div class="download-links"><span>✓ Resultados incorporados neste arquivo HTML; não há dependências externas para a visualização.</span></div>',
        standalone,
        count=1,
        flags=re.DOTALL,
    )
    OUTPUT.write_text(standalone, encoding="utf-8")
    print(f"Gerado: {OUTPUT}")
    print(f"Tamanho: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
