"""Constantes utilizadas no processo de extração e padronização da PNAD Contínua."""

from __future__ import annotations
from typing import Final

# Colunas selecionadas para download do microdado da PNAD Contínua
COLUMNS: Final[list[str]] = [
    "Ano",
    "Trimestre",
    "UF",
    "V2007",
    "V20082",
    "V2010",
    "V2009",
    "V3003",
    "VD3005",
    "V403311",
    "V403312",
    "VD4017",
    "VD4001",
    "VD4002",
    "VD4003",
    "VD4004",
    "VD4005",
    "VD4009",
    "VD4010",
    "VD4011",
    "VD4012",
    "VD4016",
    "VD4019",
    "VD4031",
]

# Mapeamento para renomeação com nomes intuitivos e padronizados
COLUMNS_RENAME: Final[dict[str, str]] = {
    "V2007": "Sexo",
    "V20082": "Ano_Nascimento",
    "V2010": "Cor",
    "V2009": "Idade",
    "V3003": "Escolaridade",
    "VD3005": "Anos_de_Estudo",
    "V403311": "Renda_Faixa",
    "V403312": "Renda_PEA_Reais",
    "VD4017": "Renda_Efetivo_14_Reais",
    "VD4001": "F_Trabalho",
    "VD4002": "Condicao_Ocupacao",
    "VD4003": "Forca_Trabalho_potencial",
    "VD4004": "Subocupacao_por_insu_horas",
    "VD4005": "Pessoa_desalentadas",
    "VD4009": "Categoria_emprego",
    "VD4010": "Grupo_atv_princ_empreedimento",
    "VD4011": "Grupo_ocupacional_no_emprego",
    "VD4012": "Contribuicao_previdencia",
    "VD4016": "Rendimento_hab_Trab_princ",
    "VD4019": "Rendimento_hab_Tds_Trabs",
    "VD4031": "Horas_hab_trabalhadas",
}

# Mapeamento do código da UF (IBGE) para a Macrorregião correspondente
UF_PARA_REGIAO: Final[dict[int, str]] = {
    # Norte
    11: "Norte",
    12: "Norte",
    13: "Norte",
    14: "Norte",
    15: "Norte",
    16: "Norte",
    17: "Norte",
    # Nordeste
    21: "Nordeste",
    22: "Nordeste",
    23: "Nordeste",
    24: "Nordeste",
    25: "Nordeste",
    26: "Nordeste",
    27: "Nordeste",
    28: "Nordeste",
    29: "Nordeste",
    # Sudeste
    31: "Sudeste",
    32: "Sudeste",
    33: "Sudeste",
    35: "Sudeste",
    # Sul
    41: "Sul",
    42: "Sul",
    43: "Sul",
    # Centro-Oeste
    50: "Centro-Oeste",
    51: "Centro-Oeste",
    52: "Centro-Oeste",
    53: "Centro-Oeste",
}

# Defaults
DATA_BASE_DEFLACAO_PADRAO: Final[str] = "01/01/2026"
IPCA_SGS_URL_PADRAO: Final[str] = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"
)
