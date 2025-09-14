from source.extract import Extractor
import pandas as pd

data = Extractor()

class Transform:

    def __init__(self):
        pass
        
    def get_data(self, year, quarter):
        
        columns = [
                            "Ano",
                            "Trimestre",
                            "UF",
                            "V2007",    # Sexo
                            "V20082",   # Ano Nascimento
                            "V2010",    # Cor/Raça
                            "V3003",    # Escolaridade
                            "V403311", # Rendimento
                            "VD4001",   # Força de trabalho
                            "VD4012"    # Ocupação no setor
                        ]

        columns_rename = {
            "V2007":"Sexo",
            "V20082": "Ano_Nascimento",
            "V2010":"Cor",
            "V3003":"Escolaridade",
            "V403311":"Renda",
            "VD4001":"FTrabalho",
            "VD4012":"Setor_Ocupacao"
        }
            
        col_filter = [
        "Ano",
        "Trimestre",
        "UF",
        "Sexo",
        "Cor",
        "Escolaridade",
        "Renda",
        "FTrabalho",
        "Setor_Ocupacao"
        ]

        df = data.download_data(year=year,quarter=quarter,cols=columns)

        df_rename = df.rename(columns=columns_rename)

        df = df_rename[col_filter]

        return df
    



        
    