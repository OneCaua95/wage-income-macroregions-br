import pnadium

class Extractor:
    def __init__(self):
        pass

    def download_data(self, year, quarter, cols):
        data = pnad_filter = pnadium.trimestral.download(ano=year, t=quarter, colunas=cols)

        return data