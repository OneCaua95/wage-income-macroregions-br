# PNAD Pipeline

Pipeline modular em Python para extração, transformação, carga e análise de dados da **PNAD Contínua (IBGE)** com deflação via **IPCA (Banco Central do Brasil - SGS)**.

---

## Estrutura do Pacote

```
pnad_pipeline/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   └── pnad_pipeline/
│       ├── __init__.py
│       ├── extracao/
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── ipca.py
│       │   ├── pnad_extractor.py
│       │   ├── filtros.py
│       │   ├── deflacao.py
│       │   ├── regiao.py
│       │   ├── coleta.py
│       │   └── io.py
│       ├── transformacao/   # limpezas adicionais e agregações
│       ├── carga/           # [stub] carga em data warehouse / lakes
│       └── analise/
│           ├── descritiva.py
│           └── regressao/   # modelos de regressão e inferência
├── scripts/
│   └── 01_coletar_pnad.py
├── tests/
│   ├── conftest.py
│   └── extracao/
│       ├── test_ipca.py
│       ├── test_filtros.py
│       ├── test_deflacao.py
│       ├── test_regiao.py
│       ├── test_coleta.py
│       └── test_io.py
└── data/
    ├── raw/
    │   └── dicionario_PNADC_microdados_trimestral.xls
    └── processed/
```

Os notebooks `.ipynb` são arquivos de trabalho locais e estão fora do versionamento do GitHub. A estrutura do repositório fica organizada em quatro áreas principais:

- `extracao/`: coleta e leitura dos microdados e indicadores externos;
- `transformacao/`: limpeza, padronização e agregações;
- `analise/`: estatística descritiva e, em `analise/regressao/`, modelos de regressão;
- `frontend/`: pasta reservada para a futura interface de visualização.

## Dicionário de dados

O arquivo `data/raw/dicionario_PNADC_microdados_trimestral.xls` contém o dicionário dos microdados trimestrais da PNAD Contínua e deve ser consultado para conferir nomes, descrições e categorias das variáveis usadas no pipeline.

---

## Configuração

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. (Opcional) Copie o `.env.example` para `.env` e configure conforme sua necessidade:
   ```bash
   cp .env.example .env
   ```

---

## Execução via CLI

Para rodar a extração completa ou de um período específico:

```bash
python scripts/01_coletar_pnad.py --ano-inicio 2013 --ano-fim 2025
```

Argumentos disponíveis:
* `--ano-inicio`: Ano inicial da coleta (padrão: 2013)
* `--ano-fim`: Ano final da coleta (padrão: 2025)
* `--saida-parquet`: Caminho de persistência Parquet (padrão: `data/processed/pnad_consolidada_2013_2025.parquet`)
* `--saida-stata`: Caminho de persistência Stata .dta (padrão: `data/processed/pnad_descritiva_2013_2025.dta`)
* `--log-level`: Nível de detalhe do logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

---

## Testes Automatizados

Para rodar os testes unitários isolados com dados sintéticos:

```bash
pytest
```
