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

A preparação analítica aplica o intervalo inclusivo de **25 a 65 anos** (`V2009`), valida as variáveis-chave do dicionário, cria o salário-hora com rendimento habitual (`VD4016`) e horas habituais (`VD4031`), identifica ocupação (`VD4002`) e deriva setor, formalidade, escolaridade, faixa etária, atividade econômica e região.

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

Após a coleta, execute o tratamento da base analítica para indivíduos de 25 a 65 anos:

```bash
python scripts/02_tratar_pnad.py
```

Esse comando gera `data/processed/pnad_analise_25_65.parquet` e atualiza as quatro tabelas em `data/processed/tabelas_descritivas/` usadas nas análises do notebook.

O mesmo fluxo está documentado no notebook local `notebooks/02_tratar_pnad.ipynb`. Notebooks permanecem ignorados pelo Git conforme a política do repositório.

Para comparar modelos de estimação do salário-hora:

```bash
python scripts/03_modelar_salario_hora.py
```

O script compara OLS, OLS com efeitos fixos, regressão quantílica, modelo multinível por coorte, splines com Ridge, Random Forest e Gradient Boosting. Os resultados ficam em `data/processed/modelos_salario_hora/`.

Para gerar os coeficientes lineares do M5 ampliado com Ridge:

```powershell
python scripts/05_coeficientes_m5_ampliado_ridge.py
```

Para estimar a especificação agrupada que separa o efeito médio da macroregião dos desvios de cada estado e testa interações regionais:

```powershell
python scripts/06_modelar_m5_hierarquico_ridge.py
```

Essa especificação seleciona o `alpha` na validação temporal e grava as métricas, o resumo do modelo e os coeficientes em `data/processed/modelos_salario_hora/`.

Para executar as análises complementares recomendadas para a especificação regional:

```powershell
python scripts/07_analises_complementares.py
```

O script estima Ridge separada por macroregião, compara a especificação pooled com as regressões separadas, testa interações regionais, calcula decomposições Oaxaca–Blinder para sexo e cor e registra a sensibilidade de seleção. Também documenta quando uma análise não é identificável com a base disponível: neste recorte consolidado não há pesos amostrais, instrumento exógeno documentado ou grupo de não ocupados para estimar Heckman. Os resultados ficam em `data/processed/modelos_salario_hora/`, incluindo `analises_complementares.md`, `regressoes_por_macroregiao.csv`, `comparacao_pooled_separado.csv`, `oaxaca_decomposicao.csv` e `teste_interacoes_regionais.json`.

Para estimar a versão inferencial em todas as observações elegíveis, com uso conservador de memória:

```powershell
python scripts/08_inferencia_m5.py --batch-size 100000
```

O script estima OLS para `log(Salario_Hora)` no conjunto geral e em cada macroregião. Ele compara HC1, cluster por UF e cluster em duas vias por coorte e UF; a última é a especificação principal. A matriz de desenho é processada em lotes e apenas `X'X`, `X'y` e os acumuladores das variâncias são mantidos em memória. Os p-valores e intervalos de confiança estão nos arquivos `inferencial_m5_geral.csv` e `inferencial_m5_*.csv`. A causalidade é auditada em `auditoria_causalidade_m5.json`, mas não é afirmada: faltam instrumento exógeno, experimento natural ou painel individual identificável.

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
