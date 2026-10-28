# Pseudo-painel PNAD

Projeto para extração, transformação e análise dos microdados da PNAD Contínua, com dashboard interativo local integrado ao pipeline Python.

## Dashboard interativo

No Windows, dê dois cliques em **abrir_dashboard.cmd** na raiz do projeto.
O inicializador mantém o servidor em segundo plano e abre o navegador.
Novas aberturas reutilizam a instância ativa. Para encerrar, use
**fechar_dashboard.cmd**.

Pelo terminal, se necessário:

```powershell
python -m pip install -r pnad_pipeline/requirements.txt
python scripts/abrir_dashboard.py
```

Abra **http://127.0.0.1:8000**. O servidor entrega a interface e a API no
mesmo endereço. Abrir `frontend/index.html` diretamente mostra as instruções
de inicialização; a consulta de dados depende do servidor Python.

A aplicação inclui visão geral, descritivas com filtros e comparação de dois
grupos/períodos, previsão por perfil, coeficientes do M5 ponderado e metodologia.
Gráficos, busca, ordenação e exportações CSV/JSON usam resultados locais reais.

Para gerar ou atualizar o modelo operacional e seu pré-processamento:

```powershell
python pnad_pipeline/scripts/12_treinar_previsao.py
```

O comando preserva o benchmark histórico e salva a nova execução em
`pnad_pipeline/data/processed/previsao_perfis/`. Consulte as funcionalidades,
os domínios e as limitações no [README da interface](frontend/README.md).

O dashboard já usa a série ponderada de 2013–2025 com pesos e desenho amostral.
Para reproduzir a coleta, o tratamento e o M5, siga a seção de
[recoleta ponderada](pnad_pipeline/README.md). Os arquivos gerados permanecem
locais e o benchmark preditivo histórico continua separado.

## Organização

```text
.
├── pnad_pipeline/
│   ├── src/pnad_pipeline/
│   │   ├── extracao/       # coleta e leitura dos dados
│   │   ├── transformacao/  # limpeza e agregações
│   │   └── analise/
│   │       ├── descritiva.py
│   │       └── regressao/   # modelos de regressão e inferência
│   ├── data/
│   │   ├── raw/            # fontes e dicionários
│   │   └── processed/      # saídas geradas localmente
│   ├── scripts/
│   └── tests/
├── frontend/               # dashboard interativo e relatórios HTML anteriores
└── scripts/                # utilitários auxiliares
```

Os notebooks Jupyter são mantidos apenas como arquivos locais de trabalho e estão ignorados pelo Git. Consulte o [README do pipeline](pnad_pipeline/README.md) para instalação, execução e testes.

A interface interativa começa em [frontend/index.html](frontend/index.html),
servida pelo comando acima. O [benchmark HTML anterior](frontend/dashboard_salario_hora.html)
continua disponível, com as métricas históricas e o M5 reproduzido do notebook `analise`.
