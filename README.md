# Pseudo-painel PNAD

Projeto para extração, transformação e análise dos microdados da PNAD Contínua, com espaço reservado para uma futura interface de visualização.

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
├── frontend/               # dashboard HTML e futura interface de visualização
└── scripts/                # utilitários auxiliares
```

Os notebooks Jupyter são mantidos apenas como arquivos locais de trabalho e estão ignorados pelo Git. Consulte o [README do pipeline](pnad_pipeline/README.md) para instalação, execução e testes.

O dashboard atual está em [frontend/dashboard_salario_hora.html](frontend/dashboard_salario_hora.html) e apresenta as métricas dos modelos, incluindo o M5 reproduzido do notebook `analise`.
