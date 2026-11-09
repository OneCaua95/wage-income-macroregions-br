# PNAD Pipeline

## Aplicação interativa local

Na raiz do repositório:

```powershell
python -m pip install -r pnad_pipeline/requirements.txt
python pnad_pipeline/scripts/12_treinar_previsao.py
python pnad_pipeline/scripts/13_servir_dashboard.py
```

Abra http://127.0.0.1:8000. O treinamento é necessário apenas quando o
artefato operacional estiver ausente ou precisar de atualização. Ele salva
modelo e pré-processamento juntos em `data/processed/previsao_perfis/`, com
manifesto de validação, amostras, versões e hash. A seleção usa exclusivamente
a validação 2023–2024; o teste 2025 avalia apenas o vencedor. O histórico
anterior é preservado. Veja [frontend/README.md](../frontend/README.md).

O script 03 agora grava novas execuções em
`modelos_salario_hora/benchmark_validacao/` e seleciona pela validação; arquivos
históricos do diretório pai mantêm sua identificação original. A idade é
convertida para float64 antes do quadrado para evitar overflow do int8 no
Parquet compacto. As métricas históricas não foram recalculadas após essa
correção e continuam identificadas como resultados exploratórios anteriores.

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
│   ├── 01_coletar_pnad.py
│   └── 14_reextrair_historico_ponderado.py
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

A coleta preserva a **amostra completa**, incluindo `V1028` (peso final trimestral calibrado), `Estrato`, `UPA`, `Ano` e `Trimestre`. O peso é mantido em precisão de 64 bits, sem deflação ou arredondamento. Os filtros de renda e idade não são aplicados na coleta, para preservar UPAs sem pessoas no domínio analítico.

A preparação analítica aplica o intervalo inclusivo de **25 a 65 anos** (`V2009`), preserva o desenho amostral, cria o salário-hora com rendimento habitual (`VD4016`) e horas habituais (`VD4031`), identifica ocupação (`VD4002`) e deriva setor, formalidade, escolaridade, faixa etária, atividade econômica e região. O salário-hora usa ocupados com renda e horas positivas; a taxa de formalidade usa apenas ocupados classificados como formais ou informais. Esses domínios não representam toda a população brasileira.

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

Para refazer a série histórica ponderada sem substituir os arquivos anteriores,
use a rotina retomável. Ela valida cada ano antes de consolidar a nova base e
preserva V1028, Estrato e UPA:

```powershell
python scripts/14_reextrair_historico_ponderado.py --ano-inicio 2013 --ano-fim 2025 --workers 4
python scripts/02_tratar_pnad.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida data/processed/recoleta_ponderada/pnad_analise_25_65.parquet `
  --tabelas-dir data/processed/recoleta_ponderada/tabelas_descritivas
python scripts/08_inferencia_m5.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral `
  --upa-unica ajustar `
  --upa-multiestrato separar
```

As partes anuais validadas e os registros de coleta ficam em
`data/processed/recoleta_ponderada/`; uma execução interrompida pode ser
retomada com o mesmo comando. O arquivo histórico original permanece intacto.

Após a coleta, execute o tratamento da base analítica para indivíduos de 25 a 65 anos:

```bash
python scripts/02_tratar_pnad.py
```

Esse comando trata a base em lotes, gera `data/processed/pnad_analise_25_65.parquet` e atualiza as quatro tabelas em `data/processed/tabelas_descritivas/`. Médias, percentis, taxas e médias por escolaridade usam `V1028`. Os percentis são a inversa da distribuição empírica ponderada, sem interpolação. Também é gerado `representatividade_trimestral.csv`, com tamanho amostral, total populacional do domínio, UPAs, estratos e tamanho efetivo de Kish por trimestre. O indicador de Kish mede apenas a concentração dos pesos; não substitui a variância com UPA e estrato.

**Bases antigas:** se faltarem `V1028`, `Estrato` ou `UPA`, o tratamento e a inferência interrompem a execução com orientação de recoleta. Não há fallback para pesos iguais. É necessário executar novamente a coleta com o código atualizado e depois o tratamento. Os dados e resultados já produzidos não ganham representatividade apenas com a alteração do código; devem ser regenerados. Use os microdados atuais do IBGE para manter uma única versão de ponderação em toda a série.

O mesmo fluxo está documentado no notebook local `notebooks/02_tratar_pnad.ipynb`. Notebooks permanecem ignorados pelo Git conforme a política do repositório.

Para comparar modelos de estimação do salário-hora:

```bash
python scripts/03_modelar_salario_hora.py
```

O script compara OLS, OLS com efeitos fixos, regressão quantílica, modelo multinível por coorte, splines com Ridge, Random Forest e Gradient Boosting. Os resultados ficam em `data/processed/modelos_salario_hora/`. Os scripts **03 a 07 são benchmarks exploratórios não ponderados**; para estimativas do M5 que incorporem peso e desenho amostral, utilize o script 08 abaixo.

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

O script estima Ridge separada por macroregião, compara a especificação pooled com as regressões separadas, testa interações regionais, calcula decomposições Oaxaca–Blinder para sexo e cor e registra a sensibilidade de seleção. Essas rotinas exploratórias não usam os pesos amostrais; a presença deles na base não muda esse comportamento. Não há instrumento exógeno documentado. Os resultados ficam em `data/processed/modelos_salario_hora/`, incluindo `analises_complementares.md`, `regressoes_por_macroregiao.csv`, `comparacao_pooled_separado.csv`, `oaxaca_decomposicao.csv` e `teste_interacoes_regionais.json`.

Para estimar a versão inferencial em todas as observações elegíveis, com uso conservador de memória:

```powershell
python scripts/08_inferencia_m5.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral `
  --batch-size 100000 `
  --upa-unica ajustar `
  --upa-multiestrato separar
```

O script estima uma regressão linear para `log(Salario_Hora)` **ponderada por `V1028`**, no conjunto geral e em cada macroregião, usando todas as observações elegíveis. O desenho é registrado na amostra completa antes dos filtros de domínio. São acumulados `X'WX`, `X'Wy` e scores ponderados, em lotes.

A inferência principal usa **linearização por UPA dentro de estrato**, com a correção `m/(m-1)` e graus de liberdade iguais ao número de UPAs menos o número de estratos. HC1 e agrupamentos por UF, coorte e duas vias permanecem como sensibilidades, também ponderadas. Estratos com uma única UPA geram erro por padrão. A opção explícita `--upa-unica ajustar` centra o score desses estratos na média global das UPAs, conforme `survey.lonely.psu="adjust"`; não presume variância zero ou seleção com certeza. A política e o número de estratos afetados ficam registrados no resumo.

Colunas constantes ou redundantes são retiradas da estimação com identificação de posto e permanecem marcadas como não identificáveis nos CSVs. Isso permite executar uma única onda sem atribuir coeficientes e p-valores a dummies de ano/trimestre constantes.

No empilhamento de trimestres, a mesma UPA permanece agrupada ao longo do tempo enquanto mantém o estrato. A variância é uma **aproximação com reposição**, sem correção de população finita ou recalibração dos pesos finais; não reproduz integralmente as variâncias oficiais do IBGE. Se uma UPA aparecer em mais de um estrato, a execução falha por padrão. A opção explícita `--upa-multiestrato separar` trata cada combinação `(UF, Estrato, UPA)` como uma versão de conglomerado distinta e registra a reclassificação; não estima a covariância entre as versões. A mudança da amostra mestra iniciada em 2025T3 e o esquema rotativo da PNAD também limitam a interpretação da variância empilhada. A soma dos pesos de vários trimestres representa pessoas-trimestres, não pessoas únicas; dividir todos os pesos pelo mesmo número de períodos não altera coeficientes, médias ou sua variância.

Os resultados ficam em **`data/processed/recoleta_ponderada/inferencia_amostral/`** para a série completa. Os arquivos `inferencial_m5_*.csv` incluem erros-padrão, p-valores e intervalos com o sufixo `desenho_pnad`, além do ajuste Benjamini–Hochberg do método principal. Os arquivos `perfil_idade_m5_*.csv` contêm a curva de salário-hora previsto para o perfil médio ponderado, com IC pontual pela covariância completa do desenho. O relatório e o resumo registram o domínio, o peso, a aproximação da variância e os diagnósticos. O dashboard interativo prioriza a nova base histórica ponderada e lê os resultados do M5 desse diretório após reiniciar o servidor. A ponderação não identifica causalidade.

### Retorno associado à escolaridade

Para comparar formas da escolaridade e testar diferenças de perfil com os mesmos pesos e a mesma amostra:

```powershell
python scripts/15_inferencia_retorno_educacao.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral/retorno_educacao `
  --batch-size 100000 `
  --upa-unica ajustar `
  --upa-multiestrato separar
```

O script estima escolaridade por categorias VD3005, tendência linear e spline linear por segmentos (nós pré-fixados em 4, 8 e 11 anos), além de perfis spline por sexo, cor/raça e macrorregião. Apresenta diferenças acumuladas contra VD3005=0 e retornos marginais entre níveis adjacentes, com intervalos de 95%, testes Wald-F conjuntos e ajuste BH. VD3005=16 significa 16 anos ou mais; nas tendências é aproximado como 16. Os efeitos se referem ao salário-hora de pessoas ocupadas, são associações condicionais e não identificam causalidade. O diretório mantém CSV, resumo JSON e relatório Markdown; o dashboard lê os resultados após reiniciar o servidor.

### Seleção ocupacional e salário-hora observável

Para analisar quem entra na regressão de salário-hora, separando ocupação de observação de um rendimento e horas válidos:

```powershell
python scripts/16_selecao_ocupacional.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --entrada-analise data/processed/recoleta_ponderada/pnad_analise_25_65.parquet `
  --efeitos-educacao data/processed/recoleta_ponderada/inferencia_amostral/retorno_educacao/efeitos_retorno_educacao.csv `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral/selecao_ocupacional `
  --batch-size 100000 `
  --upa-unica ajustar `
  --upa-multiestrato separar
```

O evento de seleção é pertencer à mesma amostra salarial do M5: ocupação com rendimento e horas habituais positivos. O modelo de probabilidade usa V1028, escolaridade, sexo e interações sexo–escolaridade, cor/raça, idade, UF, ano e trimestre; os pesos inversos são limitados a 20. A incerteza do modelo salarial IPW incorpora a estimação da probabilidade por scores empilhados e linearização UPA/estrato. O alvo é a população de 25–65 anos com covariáveis observadas válidas; as taxas de ocupação e de salário válido são divulgadas separadamente.

O IPW é uma sensibilidade sob a hipótese de seleção explicada pelas variáveis observadas. As colunas disponíveis não fornecem uma restrição de exclusão defensável para Heckman, e seleção por fatores não observados permanece sem identificação. Os resultados não são causais. A saída contém `taxas_selecao.csv`, `sensibilidade_ipw_escolaridade.csv`, `resumo_selecao_ocupacional.json` e `relatorio_selecao_ocupacional.md`; a execução grava checkpoint para retomada. O dashboard expõe as taxas e a comparação após reiniciar o servidor.

### Heterogeneidade regional do salário-hora

Para testar diferenças regionais na associação da escolaridade e da formalidade e comparar especificações pooled e regionais:

```powershell
python scripts/17_heterogeneidade_regional.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral/heterogeneidade_regional `
  --batch-size 100000 `
  --upa-unica ajustar `
  --upa-multiestrato separar
```

O modelo conjunto deixa variar por macroregião os retornos categóricos da escolaridade e a associação da formalidade; Norte é a referência das interações. Testes Wald-F conjuntos usam a mesma covariância linearizada por UPA/estrato e os dois testes primários recebem ajuste Benjamini–Hochberg. Cinco regressões adicionais estimam todos os controles separadamente por macroregião. RMSE log e R² ponderados comparam o ajuste na própria amostra, sem interpretação como desempenho preditivo. As saídas incluem efeitos percentuais e IC95% por região, associações de formalidade, tabela de ajuste, resumo e relatório Markdown. As estimativas são condicionais entre ocupados; a variância empilhada é uma aproximação temporal, e os resultados não são causais.

### Pseudo-painel por coortes

O passo seguinte organiza os cortes transversais repetidos em médias de coortes fixas de sexo × faixa de nascimento, por trimestre. A faixa de nascimento preserva a codificação usada pelo M5 histórico; escolaridade, formalidade, setor público, cor/raça e região entram como médias/proporções que podem variar em cada célula. Assim, mudanças de escolaridade ou região não redefinem a coorte.

```powershell
python scripts/18_pseudopainel_coortes.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral/pseudopainel `
  --batch-size 100000
```

O desfecho principal é a média ponderada de log do salário-hora real, usando V1028 dentro da célula. A regressão WLS usa o efetivo de Kish como peso de precisão aproximado. São comparados efeitos fixos de coorte e trimestre; o mesmo modelo com controles de composição; e uma especificação com tendência linear diferencial por coorte, relativa à coorte de referência. A especificação principal exige pelo menos 100 registros e Kish 50 em cada célula, além de oito trimestres elegíveis por coorte. As sensibilidades usam (n, Kish) de (200, 100) e (500, 250); esses cortes são escolhas do estudo, não regras universais.

Os resultados ficam em `medias_coorte_trimestre.csv`, `coeficientes_pseudopainel.csv`, `comparacao_especificacoes.csv`, `resumo_pseudopainel.json` e `relatorio_pseudopainel.md`. O dashboard lê os artefatos ao reiniciar o servidor e não reestima as especificações com filtros globais.

As médias de coorte têm erro amostral e podem atenuar associações ou contaminar tendências. Os erros-padrão clusterizados por coorte e trimestre são exploratórios, têm poucos grupos de coorte e não substituem a linearização oficial da PNAD por UPA/estrato. A rotação da pesquisa também limita a covariância temporal. O pseudo-painel descreve médias de grupos; não acompanha as mesmas pessoas e não identifica retorno individual da escolaridade ou causalidade. Para a construção clássica de coortes fixas e os problemas de erro de amostragem em células pequenas, veja [Deaton (1985)](https://www.princeton.edu/~deaton/downloads/Panel_Data_from_Time_Series_of_Cross_Sections.pdf) e [Khan (2021)](https://onlinelibrary.wiley.com/doi/abs/10.1111/obes.12416).

### Desigualdade salarial e regressões quantílicas

```powershell
python scripts/19_desigualdade_quantis.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral/desigualdade `
  --batch-size 100000 `
  --amostra-quantis 30000
```

Percentis P10, P25, P50, P75 e P90, a razão P90/P10, Gini e pontos da curva de Lorenz são calculados com V1028 sobre todo o arquivo, por trimestre e para Brasil, macroregião, sexo e cor/raça. O domínio é o salário-hora real de pessoas ocupadas de 25 a 65 anos com renda e horas habituais positivas; os valores estão em reais de janeiro/2026. O Gini não mede renda domiciliar.

As regressões quantílicas de log(salário-hora) comparam categorias de escolaridade ao código 0 e controlam idade, sexo, cor/raça, macroregião, formalidade, setor público, ano e trimestre. O ajuste inicial é ponderado por V1028 e estratificado por trimestre × macroregião; duas correções dos scores percorrem a série completa. Os intervalos usam scores por UPA dentro de estrato e uma densidade escalar aproximada, sem FPC nem recalibração dos pesos. São aproximações para esta regressão quantílica e não substituem variâncias oficiais específicas para amostras complexas. Não há interpretação causal.

As saídas são `distribuicao_desigualdade.csv`, `curva_lorenz.csv`, `coeficientes_quantis.csv`, `resumo_desigualdade.json` e `relatorio_desigualdade.md`. O dashboard mostra a distribuição histórica ponderada na aba **Desigualdade salarial** e adiciona as regressões quantílicas à aba **Associação e inferência**. Os filtros demográficos globais não reestimam essas análises.

Referências metodológicas: [IBGE — notas técnicas da PNAD Contínua, planejamento e medidas de precisão](https://www.ibge.gov.br/biblioteca/visualizacao/livros/liv101640.pdf), [dicionário e input trimestrais](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/Documentacao/) e [survey — regressões ponderadas e graus de liberdade](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyglm.html).

### Diferenças salariais brutas e ajustadas por grupos

```powershell
python scripts/20_diferencas_grupos_salariais.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral/diferencas_grupos `
  --batch-size 100000
```

O script usa a mesma amostra elegível, V1028 e desenho do M5 histórico. Para sexo e cor/raça, as diferenças brutas são contrastes entre médias ponderadas de log(salário-hora); as ajustadas são contrastes padronizados do M5 na distribuição comum de covariáveis. O contraste por sexo incorpora a interação com escolaridade na média ponderada de anos de estudo. Percentuais são calculados por 100 × (exp(Δ log) − 1), com intervalos de 95% por linearização UPA/estrato. Os resultados são associativos e não medem automaticamente discriminação ou causalidade.

As saídas são `diferencas_brutas_ajustadas.csv`, `resumo_diferencas_grupos.json` e `relatorio_diferencas_grupos.md`. O dashboard mostra o gráfico e a tabela na aba **Associação e inferência**, após selecionar a série histórica ponderada.

### Decomposição Oaxaca–Blinder ponderada

```powershell
python scripts/21_decomposicao_oaxaca.py `
  --entrada data/processed/recoleta_ponderada/pnad_consolidada_2013_2025.parquet `
  --saida-dir data/processed/recoleta_ponderada/inferencia_amostral/oaxaca `
  --batch-size 100000 `
  --upa-unica ajustar `
  --upa-multiestrato separar
```

Usa todas as observações elegíveis do M5 histórico com V1028 para cinco pares:
mulheres versus homens, e pessoas pretas, amarelas, pardas e indígenas versus
brancas. Para cada par, estima a referência pooled com indicador de grupo,
conforme [Jann (2008)](https://repec.ethz.ch/ets/papers/jann_oaxaca.pdf).
O vetor de controles comuns mantém a família M5, retirando a variável definidora
do par e sua interação com escolaridade. O indicador entra somente no ajuste
pooled. A diferença é sempre grupo menos referência.

O total D é a diferença entre médias ponderadas de log do salário-hora;
a parcela explicada E é a diferença de médias dos controles multiplicada
pelos coeficientes pooled; a parcela não explicada é U = D − E. Os componentes
somam em log. A transformação percentual é apresentada apenas para o total,
pois percentuais exponenciados dos componentes não somam. Os ICs pontuais
usam scores conjuntos de médias e coeficientes, agregados por UPA/estrato,
preservando a covariância entre as parcelas. Cadastro completo de UPAs,
políticas explícitas e limites da variância temporal seguem o M5.

Saídas: `decomposicao_oaxaca.csv`, `covariancia_oaxaca.npz` (com a ordem dos
componentes), `resumo_oaxaca.json` e `relatorio_oaxaca.md`. O dashboard lê a
análise em `/api/oaxaca` e a exibe em **Associação e inferência**, com seletor
de comparação, gráfico, tabela, CSV e inclusão na exportação JSON. O resultado
é fixo na série 2013–2025; filtros globais e período não reestimam os modelos.

A parcela não explicada inclui fatores não observados e depende dos controles
e da referência; não representa automaticamente discriminação ou causalidade.
Formalidade, setor e ocupação podem mediar diferenças. A parcela explicada
também é estatística, sem atribuição causal às características. Não se reporta
uma fração explicada quando o total é próximo de zero.

### Diagnósticos da previsão por grupo

```powershell
python scripts/22_diagnosticos_previsao.py
```

O script usa `previsao_perfis/manifesto.json` e o pipeline operacional já
exportado. Reconstrói somente o teste de 2025 com a semente original (o fluxo
aleatório preserva o índice do conjunto), confere o tamanho da base e exige
igualdade numérica com MAE, RMSE e R² em log e em reais do manifesto. Não há
novo ajuste ou seleção do modelo. O script 12 também exporta os diagnósticos
em novos treinos, a partir das mesmas previsões usadas nas métricas de teste.

O arquivo `diagnostico_<run>_<hash>.json` contém métricas gerais e por região,
sexo, faixas de anos de estudo e decis do salário observado. Viés é a média
de previsto menos observado. Limites dos decis são fechados à direita e empates
permanecem juntos; limites repetidos são unidos. Grupos com menos de 30
registros têm métricas suprimidas, preservando as contagens. Os pontos do
gráfico são médias de até 100 faixas quantis, com pelo menos 30 registros;
os microdados não são publicados pela API.

O manifesto recebe a referência e o SHA-256 do diagnóstico, e o arquivo registra
o identificador/hash do modelo e a impressão da amostra ordenada. A API
`/api/diagnostico-previsao` valida essa correspondência. A aba **Previsão**
exibe erros por grupo, seletor de métrica/escala, observado versus previsto,
tabelas e CSV. A exportação JSON do recorte inclui o diagnóstico carregado.

São resultados descritivos não ponderados na amostra reservada de 2025, com
salário-hora a preços de janeiro/2026. A conversão original prevê `exp(log previsto)`
sem smearing; a extensão abaixo avalia também a conversão corrigida. Diferenças de
erro entre grupos também dependem de suas distribuições salariais. Os
diagnósticos não identificam causalidade, não calibram intervalos de previsão
e não validam projeções futuras. Filtros populacionais não alteram essa avaliação.

### Correção global de Duan e comparação temporal

```powershell
python scripts/23_avaliar_duan.py
```

O pipeline já exportado é mantido fixo. O script reproduz as seis métricas da
validação original 2023–2024 e separa suas 50.000 observações por ano. Em 2023,
calcula `S = média(exp(log observado − log da conversão base))`; a conversão
corrigida é `exp(log previsto) × S`. A conversão base mantém o limite numérico
do serviço em `[-20, 20]` no log; o fator é aplicado após essa conversão, sem
truncar resíduos ou ajustar limites usando resultados de teste.

Critério pré-fixado: recomendar Duan se o RMSE em reais diminuir e o viés
absoluto não aumentar em 2024. MAE, erros em log e resultados por grupo são
diagnósticos. A decisão é persistida em `decisao_duan_<run>_<hash>.json` antes
de reconstruir a amostra de teste de 2025. O relatório posterior fica em
`comparacao_duan_<run>_<hash>.json`; o manifesto recebe seu caminho e hash.
O relatório inclui as duas conversões nos dois períodos, por grupo e em
faixas de observado/previsto, com as mesmas contagens e impressões de amostra.

A API `/api/duan` verifica o vínculo com o pipeline e o hash do registro
congelado. `/api/previsao` aplica a recomendação da validação ao campo
`salario_hora` e retorna também `salario_hora_original`, `salario_hora_duan`,
`fator_duan`, `retransformacao` e o hash da decisão. As métricas de teste do
resultado acompanham a conversão usada; as originais são preservadas em campo
separado. A interface mostra as duas estimativas, a comparação geral e por
grupos, permite alternar os diagnósticos e exporta CSV/JSON com proveniência.

Nesta execução: 24.416 registros em 2023, 25.584 em 2024 e 50.000 no teste de
2025. O fator foi 1,21226216 e passou pelo critério de 2024. A conversão melhora
RMSE e viés em reais, mas piora MAE e métricas em log; os dois alvos e essa
troca são apresentados explicitamente. O pipeline base, seu identificador e
hash não mudam. Novos treinos do script 12 precisam de nova avaliação pelo 23.

O fator global pressupõe transportabilidade da distribuição dos resíduos e
pode falhar em grupos com heterogeneidade. Esta é uma análise retrospectiva:
o modelo base já foi selecionado com 2023–2024 e os diagnósticos de 2025 já
foram inspecionados. Não representa uma nova avaliação independente, previsão
futura ou intervalo preditivo. Referência:
[Duan (1983), Smearing Estimate](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017).

## Validação real e relatório HTML

Para reproduzir a validação sobre uma onda oficial de 2025T4, executar a suíte do código e incorporar as evidências no HTML, a partir de `pnad_pipeline/`:

```powershell
python scripts/10_validar_dados_reais.py
python -m pytest -q --junitxml=data/processed/testes_codigo.xml
python scripts/11_gerar_html_testes_reais.py
```

O script 10 baixa `PNADC_042025.zip` por HTTPS, processa as 498.494 observações da amostra completa, preserva os pesos calibrados, calcula o IPCA com base em janeiro/2026 e executa o M5 em 184.594 observações elegíveis. A validação encontrou cinco estratos com uma UPA e registra o uso explícito de `ajustar`. Confere também os coeficientes em 12 mil registros reais contra `statsmodels.WLS`; essa conferência numérica não é uma nova estimativa populacional. O ZIP e a base processada são reutilizados em execuções seguintes; use `--reprocessar` para reler o TXT oficial.

O relatório autônomo fica em `../frontend/resultados_testes_reais.html`. Ele é um snapshot de 16/10/2026 que contém resultados ponderados de **2025T4**, a versão histórica disponível naquela data, verificações reais, a lista da suíte com fixtures sintéticas, hash SHA-256 do arquivo oficial e referências metodológicas. A série histórica ponderada concluída em 28/10/2026 é lida pelo dashboard interativo e está documentada em `frontend/validacao_dashboard.md`; o relatório autônomo não foi regenerado nesta execução.

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
# Intervalos de previsão · execução temporal separada

```powershell
python pnad_pipeline/scripts/24_intervalos_previsao.py
```

Usa `data/processed/pnad_analise_25_65.parquet`, sem ponderação, e salva em
`data/processed/previsao_intervalos/`, preservando os artefatos preditivos anteriores.
Parâmetros: `--entrada`, `--artefatos-dir`, `--max-treino` (150.000),
`--max-avaliacao` (50.000 por etapa) e `--seed` (42).

Protocolo persistido antes da execução: treino 2013–2021; seleção por RMSE log em
2022, entre duas configurações de Gradient Boosting, Random Forest e OLS; fator
global de Duan em 2023T1–T2; escolha da conversão por RMSE e viés absoluto em
2023T3–T4. Modelo e conversão são congelados antes de ler 2024. Não há refit.

Split conformal com escore absoluto em log em 2024: `k=ceil((n+1)*0.90)`;
raio é a k-ésima estatística de ordem, sem interpolação. Centro é
`clip(log previsto,-20,20)+log(fator aplicado)`; limites são `exp(centro ± raio)`.
O raio e seu hash são gravados antes de ler 2025. O teste usa semente 44,
preservando as mesmas 50 mil linhas da avaliação anterior; não ajusta nenhuma
decisão. Demais etapas usam sementes 42, 43, 45, 46 e 47, respectivamente.

`GET /api/intervalos-previsao` entrega relatório agregado e manifesto da nova
execução. `POST /api/previsao-intervalos` usa o mesmo esquema e derivação de
variáveis do serviço anterior, mas carrega exclusivamente seu próprio pipeline.
O serviço verifica SHA-256, execução e correspondência entre protocolo, decisão,
calibração e relatório; o pipeline também é verificado antes de uso. O ponteiro
`manifesto.json` é trocado atomicamente só ao concluir todas as etapas.

Cobertura inclusiva e larguras média/mediana/P90 são relatadas no geral, por
região, sexo e escolaridade em calibração e teste; grupos com n<30 mostram
apenas n. Nenhum microdado é publicado. Referência:
[Angelopoulos e Bates](https://arxiv.org/abs/2107.07511).

Execução `20261108T132604904108Z`: GradientBoosting_folhas31; Duan 1,2101498;
raio 1,0482523; calibração 90,002%; teste 90,736%; largura média R$47,5324/h.
A cobertura depende de hipóteses de permutabilidade não asseguradas por dados
temporais com rotação. Esta é avaliação retrospectiva, pois 2025 já foi visto
em extensões anteriores, sem garantia prospectiva, populacional ou por grupo.
Não confundir faixa individual com IC de médias/coeficientes.
