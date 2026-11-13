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

### Robustez do M5 a salários extremos

```powershell
python scripts/26_robustez_m5.py
```

O script compara o M5 nacional de 2013–2025 com o mesmo modelo após limitar o
salário-hora real aos percentis 1 e 99 ponderados da amostra elegível. As duas
versões usam os mesmos registros, controles, pesos V1028, UPAs e estratos. O CSV
apresenta os contrastes padronizados de escolaridade (12 vs. 8 anos), formalidade
e sexo, além da mudança entre versões em pontos percentuais.

Os IC95% pontuais usam a linearização do desenho do M5. O intervalo da mudança
usa o método delta com a covariância entre os ajustes. A variância mantém fixos
os pontos de corte e as médias de covariáveis usadas nos contrastes. O CSV e o
resumo JSON ficam em `data/processed/recoleta_ponderada/inferencia_amostral/robustez_m5/`.

### Estabilidade anual do M5

```powershell
python scripts/27_estabilidade_m5.py
```

Um ajuste do M5 por ano completo usa a elegibilidade, os controles e os pesos
V1028 da série histórica. Os contrastes de escolaridade, formalidade e sexo
são avaliados na distribuição ponderada da série inteira, mantida como
referência comum. A tabela traz diferenças percentuais e IC95% pontuais.

A linearização por UPA e estrato estima a covariância conjunta entre os anos.
O teste F de Wald compara o contraste em log entre todos os anos, com ajuste de
Holm para os três testes. O perfil comum entra como fixo na inferência. A
aproximação da variância e o tratamento de reclassificação de UPAs seguem o M5
histórico. O CSV e o JSON ficam em
`data/processed/recoleta_ponderada/inferencia_amostral/estabilidade_m5/`.

### M5 com e sem efeitos fixos de UF

```powershell
python scripts/28_sensibilidade_uf_m5.py
```

O script ajusta as duas especificações na mesma amostra histórica ponderada,
com o salário-hora original e um perfil de referência comum. A versão reduzida
retira os indicadores de UF da regressão. A UF permanece na identificação de
estratos e UPAs do desenho amostral. O CSV mostra os contrastes de escolaridade,
formalidade e sexo, e a diferença em pontos percentuais calculada como sem UF
menos com UF.

Os IC95% pontuais tratam o perfil de referência como fixo. A linearização por
UPA e estrato estima a covariância entre os dois ajustes. O intervalo da mudança
usa o [método delta](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svycontrast.html).
O CSV e o JSON ficam em `data/processed/recoleta_ponderada/inferencia_amostral/sensibilidade_uf_m5/`.

### M5 com e sem controles do emprego

```powershell
python scripts/29_sensibilidade_emprego_m5.py
```

A versão reduzida retira ocupação, setor de atividade, setor público,
formalidade e as interações da escolaridade com setor público e formalidade.
O contraste compara 12 e 8 anos de estudo na mesma amostra ponderada, com
efeitos fixos de UF, ano e trimestre. A mudança em pontos percentuais é a
estimativa reduzida menos a estimativa completa. Os IC95% pontuais usam o
perfil de referência fixo e a covariância dos dois ajustes pelo desenho
amostral. A tabela descreve a sensibilidade da associação aos controles.
O CSV e o JSON ficam em `data/processed/recoleta_ponderada/inferencia_amostral/sensibilidade_emprego_m5/`.

### M5 com salário-hora real e nominal

```powershell
python scripts/35_comparar_nominal_real_m5.py
```

O script lê a data-base registrada no Parquet e recalcula o fator trimestral
da coleta com a série mensal SGS 433 do Banco Central. O log nominal resulta
de subtrair o log desse fator do log real arquivado. As duas regressões usam
os mesmos registros, pesos V1028 e controles do M5. A diferença é a estimativa
nominal menos a real em pontos percentuais. O IC95% pareado usa a covariância
dos ajustes por UPA dentro de estrato e trata o perfil de referência e o IPCA
como fixos. O JSON registra os fatores usados. As saídas ficam em
`data/processed/recoleta_ponderada/inferencia_amostral/nominal_real_m5/`.

### Múltiplos testes do M5 e mercado de trabalho

```powershell
python scripts/36_multiplicidade_m5.py
python scripts/37_mercado_trabalho.py
python scripts/19_desigualdade_quantis.py --somente-distribuicao
```

O script 36 estima testes Wald-F para dez blocos de coeficientes do M5,
com Holm na família de blocos identificados. Os três contrastes principais
formam outra família, com IC95% simultâneos de Bonferroni e perfil ponderado
tratado como fixo. As saídas ficam em `inferencia_amostral/multiplicidade_m5/`.
A tabela principal do dashboard oferece Holm para a família completa de
coeficientes identificados de cada ajuste, incluindo o intercepto, seguindo
a família BH histórica. Arquivos anteriores são compatíveis, pois o ajuste
é calculado sobre a tabela completa antes dos filtros de visualização.

O script 37 usa a base consolidada completa para estimar participação,
ocupação e desocupação entre 25 e 65 anos, por trimestre, Brasil,
macroregião, sexo e cor/raça. Usa VD4001 e VD4002, pesos V1028 e IC95%
pontuais por linearização de razões com cadastro completo de UPAs antes
do domínio. Participação e ocupação têm a população do domínio como
denominador. Desocupação tem a força de trabalho como denominador.
As saídas em `inferencia_amostral/mercado_trabalho/` registram contagens,
totais ponderados, exclusões e denominadores. O dashboard apresenta essas
taxas na Visão geral. A faixa etária é diferente da população de 14 anos
ou mais usada nas divulgações nacionais do IBGE.

A opção `--somente-distribuicao` requer uma execução quantílica anterior e
atualiza apenas a distribuição e Lorenz. Acrescenta Theil T ponderado,
P50/P10, P90/P50 e participação dos 10% com maiores salários-hora.
O topo usa exatamente 10% da massa V1028, fracionando o peso na fronteira.
As medidas se referem ao salário-hora dos ocupados do domínio M5 e têm
série trimestral por grupo. O Theil usa a convenção `0 log(0) = 0`.

### Intervalos descritivos, coortes alternativas e Theil regional

```powershell
python scripts/38_incerteza_descritiva.py
python scripts/39_sensibilidade_coortes.py
python scripts/40_decompor_theil.py
```

O script 38 usa o salário-hora arquivado na base descritiva e o cadastro de
UPAs da base completa, anterior aos filtros de domínio. Estima a média,
P10, P50 e P90 por trimestre para Brasil e cinco macroregiões. A média usa
linearização de razão e distribuição t. Os quantis seguem Woodruff com
intervalo t da CDF no quantil e inversa empírica ponderada, preservando
empates. Limites fora do suporte de probabilidade são indisponíveis.
As faixas são pontuais. A API `/api/incerteza-descritiva` e o painel da
Visão geral usam os arquivos em `inferencia_amostral/incerteza_descritiva/`.
O domínio descritivo exige ocupação, renda e horas positivas e idade de
25 a 65 anos. Os controles adicionais de elegibilidade do M5 definem
a amostra das análises do pseudo-painel e do Theil regional.
[Referência dos intervalos de quantis](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyquantile.html).

O script 39 mantém a amostra salarial do M5 e compara as faixas históricas
de nascimento, faixas de cinco anos deslocadas e faixas decenais. As
bordas parciais preservam o suporte de 1947 a 2003 e os limites ficam
registrados no JSON. Reestima as três especificações e os três limiares
de n/Kish, totalizando 27 modelos. Cada célula recebe EP da média de
log salário e da escolaridade média pelo desenho trimestral, além do
IC95% da média geométrica. Esses EPs quantificam o erro amostral das
médias e são diagnósticos. As regressões mantêm a inferência exploratória
clusterizada por coorte e trimestre e não corrigem erro de mensuração.
As saídas ficam em `inferencia_amostral/sensibilidade_coortes/` e a API
`/api/sensibilidade-coortes` as apresenta na aba Associação e inferência.

O script 40 usa os totais ponderados de salário-hora por macroregião.
O Theil nacional soma a parcela dentro das regiões e a parcela entre
regiões. O componente interno pondera cada Theil regional pela participação
regional na soma ponderada dos salários-hora. Contribuições regionais
individuais ao componente entre podem ser negativas, enquanto a soma é
não negativa. Os totais são conferidos pela identidade aditiva. A API
`/api/decomposicao-theil` lê as saídas em `inferencia_amostral/decomposicao_theil/`
para a aba Desigualdade salarial. Esta decomposição é descritiva e os
componentes não recebem intervalos de confiança.

### Emprego ampliado e evolução anual por região

**Conferência do dicionário:** VD4031 mede horas habituais semanais em todos os
trabalhos; VD4016 mede renda habitual do trabalho principal. A proxy salarial
histórica é VD4016 / (VD4031 × 4,33), o que limita a interpretação para pessoas
com múltiplos vínculos. Esta entrega preserva a série e os modelos anteriores.
A jornada nova usa o nome correto de todos os trabalhos. A composição setorial
usa os códigos VD4010 diretamente, com 8 = administração pública, 9 = educação/
saúde, 10 = outros serviços, 11 = domésticos e 12 = atividades mal definidas.
Não reutiliza os rótulos antigos arquivados na base analítica. O mapeamento para
novas preparações também foi corrigido conforme o dicionário local do IBGE.

Execute a partir da raiz do repositório:

```powershell
python pnad_pipeline/scripts/37_mercado_trabalho.py
python pnad_pipeline/scripts/44_evolucao_regional_anual.py
```

O script 37 mantém as três taxas anteriores e acrescenta seis indicadores com
IC95% por linearização de razões, cadastro completo de UPAs, V1028 e distribuição
t trimestral. Conta própria (VD4009=9), empregados formais (1,3,5,7) e setor público
(5,6,7) usam ocupados com categoria válida. Informalidade é o complemento da
classificação de formalidade do projeto: carteira/categoria e contribuição
previdenciária para empregadores e conta própria, excluindo contribuição ausente.
Não é a [proxy oficial do IBGE baseada em CNPJ](https://agenciadenoticias.ibge.gov.br/agencia-sala-de-imprensa/2013-agencia-de-noticias/releases/45908-pnad-continua-trimestral-desocupacao-recua-em-seis-das-27-ufs-no-4-trimestre-de-2025).

Jornada usa horas habituais positivas em todos os trabalhos (VD4031), sem exigir
renda positiva. Rendimento usa renda habitual mensal positiva do trabalho principal
(VD4016), sem exigir horas positivas, em reais de janeiro/2026. Cada indicador
publica n e soma dos pesos do denominador, estimativa, IC e status. O mínimo é de
30 registros no denominador. Os seis indicadores não formam uma partição.

A composição setorial usa VD4010 e o total de ocupados como denominador, incluindo
Sem informação para códigos ausentes/não reconhecidos. As treze participações somam
100%, com intervalos pontuais. O JSON do script 37 inclui `composicao_setorial`;
a API `/api/mercado-trabalho` expõe `sector_rows` no trimestre e `all_sector_rows`
na série completa. A execução gerou 676 linhas de indicadores e 8.788 setoriais.

O script 44 usa a base descritiva, não a elegibilidade adicional do M5. Calcula
medianas empíricas ponderadas sobre os quatro trimestres de cada ano e crescimento
da mediana contra 2013 na mesma região. Não calcula média de medianas. Mantém
empates, exige quatro trimestres e 30 salários válidos por célula, e publica
contagens pessoa-trimestre e população salarial média (soma V1028 / 4). A série
tem 78 linhas: 13 anos × Brasil e cinco regiões. É descritiva, sem IC. A API
`/api/evolucao-regional-anual` ignora explicitamente período e filtros globais.

O mapa `/api/mapa-uf` acrescenta `informalidade_pct` com domínio independente do
salário, e `crescimento_media_pct`, média real contra 2013T1, com os mesmos filtros
nas duas amostras e mínimo de 30 salários em ambas. O filtro UF não limita o mapa.
A fonte real não recebe crescimento sem referência histórica. Há contagens dos
dois períodos, referência e critério de informalidade na proveniência/CSV. São
comparações descritivas sem IC; mudanças de composição e sazonalidade permanecem.

### Incerteza da desigualdade, crescimento e regressões RIF (scripts 41–43)

```powershell
python scripts/41_incerteza_desigualdade.py
python scripts/42_crescimento_salarial.py
python scripts/43_rif_desigualdade.py
```

O script 41 estima IC95% por trimestre para Brasil e macroregiões usando
a mesma amostra do M5 e da distribuição salarial histórica. Gini, Theil T
e participação dos 10% maiores salários usam funções de influência com
linearização por UPA/estrato. O topo preserva exatamente 10% da massa
ponderada, inclusive o fracionamento na fronteira. O intervalo de P90/P10
combina limites de Woodruff a 97,5% dos dois quantis por Bonferroni, formando
um intervalo conservador de 95%. Os intervalos são pontuais e respeitam o
suporte dos indicadores. As saídas ficam em `inferencia_amostral/incerteza_desigualdade/`.

O script 42 usa a amostra descritiva e compara médias aritméticas reais
por Brasil e macroregião. A série acumulada fixa a referência em 2013T1.
A série interanual compara o mesmo trimestre do ano anterior. Os scores
das médias são agregados nas UPAs comuns, e a variância conjunta usa o
cadastro completo da união dos dois períodos. O intervalo aplica o método
delta ao log da razão. A aproximação com reposição, a rotação e as UPAs
reclassificadas limitam a variância temporal estimável. As saídas ficam em
`inferencia_amostral/crescimento_salarial/`.
[Referência de contrastes](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svycontrast.html).

O script 43 estima regressões RIF para Gini e Theil na distribuição nacional
ponderada da série histórica empilhada. Usa os mesmos registros e controles
do M5 e publica os contrastes de 12 versus 8 anos de estudo, formalidade
e sexo. Gini é expresso em pontos da escala 0–100 e Theil em unidades
originais. A covariância inclui a resposta RIF gerada, com derivadas da
distribuição empírica, da média e dos índices, além dos scores da regressão.
As derivadas são verificadas por perturbações independentes dos pesos.
Os perfis de referência são tratados como fixos e Holm é aplicado aos seis
contrastes. A interpretação é associativa e depende da projeção linear.
As saídas ficam em `inferencia_amostral/rif_desigualdade/`.
[Firpo, Fortin e Lemieux (2018)](https://www.mdpi.com/2225-1146/6/2/28).

As APIs `/api/incerteza-desigualdade`, `/api/crescimento-salarial` e
`/api/rif-desigualdade` entregam os agregados e a proveniência. Os painéis
usam seletores locais, tabelas e CSV. A aba Associação e inferência organiza
as análises em resultados principais, diagnósticos, desigualdade e
decomposições, e pseudo-painel. A navegação aceita setas, Home e End e
mantém o grupo selecionado durante a sessão.

### Multicolinearidade do M5

```powershell
python scripts/30_multicolinearidade_m5.py
```

Na mesma amostra V1028 do M5, o diagnóstico centra e padroniza os termos
estimáveis da regressão. Para cada bloco calcula o fator generalizado de
inflação de variância (GVIF) e `GVIF^(1/(2d))`, com `d` igual ao número de
colunas estimáveis do bloco. O bloco de escolaridade inclui as interações e
seus termos principais. A versão reduzida usa as mesmas exclusões do script 29.
[Fox e Monette (1992)](https://doi.org/10.1080/01621459.1992.10475190).

O número de condição é a razão entre o maior e o menor valor singular da
matriz de regressoras ponderada e padronizada. As omissões seguem o critério
de posto do M5. O CSV e o JSON ficam em
`data/processed/recoleta_ponderada/inferencia_amostral/multicolinearidade_m5/`.

### Resíduos do M5 ponderado

```powershell
python scripts/31_residuos_m5.py
```

O script ajusta o M5 completo e calcula o resíduo como log do salário-hora
observado menos log do ajustado. A média e o desvio-padrão descritivo usam
V1028 na própria amostra de ajuste. Os decis nacionais são definidos pela
distribuição ponderada dos ajustes, com empates agrupados. A tabela exibe os
cortes em reais por hora a preços de janeiro de 2026.

Os efeitos fixos de UF centram o resíduo médio em cada macrorregião por
construção. Os desvios-padrão mostram como a dispersão varia entre regiões
e faixas de salário-hora estimado. O CSV e o JSON ficam em
`data/processed/recoleta_ponderada/inferencia_amostral/residuos_m5/`.
[NIST · diagnóstico da dispersão dos resíduos](https://itl.nist.gov/div898/handbook/pmd/section4/pmd442.htm).

### Influência de observações no M5 ponderado

```powershell
python scripts/32_influencia_m5.py
```

O script usa a atualização exata de mínimos quadrados ponderados para
retirar uma linha por vez. Os contrastes de escolaridade, formalidade e sexo
usam o perfil de referência da amostra completa. A mudança em pontos
percentuais é a estimativa da exclusão menos a estimativa completa. Exclusões
com alavancagem numericamente próxima de 1 ficam separadas na contagem.
[SAS · diagnósticos de influência em regressão linear](https://support.sas.com/documentation/onlinedoc/stat/132/reg.pdf).

O 1% com maior mudança absoluta é selecionado separadamente para cada
contraste, com empates preservados e mudanças nulas fora da seleção. A
concentração é calculada por contagem de registros. Os decis ponderados do
salário-hora estimado seguem o diagnóstico de resíduos. A saída contém apenas
resumos nacionais e contagens por decil ou macrorregião. O CSV e o JSON ficam em
`data/processed/recoleta_ponderada/inferencia_amostral/influencia_m5/`.

### Dispersão dos resíduos nos extremos do M5

```powershell
python scripts/33_dispersao_m5.py
```

Compara o desvio-padrão ponderado dos resíduos em log de D1 e D10 com o
desvio-padrão dos registros de D5 e D6 reunidos. As duas diferenças e as
variações relativas têm IC95% pontuais. A linearização por UPA dentro de
estrato usa a covariância conjunta, inclui o reajuste dos coeficientes do M5
e trata a atribuição às faixas como fixa. A variação relativa usa o método
delta na escala do log da razão. [Referência do método delta](https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svycontrast.html).

O CSV e o JSON ficam em
`data/processed/recoleta_ponderada/inferencia_amostral/dispersao_m5/`.

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

Usa todas as observações elegíveis do M5 histórico com V1028 para nove pares:
mulheres versus homens, e pessoas pretas, amarelas, pardas e indígenas versus
brancas, e quatro macroregiões versus Sudeste. Para cada par, estima a referência pooled com indicador de grupo,
conforme [Jann (2008)](https://repec.ethz.ch/ets/papers/jann_oaxaca.pdf).
O vetor de controles comuns mantém a família M5, retirando a variável definidora
do par e sua interação com escolaridade. O indicador entra somente no ajuste
pooled. A diferença é sempre grupo menos referência.

Nas comparações regionais, todas as indicadoras de UF são retiradas dos
controles, pois elas determinam a macroregião e impediriam identificar o
indicador do grupo. Permanecem cor/raça, sexo, escolaridade e os demais
controles do M5. As cinco comparações demográficas mantêm sua especificação.

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

### Comparação temporal do M5 ponderado em reais e em log

```powershell
python scripts/34_comparar_escala_m5.py
```

O M5 temporal usa os mesmos controles do M5 histórico, com tendência linear
de ano centrada em 2013 no lugar dos indicadores anuais. Os dois ajustes por
mínimos quadrados ponderados usam 2013–2023. O fator global de Duan é a média
ponderada de `exp(y_log - previsto_log)` em 2024. [Duan (1983)](https://www.jstor.org/stable/2288126).

As três previsões são avaliadas nos mesmos registros de 2025. MAE, RMSE e viés
(`previsto − observado`) usam pesos V1028 e reais por hora a preços de janeiro
de 2026. Os decis nacionais são definidos pelo salário-hora observado no teste.
O CSV e o JSON ficam em `data/processed/recoleta_ponderada/previsao_m5/`.

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

O relatório mostra cobertura inclusiva e larguras média, mediana e P90 na
calibração e no teste. Os recortes são região, sexo, escolaridade e decis do
salário-hora observado. Grupos com menos de 30 registros exibem a contagem.
O método segue [Angelopoulos e Bates](https://arxiv.org/abs/2107.07511).

Use `python pnad_pipeline/scripts/24_intervalos_previsao.py --reavaliar` para
atualizar o relatório existente com o modelo, a conversão e o raio registrados.
O script exige as mesmas amostras de 2024 e 2025 e confere as métricas já
publicadas antes de trocar o relatório. Os decis de cada período mantêm
salários empatados juntos e os diagnósticos e a cobertura usam os mesmos
limites no teste. O MdAPE é `mediana(abs(previsto - observado) / observado)`
em salários reais positivos, armazenado como proporção e exibido em porcentagem.

Execução `20261108T132604904108Z`: GradientBoosting_folhas31; Duan 1,2101498;
raio 1,0482523; calibração 90,002%; teste 90,736%; largura média R$47,5324/h.
A cobertura depende de hipóteses de permutabilidade não asseguradas por dados
temporais com rotação. Esta é avaliação retrospectiva, pois 2025 já foi visto
em extensões anteriores, sem garantia prospectiva, populacional ou por grupo.
Não confundir faixa individual com IC de médias/coeficientes.

## Comparação de intervalos com regressão quantílica conformal

Execute os comandos desta seção a partir da raiz do repositório. A comparação
requer a base histórica `pnad_pipeline/data/processed/pnad_analise_25_65.parquet`
e a referência em `pnad_pipeline/data/processed/previsao_intervalos/`. Para
gerar a referência, execute `python pnad_pipeline/scripts/24_intervalos_previsao.py`.

```powershell
python pnad_pipeline/scripts/25_comparar_intervalos.py
```

O script usa o pipeline e a calibração registrados em `previsao_intervalos`
como referência. Os dois Gradient Boostings da faixa adaptativa estimam P5 e
P95 do log do salário-hora na amostra de treino de 2013–2021. A amostra de
2024 fornece o ajuste conformal com o escore `max(P5 − y, y − P95)` depois de
ordenar os limites. O ajuste é o escore de ordem `ceil((n+1)*0.90)`. Se uma
contração inverter os limites, a faixa vira o ponto médio. O protocolo,
os modelos e o ajuste são gravados antes de ler as amostras de avaliação.
[Romano, Patterson e Candès](https://arxiv.org/abs/1905.03222).

Cada método usa os mesmos 50 mil registros de 2025, verificados pelo hash da
execução anterior, e os mesmos 50 mil de 2026T1, sorteados com semente 202601.
O teste reservado usa o ZIP oficial, o dicionário de posições e a matriz de
disponibilidade de variáveis. O fator `1/1,007` estende a convenção histórica
de deflação até a base de janeiro/2026. Nessa convenção, a renda do trimestre
é convertida a partir do índice do mês anterior ao fechamento do trimestre.
O script registra os hashes das fontes e guarda a base e os relatórios em
`data/processed/previsao_adaptativa/`. [Microdados oficiais do IBGE](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/2026/).

| Amostra | Método | Cobertura geral | Cobertura no 1º decil | Cobertura no 10º decil | Largura média |
| --- | --- | ---: | ---: | ---: | ---: |
| 2025 | Atual | 90,74% | 64,38% | 74,62% | R$ 47,53/h |
| 2025 | Adaptativa | 90,14% | 76,26% | 68,89% | R$ 39,38/h |
| 2026T1 | Atual | 90,86% | 66,51% | 72,89% | R$ 48,40/h |
| 2026T1 | Adaptativa | 90,48% | 77,71% | 68,05% | R$ 40,47/h |

Resultados não ponderados da execução `20261109T225853928638Z`. Os decis usam
o salário observado de cada período. A cobertura empírica depende do período
e da amostra. O teste de 2026T1 inclui um trimestre e a referência de 2025
reúne quatro trimestres. A API `GET /api/comparacao-intervalos` entrega os
agregados dos períodos disponíveis e a proveniência da comparação.

### Ajustes conformais separados por limite

```powershell
python pnad_pipeline/scripts/25_comparar_intervalos.py --por-cauda
```

A extensão reutiliza os pipelines P5 e P95 registrados pelo script 25 e
confere os hashes dos modelos e da amostra de 2024. Os ajustes usam os
escores `P5 − y` e `y − P95`, cada um com ordem `ceil((n+1)*0.95)`. A meta
nominal é de 5% de saídas por lado. Se a contração inverter os limites,
usa-se o ponto médio dos limites ajustados. A execução de 2024 teve 2.499
saídas por lado em 50 mil registros. [Referência do método](https://arxiv.org/html/1905.03222#Thmtheorem2).

Os ajustes foram registrados antes da leitura dos 50 mil registros oficiais
reservados de 2026T2. A semente é 202602. O rendimento foi convertido à base
de janeiro/2026 com o fator 0,972195835, conforme a convenção histórica e
o IPCA de fevereiro a maio. [Microdados do IBGE](https://ftp.ibge.gov.br/Trabalho_e_Rendimento/Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados/2026/).

| Método em 2026T2 | Cobertura geral | Cobertura no 1º decil | Cobertura no 10º decil | Largura média |
| --- | ---: | ---: | ---: | ---: |
| Faixa atual | 90,90% | 67,62% | 72,01% | R$ 48,80/h |
| CQR com ajuste único | 90,26% | 77,81% | 67,21% | R$ 40,70/h |
| CQR com ajustes separados | 90,41% | 76,52% | 69,14% | R$ 42,21/h |

O manifesto `manifesto_caudas.json` vincula a extensão à execução CQR de
referência. A API agrega os três períodos e disponibiliza os hashes de
calibração e protocolo da extensão `20261109T231600594795Z`.

O formulário usa a faixa atual como padrão. Em 2026T2 ela cobriu melhor o
decil superior. As versões CQR permitem comparar esse resultado com o ganho
em largura e a maior cobertura no decil inferior.

Para reproduzir a comparação em uma pasta isolada, passe o mesmo valor em
`--artefatos-dir` nas duas chamadas ao script 25. Execute primeiro a comparação
com ajuste único e depois a extensão com `--por-cauda`. Os relatórios são
gerados localmente a partir das fontes e da referência registradas.
