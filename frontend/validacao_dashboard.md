# Validação da aplicação · 16/10/2026

## Correção de inicialização

O acesso a `localhost:8000` estava indisponível porque o processo de teste
havia sido encerrado. A aplicação agora tem inicializadores Windows:
`abrir_dashboard.cmd` e `fechar_dashboard.cmd`. O servidor é iniciado em
segundo plano, a API é verificada antes de abrir o navegador e uma segunda
abertura reutiliza o mesmo PID. O comando de encerramento confere a identidade
do projeto antes de encerrar um processo.

Na revisão, `http://localhost:8000` respondeu normalmente. Foram conferidos
indicadores, filtro por sexo e execução da previsão no navegador, sem erros
de aplicação no console. O servidor permaneceu ativo após o inicializador
terminar. A suíte atual contém 66 testes aprovados, incluindo proteção contra
encerrar outro projeto e reutilização da instância ativa.

O CSS e o JavaScript usam caminhos relativos. O script é carregado sem a
restrição de módulos de arquivos `file://`; abrir o HTML diretamente mostra
uma orientação de inicialização, em vez de ficar indefinidamente carregando.

## Código e contratos estatísticos

- `python -m pytest pnad_pipeline/tests -q`: **66 testes passaram**.
- Depois dos ajustes finais de apresentação, os 8 testes específicos do
  dashboard passaram novamente.
- `node --check frontend/assets/dashboard.js`: sem erros de sintaxe.
- Importação/compilação dos novos módulos e execução real da API: aprovadas.
- Testes cobrem ponderação desigual, quantis, domínios distintos, formalidade
  sem classificação, filtros, frequências, séries, base sem pesos, ausência
  de dados, amostra insuficiente, denominador A da comparação, inferência
  sem reestimação, artefato ausente e overflow de idade².

## Conferência com saídas reais

- Média de salário-hora 2025T4 e médias das cinco macroregiões coincidem,
  após arredondamento a duas casas, com as tabelas ponderadas do pipeline.
- 274.793 registros de 25 a 65 anos na fonte oficial; 184.611 salários-hora
  válidos. A amostra elegível do M5 é distinta (184.594), conforme filtros
  adicionais documentados na estimação. Os denominadores não são confundidos.
- O histórico atualizado tem 52 trimestres e usa V1028 nas descritivas; as
  séries agregam em lotes e preservam os filtros selecionados.
- A combinação região Norte / UF SP retorna estado sem observações.
- A previsão pela API coincide exatamente com a chamada direta do pipeline
  salvo para o mesmo perfil; diferença observada: zero.
- A média de idade² no scaler do artefato corrigido é 1.860,841273,
  consistente com idade em float64, sem overflow de int8.

## Artefato operacional

Execução `20261016T153847212121Z`, selecionada na validação:
`GradientBoosting_folhas31`. Amostras: 150 mil no treino, 50 mil na validação
e 50 mil no teste. A seleção coincide com o menor RMSE log das quatro
configurações avaliadas na validação. Apenas o vencedor foi avaliado no teste.

| Métrica | Validação | Teste |
| --- | ---: | ---: |
| RMSE log | 0,620626 | 0,618691 |
| MAE log | 0,454875 | 0,452852 |
| R² log | 0,455927 | 0,445957 |
| RMSE R$/hora | 25,014752 | 26,253690 |
| MAE R$/hora | 9,039961 | 9,524007 |

Esta execução é não ponderada, usa amostra diferente e fica separada do
benchmark histórico. Não foram sobrescritos os resultados anteriores.

## Navegador e aparência

Interface examinada no navegador local, mantendo a identidade dos HTMLs:
marinho, cinza-claro, painéis brancos, realces azuis e tabelas compactas.

- Navegação pelas cinco seções e persistência do filtro Sexo ao navegar.
- Recorte inferencial não estimado mostra explicação; limpar os filtros
  recupera os coeficientes publicados.
- Busca `Idade_centrada` retorna somente os dois termos de idade, ambos
  com interpretação específica, sem associação percentual automática.
- Comparação Homem/Mulher na execução ponderada: diferença aproximada
  de −R$ 2,77/h e −11,73%, em relação à média dos homens (A).
- CSV baixado pelo navegador e inspecionado: separador `;`, decimal com
  vírgula, UTF-8 e proveniência do recorte.
- Previsão por perfil executada, com origem e aviso de ano ausente no treino.
- Protocolo M5 hierárquico apresenta apenas sua própria avaliação.
- Visão móvel testada em 390 × 844, sem transbordamento horizontal da página;
  filtros recolhidos inicialmente. Gráficos densos e tabelas têm rolagem local.
- Foco visível, rótulos, navegação nativa por teclado e diálogo com fechamento
  via botão/Escape. O atalho para conteúdo preserva a seção atual.
- Console do navegador sem erros de aplicação na revisão.

## Atualização da série ponderada · 28/10/2026

A coleta retomável `14_reextrair_historico_ponderado.py` concluiu 2013–2025:
52 trimestres, 26.641.303 linhas e V1028, Estrato e UPA presentes. A base
analítica de 25–65 anos tem 14.259.616 registros pessoa-trimestre. Foram
geradas as séries ponderadas de salário-hora, desigualdade, formalidade e
educação, cada uma com 52 linhas.

O M5 ponderado foi estimado para Brasil e cinco macroregiões. O modelo geral
tem 9.064.317 observações elegíveis, 46.034 conglomerados de desenho e 827
estratos. A política para estratos com uma UPA foi `ajustar`. Uma UPA da Bahia
(290190891) mudou de estrato entre períodos; a execução separou as duas versões
por `(UF, Estrato, UPA)` e registra que não estima a covariância entre elas.

A variância é uma aproximação com reposição, sem FPC ou recalibração. A rotação
da PNAD entre trimestres e a renovação progressiva da Amostra Mestra, iniciada
em 2025T3, limitam sua interpretação temporal. Os resultados são associações
condicionais, não efeitos causais. A especificação e os limites estão em
`pnad_pipeline/data/processed/recoleta_ponderada/inferencia_amostral/relatorio_inferencia_m5.md`.

## Retorno associado à escolaridade · 28/10/2026

Foram estimadas seis especificações na mesma amostra elegível do M5: escolaridade
categórica, tendência linear, spline linear por segmentos com nós em 4, 8 e 11
anos, e a spline com perfis por sexo, cor/raça e macrorregião. A amostra tem
9.064.317 observações, 46.034 versões de UPA e 827 estratos em 52 trimestres.
VD3005=16 representa 16 anos ou mais; nas tendências, recebe o valor 16.

O resultado apresenta diferenças acumuladas contra VD3005=0 e diferenças entre
códigos adjacentes, com IC95% pontuais e ajuste Benjamini–Hochberg. Os testes
Wald-F encontraram associação conjunta da escolaridade nas três formas
(categorias F=2.610,7; linear F=30.227,8; spline F=9.885,0), desvio da
linearidade na forma categórica (F=604,7), componentes não lineares na spline
(F=2.478,6) e heterogeneidade nos perfis por sexo (F=405,7), cor/raça (F=50,6)
e macrorregião (F=217,0); todos com p<0,001 após o ajuste aplicável. A análise
se restringe a pessoas ocupadas e não identifica efeitos causais. A aproximação
de variância e os limites temporais são os mesmos documentados para o M5.

O CSV, o resumo JSON e o relatório estão em
`pnad_pipeline/data/processed/recoleta_ponderada/inferencia_amostral/retorno_educacao/`.
O dashboard disponibiliza filtros da forma funcional, da dimensão de
heterogeneidade e da medida acumulada ou marginal; filtros descritivos não
reestimam os modelos.

## Seleção ocupacional · 28/10/2026

A análise separa pessoas ocupadas das pessoas que também têm rendimento e horas
positivos para entrar na regressão salarial. No domínio de covariáveis válidas
de 25–65 anos, há 14.256.300 observações pessoa-trimestre; a taxa ponderada de
ocupação é 68,37%, e 67,23% têm salário-hora válido. A amostra salarial
coincide com as 9.064.317 observações do modelo de escolaridade.

A regressão logística ponderada estima a probabilidade de inclusão na amostra
com escolaridade, sexo e interação sexo–escolaridade, cor/raça, idade, UF, ano
e trimestre. A mediana ponderada da propensão é 72,1%, o 5º percentil é 26,7%
e 1.638 previsões (0,01% do domínio-alvo) ficam abaixo de 5%. Na amostra salarial,
81 de 9.064.317 pesos inversos (0,0009%) foram efetivamente limitados a 20.
Para a sensibilidade salarial, os intervalos de 95% incorporam a estimação da
propensão por uma linearização conjunta dos scores, preservando UPA e estrato.

Com IPW, as 16 associações categóricas de escolaridade aumentaram entre 0,26 e
2,50 pontos percentuais; para 16 anos ou mais, passaram de 151,23% para
153,73%. A diferença não foi testada como contraste entre dois estimadores
dependentes. Mulheres têm taxa de ocupação ponderada de 57,17%, ante 80,33%
para homens; por escolaridade, varia de 40,23% sem instrução ou menos de um ano
a 83,30% para 16 anos ou mais.

O IPW é uma sensibilidade sob seleção explicada pelas covariáveis observadas,
não uma correção da seleção não observada. As colunas disponíveis não oferecem
restrição de exclusão defensável para Heckman; os resultados não são causais.
As limitações temporais da variância empilhada são as mesmas do M5 histórico.
CSV, resumo JSON e relatório ficam em
`pnad_pipeline/data/processed/recoleta_ponderada/inferencia_amostral/selecao_ocupacional/`.
O dashboard inclui taxas por escolaridade, sexo, cor/raça e macrorregião, além
da comparação salarial base/IPW. Conferência visual em 1440 px e 390 px:
seletor trocou entre escolaridade (17 grupos), sexo (2) e macrorregião (5),
sem erros de aplicação ou recursos ausentes. A largura da página móvel ficou
em 390 px; o gráfico e as tabelas mantêm rolagem local quando necessário.

## Heterogeneidade regional · 28/10/2026

A análise compara escolaridade e formalidade entre as cinco macrorregiões,
com 9.064.317 observações ponderadas por V1028. A especificação pooled
identifica 153 termos; os modelos regionais são estimados separadamente. A
variância usa a aproximação por UPA dentro de estrato documentada acima, sem
FPC ou recalibração, com o cadastro completo do desenho preservado antes da
seleção salarial.

Os testes Wald-F conjuntos apontam heterogeneidade de escolaridade
(F=38,09; 64 restrições) e formalidade (F=1.385,48; 4 restrições), ambos com
p<1e-16 após o ajuste BH aplicável. O valor numérico bruto subflui para zero
em ponto flutuante; a apresentação usa o limite `< 1e-16`, nunca p=0.
Associações acumuladas de 12 anos de estudo contra o código 0 vão de 32,9%
no Sudeste a 72,1% no Nordeste no modelo pooled. A associação de formalidade
varia de 11,8% no Sul a 39,4% no Nordeste. Essas estimativas são condicionais
entre ocupados, não causais.

A tabela compara o modelo comum, o pooled com interações e os modelos
regionais por RMSE log e R² ponderados dentro da amostra. O ajuste melhora
numericamente nos modelos mais flexíveis, mas isso não é previsão fora da
amostra nem teste para escolher uma especificação. Os intervalos por nível
são pontuais e não simultâneos; o empilhamento de 52 trimestres não estima
integralmente a covariância de domicílios revisitados.

API conferida: `/api/saude` respondeu `dashboard-1.6`; o endpoint regional
retornou estado `ok`, 320 linhas de escolaridade, 10 de formalidade e 5 de
comparação de ajuste. No navegador, a seção exibiu os três testes e tabelas;
os seletores alternaram corretamente entre modelo pooled/separado e retornos
acumulados/adjacentes. A visualização voltou ao padrão pooled/acumulado.
`py_compile`, `node --check` e `git diff --check` passaram. A suíte de testes
não foi executada nesta extensão.

## Pseudo-painel por coortes · 28/10/2026

A análise percorreu a série histórica ponderada completa em lotes e agregou
9.064.317 observações elegíveis em 928 células coorte-trimestre. O critério
principal (n ≥ 100, Kish ≥ 50 e ao menos oito trimestres por coorte) reteve
925 células, 22 coortes de sexo × nascimento e os 52 trimestres de 2013T1 a
2025T4. Cerca de 8,1% do peso das células qualificadas usou ano de nascimento
aproximado por ano menos idade.

Os modelos têm efeitos fixos de coorte e trimestre. O segundo adiciona médias
de escolaridade e proporções de formalidade, setor público, cor/raça e região;
o terceiro inclui tendências anuais diferenciais por coorte. No limiar
principal, os R² ponderados dentro da amostra foram 0,853, 0,941 e 0,983,
respectivamente. Isso descreve ajuste na própria amostra, não validação
preditiva. Nos cortes (n, Kish) de (200, 100) e (500, 250), o painel reteve
922 e 903 células; o último limiar reduziu o conjunto para 21 coortes.

O painel em `/api/pseudopainel` agrega duas faixas de cinco anos na visualização
anual, com filtro por sexo; os modelos e a tabela de tendências preservam as
coortes de cinco anos. A rota foi consultada diretamente e retornou estado
`ok`, 128 pontos anuais, nove comparações de modelo e 262 coeficientes. Após
reiniciar o servidor, `/api/saude` respondeu `dashboard-1.7`. No navegador, com
a série histórica selecionada, a seção Associação e inferência exibiu o
gráfico, as nove comparações e as 21 tendências diferenciais; o seletor de
sexo apareceu. A aba não registrou erros de console. Na janela móvel de
390 px, a largura do documento permaneceu em 375 px e tabelas/gráfico mantêm
rolagem horizontal interna.

Os resultados são médias de grupos, não trajetórias individuais. WLS usa o
efetivo de Kish como peso de precisão aproximado; ICs e p-valores usam
clusterização em duas vias por coorte e trimestre e são exploratórios. O
erro amostral nas médias, os poucos grupos de coorte e a rotação da PNAD
impedem interpretar essa variância como a estimativa oficial por UPA/estrato.
`py_compile`, `node --check` e `git diff --check` foram conferidos; a suíte
de testes não foi executada nesta extensão.

## Limites

O dashboard usa a nova série ponderada para os resumos históricos, mas mantém
o benchmark preditivo como execução separada e não ponderada. A inferência
do M5 está disponível para Brasil e macroregiões; outros recortes exigem
reestimação. O retorno da escolaridade inclui comparação das formas funcional
e heterogeneidade por sexo, cor/raça e macrorregião. A seleção inclui taxas e
sensibilidade IPW sob covariáveis observadas; fatores não observados não são
identificados. O pseudo-painel descreve médias de coortes e não identifica
trajetórias individuais. Ainda não há intervalos preditivos calibrados.
O servidor é local e precisa ser iniciado
conforme o README. Os microdados e artefatos continuam ignorados pelo Git.

## Desigualdade salarial e regressões quantílicas · 28/10/2026

A nova rota `/api/desigualdade` respondeu `ok` para a fonte histórica em 2025T4,
com 13 grupos de distribuição, 143 pontos de Lorenz e 80 coeficientes quantílicos.
A série cobre 52 trimestres, de 2013T1 a 2025T4, com 9.064.317 observações
pessoa-trimestre elegíveis. Os cinco modelos quantílicos convergiram; os erros-
padrão e intervalos foram conferidos como positivos e finitos.

No navegador local, confirmei que a aba Desigualdade salarial informa quando a
fonte 2025T4 é selecionada e carrega os percentis, o Gini, a curva de Lorenz e a
tabela ao mudar para a série histórica ponderada. A aba Associação e inferência
exibe as cinco estimativas da escolaridade selecionada, com erros-padrão,
intervalos e p-valores. A rota `/api/saude` respondeu `dashboard-1.8`.

A distribuição usa o arquivo completo; as regressões iniciam em amostra
calibrada e aplicam duas correções de scores na série completa. Os intervalos
quantílicos são aproximações de linearização por UPA/estrato com densidade
condicional escalar, não uma variância oficial específica para regressão
quantílica. As estimativas são associações, não efeitos causais. `py_compile`,
`node --check` e `git diff --check` passaram. A suíte de testes não foi executada
nesta extensão.

## Diferenças salariais brutas e ajustadas · 28/10/2026

A rota `/api/diferencas-grupos` respondeu `ok` para a série histórica ponderada,
com cinco contrastes, 9.064.317 observações pessoa-trimestre e ICs de 95% por
linearização dentro de UPA e estrato. A tela Associação e inferência mostra um
gráfico de pontos e intervalos e uma tabela filtrável/exportável. Confirmei no
navegador os cinco grupos e os valores retornados pela API.

Para mulheres versus homens, a diferença bruta foi −8,3% e a padronizada pelo
M5 foi −17,8%. Para pessoas pretas versus brancas, foi −33,2% bruta e −10,5%
ajustada; para pardas versus brancas, −34,4% e −9,1%. As cinco estimativas
ajustadas coincidem com os contrastes dos coeficientes do M5 na escolaridade
média ponderada da amostra. A população empilhada representa pessoas-trimestre,
não indivíduos únicos; os filtros do dashboard não reestimam estes contrastes.
São comparações associativas, não causais.

`/api/saude` respondeu `dashboard-1.9`; com a fonte histórica selecionada, a
rota respondeu `ok` e a interface apresentou gráfico e tabela completos.
`py_compile`, `node --check` e `git diff --check` passaram. A suíte de testes
não foi executada nesta extensão.

## Mapa interativo por UF · 28/10/2026

O endpoint `/api/mapa-uf` respondeu `ok` para a execução ponderada 2025T4 e a
série histórica ponderada. As duas fontes retornaram 27 UFs, mediana real,
P10, P90, razão P90/P10, contagens válidas e totais V1028. Em 2025T4, a mediana
foi R$ 13,84/h no Brasil e R$ 15,85/h em São Paulo; a razão P90/P10 foi 6,25
no Brasil e 6,39 em São Paulo. A malha local do IBGE foi carregada com HTTP 200.

Na interface, confirmei a escala de cor, a referência nacional, as estatísticas
de UF e a tabela equivalente. Selecionar São Paulo atualizou o filtro global,
destacou o estado e manteve as demais UFs no mapa; limpar a seleção restaurou
a visão nacional. A API ignora apenas o filtro de UF ao montar o mapa e respeita
os demais filtros. Estados com menos de 30 salários válidos ficam sem estimativa.
Os quantis usam V1028 quando disponível; a população apresentada é o total
ponderado apenas entre salários válidos. A malha é simplificada e servida como
arquivo local, sem transmissão de microdados ao navegador.

`/api/saude` respondeu `dashboard-2.1`. `py_compile`, `node --check`, validação
do JSON da malha (27 códigos únicos) e `git diff --check` passaram. A suíte de
testes não foi executada nesta extensão.

## Correção de orientação e rótulos do mapa · 28/10/2026

A inspeção visual encontrou a projeção vertical invertida: Roraima aparecia ao
sul do Rio Grande do Sul. A geometria foi regenerada da malha oficial do IBGE
com Norte acima do Sul. Também foram incluídas siglas das UFs e chamadas para
separar Rio Grande do Norte, Paraíba, Pernambuco, Alagoas, Sergipe e Distrito
Federal, cujos rótulos se sobrepunham na escala do mapa.

No navegador, confirmei a orientação correta e o novo posicionamento das
chamadas do Nordeste; o mapa continua com legenda, estatísticas por UF, seleção
e tabela. A rota `/api/saude` respondeu `dashboard-2.2`; `py_compile`,
`node --check`, conferência dos 27 códigos da malha e `git diff --check`
passaram. A suíte de testes não foi executada nesta extensão.

## Contorno dos estados no mapa · 28/10/2026

O traço branco dos polígonos se confundia com o fundo do painel, principalmente
no perímetro externo do país. Cada UF agora tem um contorno fino cinza-azulado;
hover continua azul e a UF selecionada continua marinho. A atualização visual
usa cache-busting no CSS e não altera valores ou geometria. A inspeção no
navegador confirmou contornos visíveis nas 27 UFs e linhas-guia legíveis no
Nordeste. `git diff --check` passou.

## Perfil de idade do salário-hora · 28/10/2026

O M5 agora exporta uma curva para cada fonte e macroregião, com uma linha por
idade de 25 a 65 anos. A predição usa o perfil médio ponderado e varia apenas a
idade; os ICs pontuais de 95% usam a covariância completa dos coeficientes,
linearizada por UPA dentro de estrato. `exp(log previsto)` é a média geométrica
condicional e não recebe correção de retransformação nem interpretação causal.

A tela Associação e inferência mostra a curva e a faixa para a fonte e
macroregião selecionadas; a exportação CSV inclui valores e limites. As 12
tabelas geradas (duas fontes, Brasil e cinco macroregiões) têm 41 idades cada,
com estimativa dentro do IC em todas as linhas. `py_compile`, `node --check`,
`git diff --check` e validação das rotas do dashboard passaram.
