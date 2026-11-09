# Referência e decisões do dashboard

Alvo de construção: os HTMLs locais `dashboard_salario_hora.html`,
`inferencia_m5_autonomo.html` e `resultados_testes_reais.html`.
Consulta em 16/10/2026. O Refero foi consultado, mas retornou
`NO_SUBSCRIPTION`; as fontes locais e as orientações de acessibilidade do
Refero (`craft-details.md`) fundamentam a implementação.

| Decisão | Fonte | Aplicação |
| --- | --- | --- |
| Cinza #f4f6f8, painéis brancos, marinho #172337 | HTMLs existentes | Superfícies e barra de navegação |
| Azul #2f6df6, borda #dce2e9, raio 10px | HTMLs existentes | Seleção, séries e controles |
| Inter/system-ui e números tabulares | HTMLs existentes e pedido | Tabelas compactas e indicadores alinhados |
| Resumo → exploração → ranking → detalhe | https://leagueofitems.com/champions/901 | Indicadores no topo, gráficos, tabelas ordenáveis e detalhes por resultado |
| Comparação de duas seleções com denominador explícito | League of Items e pedido | Comparador de grupos ou trimestres; B − A e (B − A) / A |
| Filtros persistentes e origem junto ao resultado | Pedido e relatório de testes reais | Recortes compartilhados; resultados fixos explicam incompatibilidades |
| Foco visível, rótulos e estados de formulário | Refero craft-details.md | Navegação por teclado, mensagens de erro e carregamento |

A direção dominante é a identidade existente. Gráficos são SVG e dados reais;
não há necessidade de imagens decorativas. A série histórica não receberá
o selo de ponderação da execução oficial de 2025T4.

## Mapa por UF · referência lock 28/10/2026

| Decisão | Fonte | Aplicação |
| --- | --- | --- |
| Preservar cinza `#f4f6f8`, branco, marinho `#172337`, azul `#2f6df6`, borda `#dce2e9`, Inter/system-ui e raio 10px | Identidade existente deste dashboard | Mapa e detalhes permanecem no mesmo painel e na mesma escala tipográfica |
| Mostrar a mediana real e P90/P10 por UF com período e filtros ativos | Planejamento consolidado do usuário | Seletor curto de métrica, escala azul sequencial e detalhe persistente da UF selecionada |
| Usar limites estaduais simplificados, armazenados localmente | API oficial de Malhas Geográficas do IBGE v3; documentada para mapas web leves | SVG interativo local, sem serviço de mapas ou requisição geográfica em tempo de uso |
| Atualizar o filtro global ao selecionar uma UF; conservar o mapa completo | Interação persistente solicitada para o dashboard | A consulta do mapa aplica os filtros ativos exceto UF; botão e tabela também permitem selecionar estado |
| Tratar geografia como gráfico, sem depender apenas da cor | Refero indisponível (`NO_SUBSCRIPTION`); orientação local de acessibilidade | Estados focáveis por teclado, foco visível, painel de detalhes, tabela equivalente e estado neutro para n<30 |

**Reference lock:** identidade visual atual como direção dominante; malha do IBGE
como recurso geográfico real e local. A legenda azul é exclusiva para magnitude
quantitativa; o marinho fica para foco e seleção. O mapa é acompanhado por
valores, amostra e população estimada V1028. Não usar tiles externos, gradientes
decorativos ou cor como único canal de informação.

A orientação da projeção mantém o Norte acima do Sul. Siglas identificam os
estados; chamadas com linhas curtas organizam os rótulos de UFs costeiras pequenas
e do Distrito Federal sem cobrir a malha.

Os limites de cada UF usam contorno fino em cinza-azulado; hover permanece azul e
a seleção ativa permanece marinho para distinguir os três estados visuais.

## Oaxaca–Blinder · referência de construção 08/11/2026

Alvo: painel existente de diferenças brutas e ajustadas, preservando os tokens
e os gráficos de pontos com intervalos do dashboard. Refero sem ferramenta
disponível nesta sessão; orientação local `craft-details.md` consultada.

| Decisão | Fonte | Aplicação |
| --- | --- | --- |
| Painel na seção Associação e inferência, após as diferenças salariais | Planejamento consolidado, item 9 após perfil de idade; interface existente | Cinco comparações em seletor local e tabela equivalente |
| Três linhas alinhadas à mesma escala, com zero e ICs | Forest plot existente; Jann (2008) | Total, explicada e não explicada em log, sem empilhar percentuais não aditivos |
| Azul para total, verde para explicada, âmbar para não explicada | Cores e papéis já usados pelo dashboard | Rótulos e valores identificam cada parcela sem depender da cor |
| Seletor rotulado, foco visível e anúncio do recálculo visual | Refero `craft-details.md` | Navegação por teclado e atualização local sem refazer modelos |
| Identificar período fixo, ponderação e referência pooled | Jann (2008) e regras do dashboard | Aviso junto ao gráfico, metadados no CSV/JSON e método em detalhes |

A decomposição usa a referência pooled com indicador de grupo, conforme
[Jann (2008)](https://repec.ethz.ch/ets/papers/jann_oaxaca.pdf). A variância
inclui a variação das médias de covariáveis. A parcela não explicada recebe
o mesmo destaque das demais e não é rotulada como discriminação.

## Diagnóstico preditivo por grupo · 08/11/2026

Alvo: aba Previsão existente, com seus painéis brancos, títulos compactos,
azul para estimativas, linha de referência neutra e tabelas exportáveis.
Refero sem ferramenta disponível; reutilizada a orientação de formulários,
foco, atualização assíncrona e alternativas textuais de `craft-details.md`.

| Decisão | Fonte | Aplicação |
| --- | --- | --- |
| Diagnóstico após o formulário e antes do ranking histórico | Planejamento consolidado, item 10; hierarquia da aba existente | Distinguir a execução operacional do benchmark histórico |
| MAE, RMSE, viés e contagem, por região, sexo, escolaridade e decis | Pedido do usuário e planejamento | Seletores locais com uma tabela equivalente e CSV |
| Observado versus previsto com linha de igualdade | Planejamento; gráficos SVG existentes | Médias de até 100 faixas quantis do observado; eixos logarítmicos rotulados em R$/h para tornar legível a cauda salarial |
| Publicar agregados, sem pontos de microdados | Arquitetura existente do dashboard | Faixas com pelo menos 30 registros; empates salariais preservados |
| Informar sinal e unidade do viés junto ao gráfico | Definição estatística de erro previsto menos observado | Valores negativos indicam subestimação; positivos indicam superestimação |
| Período e amostra próprios ao lado das métricas | Manifesto operacional e pedido | Teste 2025, não ponderado; filtros descritivos não alteram a avaliação |

**Referência de construção:** identidade visual existente; o seletor muda a
leitura dos mesmos resultados fixos, com foco visível e atualização anunciada.
O gráfico observado/previsto apresenta médias de grupos, explicitamente;
não representa uma dispersão de observações individuais nem intervalos de previsão.

## Correção de Duan · referência de construção 08/11/2026

Alvo: comparação entre conversões dentro da aba Previsão, preservando seus
painéis, tabelas e seletores. A referência metodológica é
[Duan (1983)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017).
O fator global será a média de `exp(observado_log − previsto_log)` em 2023.

Protocolo fixado antes da avaliação de 2025: reproduzir a amostra original de
validação, separar 2023 para calibração e 2024 para comparar. Recomendar Duan
se o RMSE em reais cair e o viés absoluto não aumentar em 2024. MAE, log e
resultados por grupo serão divulgados, sem exigir melhoria em todos eles.
Persistir a decisão e seu hash antes de reconstruir o teste de 2025.

| Decisão | Fonte | Aplicação |
| --- | --- | --- |
| Quadro de comparação próximo da estimativa e antes dos erros por grupo | Hierarquia atual e pedido | Quatro linhas de métricas: original/Duan × validação/teste |
| Indicar a versão recomendada com critério e amostra | Protocolo acima | Fator e escolha definidos sem resultados de 2025 |
| Mostrar ambas as conversões por perfil | Duan (1983) e interpretação atual | A estimativa principal segue a escolha da validação; valor original permanece identificado |
| Seletor de conversão nos diagnósticos | Seletores existentes; Refero `craft-details.md` | Alternar os erros e o gráfico das mesmas 50 mil observações |
| Comparar erros por grupo também em 2024 | Pedido de avaliação por grupos | Tabela local por período e dimensão, com contagens e CSV |

O quadro explica o alvo: `exp(log previsto)` aproxima a média geométrica;
o fator global de Duan busca a média em reais sob hipóteses de transportabilidade
dos resíduos. O modelo base já usou 2023–2024 na seleção e o teste de 2025 já
foi inspecionado; esta extensão é retrospectiva, condicionada ao pipeline fixo.

## Intervalos de previsão · referência de construção 08/11/2026

Referências: os painéis de previsão existentes, seu formulário e a comparação
de Duan acima; metodologia split conformal de
[Angelopoulos e Bates](https://arxiv.org/abs/2107.07511). Manter os mesmos tokens,
tipografia, foco visível, tabelas equivalentes e controles locais. A nova faixa
aparece junto da estimativa; a cobertura medida vem logo após o formulário,
com barras de 0–100% e linha de meta em 90%. Largura em R$/h aparece junto
da cobertura para não confundir faixa ampla com previsão precisa.

Protocolo fixado: ajuste 2013–2021; escolha 2022; Duan em 2023T1–T2;
escolha da conversão 2023T3–T4; calibração 2024; relato retrospectivo 2025.
A execução anterior fica recolhida e identificada como histórica. O formulário,
os erros e a faixa usam exclusivamente a nova execução e seus próprios hashes.
Meta nominal não deve ser apresentada como garantia futura ou por grupo.
