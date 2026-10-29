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
