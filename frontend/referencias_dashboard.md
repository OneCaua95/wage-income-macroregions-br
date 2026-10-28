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
