# Referências do relatório de testes reais

Alvo: interface existente do projeto, preservando a identidade dos dashboards
`inferencia_m5.html` e `dashboard_salario_hora.html`.

O Refero MCP foi consultado para estilos e telas, mas retornou assinatura
inativa. A pesquisa continuou com essas duas interfaces concretas e as
referências locais de tipografia, acessibilidade e QA do Refero Design.

| Decisão | Referência | Regra preservada |
|---|---|---|
| Barra azul-marinho e fundo cinza claro | dashboards existentes | marca e navegação |
| Branco, bordas discretas, números tabulares | inferencia_m5.html | leitura de resultados |
| Azul em controles; verde em aprovação; âmbar em atenção | tokens existentes | papéis semânticos das cores |
| Tabelas com filtros e cabeçalho fixo | ambos os dashboards | densidade e exploração |
| Tipografia neutra e foco visível | Refero typography / craft-details | ferramenta de trabalho acessível |
| Período e ponderação sempre visíveis | solicitação e proveniência real | distinguir as duas estimações |

O HTML é um arquivo autônomo, sem fontes, imagens, scripts ou dados remotos.
Dados reais de 2025T4, resultados históricos anteriores e testes com fixtures
sintéticas recebem rótulos distintos. Os resultados não são dados simulados.
