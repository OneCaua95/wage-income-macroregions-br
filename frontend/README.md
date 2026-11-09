# Dashboard interativo

Aplicação em português brasileiro, sem etapa de build, servida pela API Python
do projeto. Os HTMLs autônomos anteriores continuam disponíveis.

## Executar

Na raiz `E:\projetos\pseudo-painel-pnad`:

**Windows:** dê dois cliques em `abrir_dashboard.cmd`. O servidor permanece
ativo em segundo plano, sem depender de manter um terminal aberto. O
inicializador verifica a API antes de abrir o navegador e reutiliza uma
instância já ativa. Use `fechar_dashboard.cmd` para encerrar o servidor.

Alternativa pelo terminal:

```powershell
python -m pip install -r pnad_pipeline/requirements.txt
python scripts/abrir_dashboard.py
```

Abra http://127.0.0.1:8000. Também é possível instalar o pacote com os extras:
`python -m pip install -e "./pnad_pipeline[dashboard,dev]"`.
O servidor escuta apenas na interface local. O inicializador registra erros em
`pnad_pipeline/data/processed/dashboard_runtime/servidor_8000.log`. Para
encerrar pelo terminal: `python scripts/abrir_dashboard.py --stop`.

Para acompanhar logs no terminal, o comando original continua disponível:
`python pnad_pipeline/scripts/13_servir_dashboard.py`; neste modo, mantenha
o terminal aberto e encerre com Ctrl+C.

O HTML agora carrega seus estilos por caminhos relativos, inclusive quando
aberto diretamente. Sem o servidor, apresenta instruções de inicialização.
Pela interface HTTP, falhas de conexão mostram uma mensagem em português e
permitem tentar novamente.

## O que funciona

- Cinco seções com navegação e filtros persistentes durante a sessão.
- Recortes combinados por trimestre, região, UF, sexo, cor/raça, anos de estudo,
  faixa etária, formalidade e setor público/privado.
- Médias, medianas, quartis, dispersão, valores válidos/ausentes, população
  quando há pesos, frequências e taxa de formalidade com denominador explícito.
- Histograma com classe aberta para rendimentos altos, séries trimestrais,
  comparação de duas seleções e detalhes por grupo.
- Mapa interativo por UF na visão geral, com mediana real ou razão P90/P10,
  período e filtros selecionados, detalhes, tabela e exportação CSV. O filtro de
  UF selecionado destaca o estado sem remover as demais unidades do mapa.
- Tabelas com busca, ordenação, paginação e CSV com separador `;`, decimal
  brasileiro e metadados de origem; JSON exporta o recorte e a proveniência.
- Previsão com pipeline scikit-learn completo exportado, validação dos campos,
  variáveis derivadas consistentes e métricas próprias da nova execução.
- M5 ponderado de 2025T4 e da série histórica 2013–2025, geral ou por
  macroregião, com coeficientes, referências, IC95%, erros-padrão, p e ajuste BH.
- Perfil de idade do salário-hora previsto pelo M5, com faixa de confiança pelo
  desenho amostral e exportação CSV; a composição das demais variáveis fica fixa
  no perfil médio ponderado do recorte.
- Decomposição Oaxaca–Blinder por sexo e cor/raça na série histórica, com
  parcelas total, explicada e não explicada em log, IC95% e exportações CSV/JSON.
- Diagnóstico da previsão no teste de 2025: MAE, RMSE, viés e contagens por
  macroregião, sexo, escolaridade e decis de salário observado, em reais ou log.
  Gráfico observado/previsto usa médias de faixas quantis e tem tabela equivalente.

## Fontes e limites operacionais

**Série histórica ponderada:** `data/processed/recoleta_ponderada/pnad_analise_25_65.parquet`.
Contém os 52 trimestres de 2013 a 2025, 14.259.616 registros pessoa-trimestre,
com V1028, Estrato e UPA. A base inclui pessoas de 25 a 65 anos ocupadas ou
não; salário-hora e formalidade só são definidos para seus respectivos domínios
de ocupados. Médias, quantis, totais e proporções usam V1028. A base histórica
antiga continua disponível no repositório, mas o dashboard prioriza a nova
série ponderada quando ela existe.

**M5 histórico:** `data/processed/recoleta_ponderada/inferencia_amostral/`.
O modelo geral usa 9.064.317 observações elegíveis e há resultados para cada
macroregião. O recorte aceita o filtro de macroregião; outros filtros exigem
uma nova estimação. A variância usa linearização por UPA e estrato, com
aproximação com reposição, sem FPC ou recalibração. Uma UPA da Bahia aparece
em dois estratos ao longo da série; nesta execução suas versões `(UF, Estrato,
UPA)` foram separadas e a covariância entre elas não é estimada. O relatório
registra essa escolha e o ajuste de dois estratos com uma UPA.

A PNAD tem rotação de domicílios entre trimestres e iniciou a renovação gradual
da Amostra Mestra em 2025T3. Por isso, os erros-padrão do M5 empilhado são uma
aproximação ao desenho temporal, não uma reprodução integral das variâncias
oficiais do IBGE. Os p-valores descrevem associações condicionais e não
identificam causalidade. [Nota técnica do IBGE sobre a renovação amostral](https://biblioteca.ibge.gov.br/visualizacao/livros/liv102204.pdf).

**Retorno da educação:** `data/processed/recoleta_ponderada/inferencia_amostral/retorno_educacao/`.
Compara escolaridade categórica, linear e spline na mesma amostra e oferece
curvas por sexo, cor/raça e macrorregião. O gráfico alterna entre diferenças
acumuladas contra VD3005=0 e retorno associado a um ano adicional; ambos mostram
IC95% pontuais e p-valores BH. A análise é fixa na série 2013–2025 e não muda
com filtros descritivos. VD3005=16 é 16 anos ou mais; na tendência é aproximado
como 16. Os resultados se referem a pessoas ocupadas e não são causais. Os
limites temporais do desenho são os mesmos do M5 histórico.

**Seleção para a amostra salarial:** `data/processed/recoleta_ponderada/inferencia_amostral/selecao_ocupacional/`.
Separa a taxa de ocupação (68,37%) da taxa de salário-hora positivo (67,23%)
no domínio completo de pessoas de 25–65 anos com covariáveis válidas. A análise
repondera a amostra salarial pela probabilidade de inclusão estimada a partir
de características observadas. As estimativas da escolaridade sob IPW aumentam
entre 0,26 e 2,50 pontos percentuais nesta amostra; a diferença não foi testada
como contraste entre estimadores dependentes. O piso de propensão limita cada
peso inverso a 20. Sem uma restrição de exclusão defensável, isto é apenas uma
análise de sensibilidade: seleção não observada permanece sem identificação e
os resultados não são causais. O painel e os filtros são fixos à série
ponderada 2013–2025; filtros descritivos não reestimam esta análise.

**Heterogeneidade regional:** `data/processed/recoleta_ponderada/inferencia_amostral/heterogeneidade_regional/`.
Compara, na mesma amostra, um modelo pooled com retornos regionais de
escolaridade e formalidade a cinco regressões separadas. Exibe testes Wald-F
conjuntos com a variância por UPA/estrato, curvas e IC95% por macroregião e
comparação de ajuste ponderado dentro da amostra. A comparação de ajuste não é
validação fora da amostra; os coeficientes são associações condicionais e não
causais. O painel usa os dados históricos 2013–2025 e não reestima por filtros.

**Decomposição Oaxaca–Blinder:** `data/processed/recoleta_ponderada/inferencia_amostral/oaxaca/`.
Cinco pares usam a amostra elegível do M5: mulheres/homens e cada grupo de
cor/raça contra branca. O ajuste de referência é pooled em cada par, com
indicador de grupo. A variável que define o par e sua interação não entram
na parcela de características; os demais controles seguem a família M5.
Diferença total, explicada e não explicada somam em log. Somente o total
também é apresentado em percentual; percentuais das parcelas não seriam aditivos.
Os ICs pontuais incorporam a variação conjunta de médias e coeficientes por
UPA/estrato, com os mesmos limites temporais do M5. A parcela não explicada
depende da referência e dos controles e não mede automaticamente discriminação.
Formalidade, setor e ocupação podem mediar as diferenças, por isso a parcela
explicada também não é uma atribuição causal. O seletor interno escolhe o par;
período e filtros globais não reestimam esta análise fixa de 2013–2025.

**Execução oficial 2025T4:** `validacao_real/2025T4/pnad_analise_2025T4.parquet`.
Usa V1028 para descritivas populacionais. V1028, Estrato e UPA são preservados.
O M5 usa os resultados do pipeline com cadastro completo de UPAs anterior à
seleção do domínio, linearização por UPA/estrato e política explícita de UPA
única `ajustar`. Não se reestima o M5 ao selecionar filtros. UF, sexo e outros
recortes sem resultados inferenciais mostram indisponibilidade com explicação.

**Previsão:** execute `python pnad_pipeline/scripts/12_treinar_previsao.py`.
O artefato já foi gerado nesta entrega e fica no diretório local ignorado pelo
Git `previsao_perfis/`. Em outro checkout, execute o comando novamente.
Modelos e hiperparâmetros são escolhidos pelo RMSE log de 2023–2024; o teste
2025 é avaliado apenas para o vencedor. O treino usa 2013–2022. A amostragem
em lotes é uniforme, por prioridades aleatórias, com semente 42. A nova amostra
difere do benchmark histórico e as métricas não são misturadas. O manifesto
registra origem, amostras, parâmetros (no pipeline), versões e SHA-256.

A conversão original retorna `exp(predição em log)`, uma aproximação da média
geométrica condicional. Quando a correção de Duan é aprovada pelo protocolo
de validação abaixo, o valor principal usa o fator global para aproximar a
média em reais, e o formulário mostra também o valor original. O ano do
perfil vai até 2025; anos ausentes do treino são ignorados pelo OneHotEncoder,
como na avaliação. Não existem projeções futuras nem intervalos de previsão
validados. Escolaridade é VD3005: código 0 inclui menos de um ano e 16 inclui
16 anos ou mais. Os rótulos das faixas mostram anos de estudo, sem inferir
diplomas. Valores nominais não foram preservados na base analítica; todos os
rendimentos apresentados são reais, a preços de janeiro/2026.

**Diagnóstico preditivo:** a aba Previsão lê os agregados vinculados ao mesmo
modelo e à mesma amostra de teste de 2025 (50 mil registros na execução atual).
Para gerar a avaliação de um modelo já exportado:

```powershell
python pnad_pipeline/scripts/22_diagnosticos_previsao.py
```

Novos treinos pelo script 12 geram esses diagnósticos automaticamente. O script
22 reconstrói apenas o teste com a origem e a semente do manifesto e exige que
as seis métricas originais sejam reproduzidas. O modelo salvo mantém seu
identificador e hash. O diagnóstico recebe hash próprio e impressão da amostra;
a API rejeita arquivos incompatíveis ou alterados.

Viés = previsto − observado; negativo indica subestimação. Cada agrupamento
particiona a mesma amostra. Decis são definidos pelo salário observado, com
empates juntos e limites repetidos unidos. Grupos com menos de 30 registros
mostram somente a contagem. O gráfico observado/previsto contém médias de até
100 faixas, com ao menos 30 registros cada; os eixos são logarítmicos, rotulados
em R$/h. Não há pontos individuais. CSV exporta métricas e metadados, e o JSON
geral inclui o diagnóstico quando a aba Previsão está aberta. Período, fonte,
filtros descritivos e formulário de perfil não reestimam essa avaliação fixa.

**Correção de Duan:** execute `python pnad_pipeline/scripts/23_avaliar_duan.py`.
O script reproduz a validação original, usa as 24.416 observações de 2023 para
calcular `média(exp(resíduo log))` e compara as conversões nas 25.584 observações
de 2024. A recomendação exige RMSE em reais menor e viés absoluto não maior;
MAE e erros por grupo também são divulgados. A decisão e seu hash são gravados
antes da leitura da amostra de teste de 2025.

Nesta execução, o fator foi 1,21226216 e Duan foi recomendado. No teste de
2025, RMSE caiu de R$ 26,25/h para R$ 25,65/h e o viés foi de −R$ 4,63/h para
−R$ 1,32/h, enquanto MAE aumentou de R$ 9,52/h para R$ 9,98/h. O painel exibe
essa troca, a comparação por grupo em 2024/2025 e o seletor de conversão para
os diagnósticos. Mudar o seletor de diagnóstico não altera o perfil preenchido
nem a recomendação aplicada à estimativa principal. CSV e JSON identificam a
conversão, o fator e a decisão. Novos modelos exigem nova execução do script 23.

O fator é global e pode deixar viés por grupo; aproximar a média em reais exige
transportabilidade dos resíduos. A avaliação é retrospectiva, condicionada ao
pipeline já escolhido: 2023–2024 já participaram da seleção do modelo e 2025
já teve seus diagnósticos inspecionados. A correção não equivale a uma nova
avaliação independente, inferência causal ou intervalo de previsão.

O ranking preserva o histórico exploratório selecionado pelo teste. M5 do
script 04 e M5 hierárquico do script 06 têm filtros de elegibilidade distintos
e aparecem em protocolos separados; a equivalência das amostras entre
scripts não foi comprovada. A escolha de outra ordenação não muda a
recomendação histórica registrada em `resumo_modelos.json`.

**Ainda dependem de processamento adicional:** inferência para filtros além
das macroregiões e intervalos preditivos calibrados. A comparação de
especificações inferenciais já está disponível junto aos resultados do M5.
O benchmark de previsão permanece uma execução distinta e não ponderada; não
é substituído pelo M5 ponderado.

## Arquitetura e validação

Recortes com menos de 30 salários-hora válidos recebem aviso de amostra
insuficiente. A comparação exige ao menos 30 valores válidos em cada seleção;
este limiar é uma regra de apresentação, não uma garantia de precisão amostral.

`index.html` e `assets/dashboard.{css,js}` usam HTML/CSS/JS nativos. Os gráficos
são SVG, com títulos, foco e alternativas tabulares. A API em
`pnad_pipeline.dashboard` consulta Parquet no servidor, usa cache limitado a
quatro períodos e agrega séries em lotes. Nenhum microdado é entregue ao
navegador. A previsão carrega somente artefatos locais produzidos pelo script,
verificando SHA-256 e versão de scikit-learn. Reinicie o servidor após trocar
os arquivos processados para atualizar catálogo e caches.

```powershell
python -m pytest pnad_pipeline/tests -q
node --check frontend/assets/dashboard.js
```

As decisões visuais estão em [referencias_dashboard.md](referencias_dashboard.md).

## Relatórios anteriores

## Dashboard atual

Abra [dashboard_salario_hora.html](dashboard_salario_hora.html) para visualizar o ranking dos modelos, as KPIs de validação/teste e os diagnósticos econométricos do M5.

## Resultados estatísticos e testes reais

Abra [resultados_testes_reais.html](resultados_testes_reais.html). O arquivo é autônomo e reúne inferência ponderada em microdados oficiais de 2025T4, resultados históricos anteriores não ponderados, verificações sobre dados reais e o relatório dos testes automatizados. Permite selecionar macroregiões, filtrar coeficientes e exportar CSV/JSON.

Para regenerar o relatório, siga a seção de validação real no [README do pipeline](../pnad_pipeline/README.md). A referência visual e as decisões de apresentação estão em [referencias_relatorio_testes.md](referencias_relatorio_testes.md).
# Intervalos de previsão com meta de 90%

A aba **Previsão** usa a nova execução de `previsao_intervalos` quando disponível.
O formulário apresenta a estimativa e os dois limites do mesmo modelo e da mesma
conversão. O painel seguinte mostra cobertura e largura em 2024 (calibração) e
2025 (teste retrospectivo), por macrorregião, sexo, escolaridade e faixa de
salário-hora observado. A execução anterior de Duan aparece como histórico,
com identidade própria.

Geração local: `python pnad_pipeline/scripts/24_intervalos_previsao.py`.
Para atualizar um relatório já gerado, acrescente `--reavaliar`.
O modelo é ajustado até 2021 e selecionado em 2022. O fator de Duan é estimado em
2023T1–T2 e a conversão escolhida em 2023T3–T4. A faixa é calibrada em 2024,
com meta de 90%, antes de ler a amostra de teste de 2025.

Resultados da execução `20261108T132604904108Z`: 45.368 dos 50.000 salários do
teste dentro da faixa (90,736%); largura média R$ 47,53/h, mediana R$ 36,98/h.
Nordeste: 88,87%; escolaridade de 0–4 anos: 88,27%. A meta geral não assegura
cobertura em todos os grupos. O teste já foi inspecionado em trabalhos anteriores;
mudança temporal e rotação da PNAD impedem afirmar a garantia de permutabilidade.

Nos decis salariais do teste, a cobertura foi de 64,38% no primeiro e 74,62%
no último. Os decis usam o salário observado de cada período e mantêm os
empates juntos. O painel de erros inclui o MdAPE geral e por grupo. A mediana
do erro absoluto em relação ao salário observado foi de 38,07% no teste.

Exportações: JSON do perfil inclui limites, largura, modelo e hashes de decisão,
calibração e protocolo; CSV da cobertura contém geral e todos os grupos dos dois
períodos (44 linhas nesta execução), contagens, larguras, meta, raio e
identificação da execução. O CSV de erros traz o MdAPE como proporção.
Grupos com menos de 30 registros têm suas métricas suprimidas. Valores não
ponderados a preços de janeiro/2026. A faixa não é IC de média ou coeficientes.
