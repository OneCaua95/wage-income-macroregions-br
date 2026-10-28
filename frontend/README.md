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
- Tabelas com busca, ordenação, paginação e CSV com separador `;`, decimal
  brasileiro e metadados de origem; JSON exporta o recorte e a proveniência.
- Previsão com pipeline scikit-learn completo exportado, validação dos campos,
  variáveis derivadas consistentes e métricas próprias da nova execução.
- M5 ponderado de 2025T4 e da série histórica 2013–2025, geral ou por
  macroregião, com coeficientes, referências, IC95%, erros-padrão, p e ajuste BH.

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

A previsão retorna `exp(predição em log)`, uma aproximação da média geométrica
condicional, sem correção de retransformação. O formulário informa o ano do
perfil, até 2025; anos ausentes do treino são ignorados pelo OneHotEncoder,
como na avaliação. Não existem projeções futuras nem intervalos de previsão
validados. Escolaridade é VD3005: código 0 inclui menos de um ano e 16 inclui
16 anos ou mais. Os rótulos das faixas mostram anos de estudo, sem inferir
diplomas. Valores nominais não foram preservados na base analítica; todos os
rendimentos apresentados são reais, a preços de janeiro/2026.

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
