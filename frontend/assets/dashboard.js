(async () => {
const $ = s => document.querySelector(s);
const esc = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v,n=2) => v === null || v === undefined || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString('pt-BR',{minimumFractionDigits:n,maximumFractionDigits:n});
const money = v => v == null ? '—' : Number(v).toLocaleString('pt-BR',{style:'currency',currency:'BRL'});
const pct = v => v == null ? '—' : fmt(v*100,1)+'%';
const pfmt = v => v == null ? '—' : v < .00001 ? '< 0,00001' : fmt(v,5);
const date = v => v ? new Date(v).toLocaleString('pt-BR',{timeZone:'America/Sao_Paulo'}) : 'Não registrada';
const headings = {
  visao:['Visão geral','Uma leitura dos dados, dos perfis e das diferenças no mercado de trabalho.'],
  descritiva:['Estatística descritiva','Explore a distribuição da renda e compare grupos da população analisada.'],
  previsao:['Previsão por perfil','Informe um perfil ocupacional e consulte o modelo selecionado na validação temporal.'],
  associacao:['Associação e inferência','Investigue associações condicionais com o salário-hora e sua incerteza estatística.'],
  metodologia:['Modelos e metodologia','Compare avaliações compatíveis e conheça os domínios, os métodos e a origem dos resultados.']
};
let saved = {};
try { saved = JSON.parse(sessionStorage.getItem('pnad-dashboard') || '{}'); } catch {}
const S = {source:saved.source || 'real',period:saved.period || '2025T4',filters:saved.filters || {},group:saved.group || 'regiao',educationSpec:saved.educationSpec || 'categorias',educationDimension:saved.educationDimension || 'geral',educationContrast:saved.educationContrast || 'acumulado',selectionDimension:saved.selectionDimension || 'escolaridade',regionalApproach:saved.regionalApproach || 'Pooled com interações',regionalContrast:saved.regionalContrast || 'acumulado',view:'visao'};
let catalog, current, modelData, inferData, educationData, selectionData, regionalData, controller, revision=0, educationRevision=0, selectionRevision=0, comparisonData, predictionData, seriesData;
const tables = new Map();
const selection = () => new URLSearchParams({fonte:S.source,periodo:S.period,...S.filters});
const sourceData = () => catalog.sources.find(x=>x.id===S.source);
const save = () => {try{sessionStorage.setItem('pnad-dashboard',JSON.stringify(S));}catch{}};
const badge = (text,kind='blue') => `<span class="badge ${kind}">${esc(text)}</span>`;
const loading = text => `<div class="loading"><span class="spinner" aria-hidden="true"></span>${esc(text)}</div>`;
const empty = (title,text) => `<div class="empty"><strong>${esc(title)}</strong><p>${esc(text)}</p></div>`;
const panel = (title,subtitle,body,extra='') => `<section class="panel"><div class="panel-heading"><div><h2>${title}</h2><p>${subtitle}</p></div>${extra}</div>${body}</section>`;
const groupSelect = (id='group-select',value=S.group) => `<label><span class="sr-only">Agrupar por</span><select id="${id}">${Object.entries(catalog.labels).map(([v,t])=>`<option value="${v}" ${v===value?'selected':''}>${esc(t)}</option>`).join('')}</select></label>`;
const stat = (summary,key='Salario_Hora') => summary.estatisticas.find(r=>r.variavel===key) || {};
const varLabel = k => ({Salario_Hora:'Salário-hora (R$/hora)',Rendimento_hab_Trab_princ:'Rendimento habitual (R$/mês)',Horas_hab_trabalhadas:'Horas habituais (horas/semana)'}[k] || k);

async function api(path,options={}) {
  let response;
  try{response=await fetch(path,options);}catch(error){if(error.name==='AbortError')throw error;throw new Error('O servidor local não está respondendo. Abra abrir_dashboard.cmd na pasta do projeto e tente novamente.');}
  let body;
  try { body = await response.json(); } catch { throw new Error('Resposta inválida da API. Abra o site pelo servidor Python, em http://127.0.0.1:8000.'); }
  if (!response.ok) throw new Error(body.message || 'A consulta não pôde ser concluída.');
  return body;
}

function download(name,type,body) {
  const url = URL.createObjectURL(new Blob([body],{type}));
  const a = document.createElement('a'); a.href=url; a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function csv(name,rows,provenance={}) {
  const enriched = rows.map(r=>({...r,fonte:provenance.conjunto || provenance.fonte || '',periodo:r.periodo ?? provenance.periodo ?? S.period,
    ponderacao:provenance.ponderacao || '',versao:provenance.versao || '',filtros:JSON.stringify(provenance.filtros || S.filters)}));
  const fields = [...new Set(enriched.flatMap(Object.keys))];
  const cell = v => '"'+String(typeof v==='object' && v!==null ? JSON.stringify(v) : typeof v==='number' ? String(v).replace('.',',') : v ?? '').replaceAll('"','""')+'"';
  download(name+'.csv','text/csv;charset=utf-8','\ufeff'+[fields.map(cell).join(';'),...enriched.map(r=>fields.map(k=>cell(r[k])).join(';'))].join('\r\n'));
}
function detail(title,entries,note='') {
  $('#detail-title').textContent=title;
  $('#detail-body').innerHTML=`<dl class="detail-list">${entries.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>${note?`<p class="explanation">${esc(note)}</p>`:''}`;
  $('#detail-dialog').showModal();
}
$('#close-dialog').onclick=()=>$('#detail-dialog').close();
$('.skip').onclick=e=>{e.preventDefault();$('#conteudo').focus();$('#conteudo').scrollIntoView();};
$('#detail-dialog').addEventListener('click',e=>{if(e.target===e.currentTarget){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});

function buildFilters() {
  $('#source').value=S.source;
  for (const option of $('#source').options) option.disabled=!catalog.sources.find(x=>x.id===option.value)?.available;
  const ds=sourceData();
  if(!ds.periods.includes(S.period))S.period=ds.periods.at(-1);
  $('#period').innerHTML=[...ds.periods].reverse().map(p=>`<option ${p===S.period?'selected':''}>${p}</option>`).join('');
  $('#filter-fields').innerHTML=Object.entries(catalog.labels).map(([key,label])=>`<label>${esc(label)}<select data-filter="${key}" aria-label="Filtrar por ${esc(label)}"><option value="">${key==='sexo'?'Todos':key==='regiao'?'Brasil':'Todas as categorias'}</option>${ds.options[key].map(o=>`<option value="${esc(o.value)}" ${S.filters[key]===o.value?'selected':''}>${esc(o.label)}</option>`).join('')}</select></label>`).join('');
  document.querySelectorAll('[data-filter]').forEach(el=>el.onchange=()=>{if(el.value)S.filters[el.dataset.filter]=el.value;else delete S.filters[el.dataset.filter];refresh();});
  activeFilters(); save();
}
function activeFilters() {
  const ds=sourceData();
  $('#filter-count').textContent=Object.keys(S.filters).length;
  $('#active-filters').innerHTML=Object.entries(S.filters).map(([k,v])=>`<span class="chip">${esc(catalog.labels[k])}: ${esc(ds.options[k]?.find(o=>o.value===v)?.label || v)}<button data-remove="${k}" aria-label="Remover filtro ${esc(catalog.labels[k])}">×</button></span>`).join('');
  document.querySelectorAll('[data-remove]').forEach(b=>b.onclick=()=>{delete S.filters[b.dataset.remove];buildFilters();refresh();});
}
$('#source').onchange=e=>{S.source=e.target.value;buildFilters();refresh();};
$('#period').onchange=e=>{S.period=e.target.value;refresh();};
$('#clear-filters').onclick=()=>{S.filters={};buildFilters();refresh();};
$('#toggle-filters').onclick=()=>{const button=$('#toggle-filters');const open=button.getAttribute('aria-expanded')!=='true';button.setAttribute('aria-expanded',open);$('#filter-fields').hidden=!open;};
$('#export-all').onclick=()=>download(`pnad_${S.source}_${S.period}_${S.view}.json`,'application/json',JSON.stringify({exportado_em:new Date().toISOString(),selecao:{...S},descritiva:current,serie:seriesData,modelos:modelData,inferencia:inferData,retorno_educacao:educationData,selecao_ocupacional:selectionData,heterogeneidade_regional:regionalData,comparacao:comparisonData,previsao:predictionData},null,2));

function provenance(p) {
  $('#context-line').innerHTML=`${badge(p.ponderacao,p.ponderacao==='V1028'?'green':'amber')}<strong>${esc(S.period)}</strong><span>·</span><span>${esc(p.dominio)}</span><span>·</span><span>Preços de janeiro/2026</span><span>·</span><span>Processamento: ${esc(date(p.processado_em))}</span>`;
  const entries=[['Conjunto',p.conjunto],['Domínio',p.dominio],['Ponderação / desenho',p.ponderacao+' / '+p.desenho],['Preços',p.precos],['Unidade de contagem',p.unidade],['Processamento',date(p.processado_em)+' · Brasília'],['Versão da consulta',p.versao],['Arquivo de entrada',p.arquivo?.arquivo],['SHA-256 da fonte',p.sha256 || 'Não registrado no histórico']];
  $('#provenance').innerHTML=`<details><summary>Proveniência e processamento deste recorte ↗</summary><dl class="provenance-grid">${entries.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}${p.url_fonte?`<dt>Origem oficial</dt><dd><a href="${esc(p.url_fonte)}" target="_blank" rel="noreferrer">Microdados trimestrais / IBGE</a></dd>`:''}</dl></details>`;
}
function kpis(summary) {
  const wage=stat(summary);
  const ponderado=current?.provenance?.ponderacao==='V1028';
  return `<div class="kpis"><article class="kpi primary"><span class="kpi-label">Salário-hora médio</span><strong class="kpi-value">${money(wage.media)} <span class="kpi-unit">/h</span></strong><small>${fmt(wage.n_validos,0)} ocupados com renda e horas positivas</small></article>
    <article class="kpi"><span class="kpi-label">Mediana do salário-hora</span><strong class="kpi-value">${money(wage.mediana)} <span class="kpi-unit">/h</span></strong><small>Percentil 50 ${ponderado?'ponderado por V1028':'da amostra histórica'}</small></article>
    <article class="kpi"><span class="kpi-label">Taxa de formalidade</span><strong class="kpi-value">${pct(summary.taxa_formalidade)}</strong><small>${fmt(summary.n_formalidade,0)} ocupados com classificação válida</small></article>
    <article class="kpi"><span class="kpi-label">${summary.populacao==null?'Observações no domínio':'População estimada no domínio'}</span><strong class="kpi-value">${fmt(summary.populacao ?? summary.n,0)}</strong><small>${summary.populacao==null?'Total populacional indisponível · sem pesos':fmt(summary.n,0)+' observações na amostra'}</small></article></div>`;
}
function bars(rows) {
  const values=rows.filter(r=>r.valor!==null && stat(r).media!=null).sort((a,b)=>stat(b).media-stat(a).media);
  const max=Math.max(...values.map(r=>stat(r).media),1);
  if(!values.length)return empty('Sem grupos disponíveis','Nenhum grupo tem salário-hora válido neste recorte.');
  const ponderado=current?.provenance?.ponderacao==='V1028';
  return values.slice(0,12).map(r=>`<div class="bar-row"><span>${esc(r.grupo)}</span><div class="bar-track"><div class="bar-fill" style="width:${stat(r).media/max*100}%"></div></div><strong>${money(stat(r).media)}</strong></div>`).join('')+`<p class="note">Média do salário-hora · ${ponderado?'ponderada por V1028':'não ponderada'}${values.length>12?' · 12 maiores médias; todos os grupos na tabela':''}.</p>`;
}
function histogram(rows) {
  if(!rows.some(r=>r.n))return empty('Distribuição indisponível','Não há salário-hora válido neste recorte.');
  const max=Math.max(...rows.map(r=>r.proporcao || 0),.01), w=620,h=240, left=45,base=194, bw=52;
  return `<div class="chart"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Distribuição do salário-hora por faixa; proporção entre observações válidas"><line class="grid-line" x1="${left}" y1="${base}" x2="610" y2="${base}"/>${rows.map((r,i)=>{const bh=(r.proporcao || 0)/max*150,x=left+i*62;return `<rect tabindex="0" x="${x}" y="${base-bh}" width="${bw}" height="${bh}" fill="${i<4?'#2f6df6':'#779deb'}" rx="2"><title>R$ ${r.faixa}/hora: ${pct(r.proporcao)}; ${fmt(r.n,0)} registros</title></rect><text x="${x+bw/2}" y="${base+19}" text-anchor="middle">${esc(r.faixa)}</text><text x="${x+bw/2}" y="${base-bh-8}" text-anchor="middle">${pct(r.proporcao)}</text>`;}).join('')}<text x="330" y="238" text-anchor="middle">Salário-hora real (R$/hora)</text></svg></div><div class="chart-caption">Classes [limite inferior, superior); última classe ≥ R$ 100/h. Todas as observações válidas incluídas.</div>`;
}
async function lineChart(token) {
  const target=$('#series-chart'); if(!target)return;
  target.innerHTML=loading('Calculando evolução para os filtros ativos…');
  try {
    const result=await api('/api/serie?'+selection(),{signal:controller.signal});
    if(token!==revision||!$('#series-chart'))return;
    seriesData=result;
    const rows=result.rows.filter(r=>r.media!=null), w=700,h=240,left=50,right=674,top=28,base=190;
    if(!rows.length){target.innerHTML=empty('Série sem observações','O recorte não tem salário-hora válido nos períodos desta fonte.');return;}
    const low=Math.floor(Math.min(...rows.map(r=>r.media))*.85),high=Math.max(...rows.map(r=>r.media))*1.12;
    const x=i=>rows.length===1?(left+right)/2:left+i*(right-left)/(rows.length-1),y=v=>base-(v-low)/(high-low || 1)*(base-top);
    const points=rows.map((r,i)=>`${x(i)},${y(r.media)}`).join(' ');
    const ticks=[0,1,2,3].map(i=>low+(high-low)*i/3);
    const indices=[...new Set([0,Math.floor((rows.length-1)/3),Math.floor((rows.length-1)*2/3),rows.length-1])];
    const ponderacao=result.provenance.ponderacao==='V1028'?'ponderada por V1028':'não ponderada';
    const origem=S.source==='real'?'Execução oficial 2025T4':'Série histórica 2013–2025';
    const legenda=rows.length===1?`${origem}. Média ${ponderacao}; há um trimestre com dados válidos neste recorte.`:`${origem}. Média ${ponderacao}. Selecione um ponto para consultar o trimestre.`;
    target.innerHTML=`<div class="chart"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Evolução da média de salário-hora por trimestre">${ticks.map(t=>`<line class="grid-line" x1="${left}" y1="${y(t)}" x2="${right}" y2="${y(t)}"/><text x="40" y="${y(t)+4}" text-anchor="end">${fmt(t,0)}</text>`).join('')}${rows.length>1?`<polygon class="area" points="${left},${base} ${points} ${right},${base}"/><polyline class="line" points="${points}"/>`:''}${rows.map((r,i)=>`<circle class="point" tabindex="0" role="button" aria-label="${r.periodo}, ${money(r.media)} por hora" data-point="${i}" cx="${x(i)}" cy="${y(r.media)}" r="${rows.length===1?5:3}"><title>${r.periodo}: ${money(r.media)}/h · ${fmt(r.n_validos,0)} registros válidos</title></circle>`).join('')}${indices.map(i=>`<text x="${x(i)}" y="216" text-anchor="middle">${rows[i].periodo}</text>`).join('')}<text x="12" y="14">R$/hora</text></svg></div><div class="chart-caption" id="series-caption">${esc(legenda)}</div>`;
    const show=i=>{$('#series-caption').textContent=`${rows[i].periodo} · média ${money(rows[i].media)}/h · ${fmt(rows[i].n_validos,0)} observações válidas · ${result.provenance.ponderacao}`;};
    target.querySelectorAll('[data-point]').forEach(p=>{p.onclick=()=>show(+p.dataset.point);p.onfocus=()=>show(+p.dataset.point);p.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(+p.dataset.point);}};});
    $('#export-series').onclick=()=>csv('pnad_serie_'+S.source,result.rows,result.provenance);
  } catch(e) { if(e.name!=='AbortError'&&token===revision)target.innerHTML=empty('Série indisponível',e.message); }
}

function makeTable(id,rows,columns,provenance=current?.provenance,onDetail=null) {
  const host=$('#'+id); if(!host)return;
  const state={rows,columns,query:'',sort:null,dir:1,page:0,provenance,onDetail};tables.set(id,state);
  host.innerHTML=`<div class="table-tools"><label class="sr-only" for="search-${id}">Buscar nesta tabela</label><input id="search-${id}" type="search" placeholder="Buscar nesta tabela…"><span id="count-${id}"></span><button class="button secondary" id="export-${id}">↓ CSV</button></div><div class="table-wrap"><table><caption class="sr-only">${esc(host.dataset.title || 'Resultados do recorte')}</caption><thead><tr>${columns.map(c=>`<th scope="col" data-key="${c.key}" aria-sort="none"><button type="button">${esc(c.label)} <span aria-hidden="true">↕</span></button></th>`).join('')}</tr></thead><tbody></tbody></table></div><div class="table-pagination"><span class="count">CSV inclui todas as linhas filtradas.</span><button class="button secondary" id="prev-${id}" aria-label="Página anterior">←</button><span id="page-${id}"></span><button class="button secondary" id="next-${id}" aria-label="Próxima página">→</button></div>`;
  const visible=()=>{
    let r=state.rows.filter(r=>!state.query||columns.map(c=>String(c.value?c.value(r):r[c.key])).join(' ').toLocaleLowerCase('pt-BR').includes(state.query));
    if(state.sort){const c=columns.find(c=>c.key===state.sort);r=[...r].sort((a,b)=>{a=c.value?c.value(a):a[c.key];b=c.value?c.value(b):b[c.key];return (a==null?1:b==null?-1:typeof a==='number'&&typeof b==='number'?a-b:String(a).localeCompare(String(b),'pt-BR'))*state.dir;});}
    return r;
  };
  const render=()=>{
    const rows=visible(),pages=Math.max(1,Math.ceil(rows.length/30));state.page=Math.min(state.page,pages-1);
    host.querySelector('tbody').innerHTML=rows.slice(state.page*30,(state.page+1)*30).map((r,i)=>`<tr>${columns.map((c,j)=>{const value=c.value?c.value(r):r[c.key];return `<td>${j===0&&onDetail?`<button class="row-detail" data-index="${i}">${esc(c.format?c.format(value):value)}</button>`:esc(c.format?c.format(value):value)}${c.sub?`<small>${esc(c.sub(r))}</small>`:''}</td>`;}).join('')}</tr>`).join('')||`<tr><td colspan="${columns.length}">Nenhuma linha corresponde à busca.</td></tr>`;
    $(`#count-${id}`).textContent=`${rows.length} de ${state.rows.length} linhas`;
    $(`#page-${id}`).textContent=`${state.page+1} / ${pages}`;
    $(`#prev-${id}`).disabled=state.page===0;$(`#next-${id}`).disabled=state.page>=pages-1;
    host.querySelectorAll('[data-index]').forEach(b=>b.onclick=()=>onDetail(rows[state.page*30+Number(b.dataset.index)]));
    host.querySelectorAll('th').forEach(th=>th.setAttribute('aria-sort',th.dataset.key===state.sort?(state.dir===1?'ascending':'descending'):'none'));
  };
  $(`#search-${id}`).oninput=e=>{state.query=e.target.value.toLocaleLowerCase('pt-BR');state.page=0;render();};
  host.querySelectorAll('th button').forEach(b=>b.onclick=()=>{const key=b.parentElement.dataset.key;state.dir=state.sort===key?-state.dir:1;state.sort=key;state.page=0;render();});
  $(`#prev-${id}`).onclick=()=>{state.page--;render();};$(`#next-${id}`).onclick=()=>{state.page++;render();};
  $(`#export-${id}`).onclick=()=>csv('pnad_'+id,visible().map(r=>({...Object.fromEntries(columns.map(c=>[c.key,c.value?c.value(r):r[c.key]])),...(r.protocolo?{protocolo:r.protocolo}:{})})),provenance);
  render();
}

const groupColumns=[{key:'grupo',label:'Grupo'},{key:'n',label:'N amostral',format:v=>fmt(v,0)},
  {key:'n_validos',label:'N salário válido',value:r=>stat(r).n_validos,format:v=>fmt(v,0)},
  {key:'media',label:'Média R$/h',value:r=>stat(r).media,format:v=>fmt(v)},
  {key:'mediana',label:'Mediana R$/h',value:r=>stat(r).mediana,format:v=>fmt(v)},
  {key:'taxa_formalidade',label:'Formalidade',format:pct},
  {key:'populacao',label:'População estimada',format:v=>fmt(v,0)}];
function showGroup(r){detail(r.grupo,[['Período',S.period],['Observações no domínio',fmt(r.n,0)],['Salário-hora válido',fmt(stat(r).n_validos,0)],['Média',money(stat(r).media)+'/h'],['Desvio-padrão descritivo',money(stat(r).desvio_padrao)+'/h'],['Formalidade',pct(r.taxa_formalidade)],['Denominador da taxa',fmt(r.denominador_formalidade,0)+' · '+r.denominador_formalidade_unidade],['Ponderação',current.provenance.ponderacao]],'Médias e taxas usam domínios diferentes; os denominadores são específicos de cada indicador.');}

function renderOverview() {
  const historical=modelData.historical,lead=historical.metricas_melhor_modelo || {},m=inferData.model;
  $('#content').innerHTML=kpis(current.summary)+`<div class="grid">${panel('Evolução do salário-hora','Média trimestral · mesmo recorte e mesma fonte','<div id="series-chart"></div>','<button class="button secondary" id="export-series">↓ CSV</button>')}${panel('Comparação entre grupos','Médias do período selecionado',bars(current.groups),groupSelect())}</div>`+
    `<div class="grid equal">${panel('Modelo preditivo recomendado','Ranking histórico exploratório não ponderado',`<div class="method-card">${badge('Benchmark histórico','amber')}<h3>${esc(historical.melhor_modelo_por_rmse_log_teste || 'Resultado indisponível')}</h3><div class="model-metrics"><span><strong>${fmt(lead.rmse_log,4)}</strong>RMSE log · teste</span><span><strong>${money(lead.mae_salario_hora)}</strong>MAE R$/h · teste</span><span><strong>${fmt(lead.r2_log,3)}</strong>R² log · teste</span></div><p class="note">Treino 2013–2022 · validação 2023–2024 · teste 2025. Seleção histórica pelo teste; a previsão operacional usa uma nova execução selecionada na validação.</p><a href="#previsao">Consultar previsão por perfil →</a></div>`)}${panel('Modelo de associação disponível','M5 ponderado na fonte selecionada',`<div class="method-card">${badge('M5 · V1028 + desenho','green')}<h3>Regressão ponderada M5</h3><p class="note">Linearização por UPA dentro de estrato. Sem comparação válida de especificações por erro de validação.</p>${m?`<div class="model-metrics"><span><strong>${fmt(m.n_observacoes_elegiveis,0)}</strong>Observações elegíveis</span><span><strong>${fmt(m.n_upas_desenho,0)}</strong>UPAs</span><span><strong>${fmt(m.n_estratos_desenho,0)}</strong>Estratos</span></div>`:`<p class="explanation">${esc(inferData.message)}</p>`}<a href="#associacao">Explorar associações e intervalos →</a></div>`)}</div>`+
    panel('Resultados por grupo','Selecione um grupo na tabela para consultar os denominadores.','<div id="group-table"></div>');
  makeTable('group-table',current.groups,groupColumns,current.provenance,showGroup);
  $('#group-select').onchange=e=>{S.group=e.target.value;refresh();};
}

function renderDescriptive() {
  $('#content').innerHTML=kpis(current.summary)+`<div class="grid equal">${panel('Distribuição do salário-hora','Proporção dentro do domínio com salário-hora válido',histogram(current.histogram))}${panel('Comparação entre grupos','Média real no período selecionado',bars(current.groups),groupSelect())}</div>`+
    panel('Estatísticas do recorte','N válidos e ausentes têm como denominador todas as observações do recorte.','<div id="stats-table"></div><p class="note">Desvio-padrão da distribuição (divisor N ou soma de V1028); não é erro-padrão amostral. Rendimento e horas: ocupados com valores positivos.</p>')+
    panel('Detalhamento por grupo','Busca, ordenação e exportação dos resultados agregados.','<div id="group-table"></div>')+
    `<div class="grid equal">${panel('Evolução trimestral','O período selecionado não restringe esta série; os demais filtros são aplicados.','<div id="series-chart"></div>','<button class="button secondary" id="export-series">↓ CSV</button>')}${panel('Comparar duas seleções','Escolha grupos ou períodos da mesma fonte.','<div id="comparator"></div>')}</div>`+
    panel('Frequências e proporções','Denominador: V1028 na fonte ponderada; registros na fonte sem pesos. Inclui categorias sem informação.','<div id="freq-table"></div>');
  makeTable('stats-table',current.summary.estatisticas,[{key:'variavel',label:'Indicador',format:varLabel},{key:'n_validos',label:'Válidos',format:v=>fmt(v,0)},{key:'n_ausentes',label:'Ausentes / fora do domínio',format:v=>fmt(v,0)},{key:'media',label:'Média',format:fmt},{key:'mediana',label:'Mediana',format:fmt},{key:'desvio_padrao',label:'Desvio-padrão',format:fmt},{key:'q1',label:'Q1 · 25%',format:fmt},{key:'q3',label:'Q3 · 75%',format:fmt}]);
  makeTable('group-table',current.groups,groupColumns,current.provenance,showGroup);
  makeTable('freq-table',current.frequencies,[{key:'variavel',label:'Variável'},{key:'categoria',label:'Categoria'},{key:'n',label:'N amostral',format:v=>fmt(v,0)},{key:'total',label:current.provenance.ponderacao==='V1028'?'Total ponderado':'Registros',format:v=>fmt(v,0)},{key:'proporcao',label:'Proporção',format:pct},{key:'denominador',label:'Denominador',format:v=>fmt(v,0)}]);
  $('#group-select').onchange=e=>{S.group=e.target.value;refresh();};
  comparator();
}

function comparator() {
  const host=$('#comparator');let mode='grupos',group='sexo';
  host.innerHTML=`<div class="compare-controls"><label>Comparar<select id="compare-mode"><option value="grupos">Dois grupos</option><option value="periodos">Dois períodos</option></select></label><label id="compare-group-label">Variável<select id="compare-group">${Object.entries(catalog.labels).map(([k,v])=>`<option value="${k}" ${k===group?'selected':''}>${esc(v)}</option>`).join('')}</select></label></div><div class="compare-controls"><label>A · referência<select id="compare-a"></select></label><label>B · comparação<select id="compare-b"></select></label></div><button class="button" id="compare-run">Calcular diferença</button><div id="compare-result" style="margin-top:16px"></div>`;
  function opts(){const choices=mode==='grupos'?sourceData().options[group]:sourceData().periods.map(v=>({value:v,label:v}));$('#compare-group-label').hidden=mode==='periodos';for(const id of ['a','b'])$('#compare-'+id).innerHTML=choices.map(o=>`<option value="${esc(o.value)}">${esc(o.label)}</option>`).join('');if(choices.length>1)$('#compare-b').selectedIndex=1;$('#compare-run').disabled=choices.length<2;$('#compare-result').innerHTML=choices.length<2?empty('Comparação indisponível','Esta fonte contém apenas um período. Colete outros trimestres com pesos para comparar estimativas populacionais.'):'';}
  $('#compare-mode').onchange=e=>{mode=e.target.value;opts();};$('#compare-group').onchange=e=>{group=e.target.value;opts();};opts();
  $('#compare-run').onclick=async()=>{const btn=$('#compare-run');btn.disabled=true;$('#compare-result').innerHTML=loading('Calculando comparação…');try{const q=selection();q.set('modo',mode);q.set('grupo',group);q.set('a',$('#compare-a').value);q.set('b',$('#compare-b').value);const r=await api('/api/comparacao?'+q,{signal:controller.signal});comparisonData=r;const out=$('#compare-result');if(!out)return;if(r.status!=='ok'){out.innerHTML=empty('Amostra insuficiente','A comparação exige pelo menos 30 salários-hora válidos em cada seleção. Nenhuma diferença foi apresentada.');return;}out.innerHTML=`<div class="comparison"><div><span>A · referência</span><strong>${money(stat(r.a).media)}</strong><p>${esc(r.a.selecao)} · N válido ${fmt(r.a.n_salario,0)}</p></div><div><span>B · comparação</span><strong>${money(stat(r.b).media)}</strong><p>${esc(r.b.selecao)} · N válido ${fmt(r.b.n_salario,0)}</p></div><div class="delta"><span>B − A</span><strong>${money(r.diferenca_absoluta)}</strong><p>${fmt(r.diferenca_percentual)}% em relação a A</p></div></div><p class="note" style="margin-top:12px">${esc(r.denominador)}. O filtro da variável comparada é substituído por A e B; os demais filtros permanecem.</p><button class="text-button" id="export-compare">↓ Exportar comparação CSV</button>`;$('#export-compare').onclick=()=>csv('pnad_comparacao',[{a:r.a.selecao,b:r.b.selecao,media_a:stat(r.a).media,media_b:stat(r.b).media,n_a:r.a.n_salario,n_b:r.b.n_salario,diferenca_absoluta:r.diferenca_absoluta,diferenca_percentual:r.diferenca_percentual,denominador:r.denominador}],r.provenance);}catch(e){if(e.name!=='AbortError'&&$('#compare-result'))$('#compare-result').innerHTML=empty('Não foi possível comparar',e.message);}finally{btn.disabled=false;}};
}

function rankingTable(id,rows,provenance=modelData.provenance) {
  const split=$('#ranking-split')?.value || 'validacao',scale=$('#ranking-scale')?.value || 'log';
  const protocol=$('#ranking-protocol')?.value || 'benchmark_03';
  provenance={...provenance,conjunto:'Benchmark histórico · '+protocol,periodo:split==='validacao'?'2023–2024':'2025',filtros:{}};
  const compatible=rows.filter(r=>r.tipo===split && r.protocolo===protocol);
  const field=scale==='log'?'log':'salario_hora';
  compatible.sort((a,b)=>a['rmse_'+field]-b['rmse_'+field]);
  makeTable(id,compatible.map((r,i)=>({...r,posicao:i+1})),[{key:'modelo',label:'Modelo',sub:r=>r.posicao===1?'Menor RMSE nesta visualização':''},{key:'tipo',label:'Conjunto',format:v=>v==='validacao'?'Validação':'Teste'},{key:'rmse',label:scale==='log'?'RMSE log':'RMSE R$/h',value:r=>r['rmse_'+field],format:v=>fmt(v,4)},{key:'mae',label:scale==='log'?'MAE log':'MAE R$/h',value:r=>r['mae_'+field],format:v=>fmt(v,4)},{key:'r2',label:scale==='log'?'R² log':'R² reais',value:r=>r['r2_'+field],format:v=>fmt(v,4)}],provenance,r=>detail(r.modelo,[['Conjunto',r.tipo],['RMSE log',fmt(r.rmse_log,5)],['MAE log',fmt(r.mae_log,5)],['R² log',fmt(r.r2_log,5)],['RMSE em reais',money(r.rmse_salario_hora)],['MAE em reais',money(r.mae_salario_hora)],['R² em reais',fmt(r.r2_salario_hora,5)],['Ponderação','Não ponderado'],['Amostras',JSON.stringify(modelData.historical.amostras)]],'Ranking histórico exploratório. Compare apenas linhas com a mesma escala, conjunto de avaliação e protocolo de amostragem. Modelos regionais, M5 in-sample e a nova execução preditiva não integram esta classificação.'));
}
function rankingPanel() {
  return panel('Ranking histórico dos modelos','Treino 2013–2022 · validação 2023–2024 · teste 2025 · amostras históricas fixas',`<div class="inline-options"><label>Protocolo / amostra de avaliação<select id="ranking-protocol"><option value="benchmark_03">Benchmark original · script 03</option><option value="m5_04">M5 ajustado · script 04</option><option value="m5_06">M5 hierárquico · script 06</option></select></label><label>Conjunto de avaliação<select id="ranking-split"><option value="validacao">Validação · 2023–2024</option><option value="teste">Teste · 2025</option></select></label><label>Escala da métrica<select id="ranking-scale"><option value="log">Log(salário-hora)</option><option value="reais">Reais por hora</option></select></label>${badge('Não ponderado','amber')}</div><div id="ranking-table"></div><p class="note">A recomendação histórica foi feita pelo menor RMSE log no teste. M5 usa filtros de elegibilidade antes da amostragem e fica em protocolo separado; a identidade das amostras entre scripts não foi comprovada. QuantReg treina em até 50 mil e MixedLM em até 40 mil registros; ambos usam a avaliação do script 03. Os filtros populacionais não recalculam estas métricas.</p>`);
}
function wireRanking(){rankingTable('ranking-table',modelData.rows);for(const id of ['ranking-protocol','ranking-split','ranking-scale'])$('#'+id).onchange=()=>rankingTable('ranking-table',modelData.rows);}

function renderPrediction() {
  const m=modelData.prediction;
  if(!m?.schema){$('#content').innerHTML=empty('Previsão ainda indisponível','Execute python pnad_pipeline/scripts/12_treinar_previsao.py para exportar o modelo e seu pré-processamento.')+rankingPanel();wireRanking();return;}
  $('#content').innerHTML=`<div class="notice">Previsão para um perfil ocupado, de 25 a 65 anos, com renda e horas positivas. Os filtros descritivos permanecem salvos; a previsão usa os campos deste formulário e o modelo global.</div><div class="grid equal">${panel('Defina o perfil','As variáveis derivadas são calculadas de forma consistente com o pipeline.',`<form id="prediction-form"><div class="form-grid">${m.schema.map(f=>`<label>${esc(f.label)}${f.type==='number'?`<input name="${f.name}" type="number" inputmode="numeric" min="${f.min}" max="${f.max}" step="1" value="${f.name==='Idade'?40:12}" required>`:`<select name="${f.name}" required>${f.options.map(o=>`<option value="${esc(o.value)}" ${(f.name==='Ano'&&o.value==='2025')||(f.name==='UF'&&o.value==='35')||(f.name==='Trimestre'&&o.value==='4')?'selected':''}>${esc(o.label)}</option>`).join('')}</select>`}</label>`).join('')}</div><div class="derived" id="derived-features"></div><div class="form-footer"><button class="button" type="submit" id="predict-button">Estimar salário-hora →</button><span class="note">Valores reais · janeiro/2026</span></div></form>`)}${panel('Estimativa do modelo','Nova execução selecionada exclusivamente na validação temporal.',`<div id="prediction-result"><div class="prediction-output"><span class="kpi-label">SALÁRIO-HORA PREVISTO</span><div class="big-value">— <span class="kpi-unit">R$/h</span></div><p>Preencha o perfil e calcule a estimativa.</p></div></div><h3>${esc(m.modelo)}</h3><p class="note">${esc(m.criterio)}.</p><div class="mini-metrics" style="margin-top:16px"><div><strong>${money(m.metricas_teste.rmse_salario_hora)}</strong><span>RMSE R$/h · teste</span></div><div><strong>${money(m.metricas_teste.mae_salario_hora)}</strong><span>MAE R$/h · teste</span></div><div><strong>${fmt(m.metricas_teste.r2_salario_hora,3)}</strong><span>R² reais · teste</span></div></div><p class="note">Treino ${esc(m.divisao_temporal.treino)} · validação ${esc(m.divisao_temporal.validacao)} · teste ${esc(m.divisao_temporal.teste)}. Não ponderado.</p><p class="explanation">${esc(m.interpretacao)} ${esc(m.intervalo)}</p><details><summary class="note">Artefato, amostra e processamento</summary><dl class="detail-list"><dt>Execução</dt><dd>${esc(m.run_id)}</dd><dt>Treinado em</dt><dd>${date(m.treinado_em)}</dd><dt>Amostras</dt><dd>${esc(JSON.stringify(m.amostras))}</dd><dt>scikit-learn</dt><dd>${esc(m.sklearn)}</dd><dt>SHA-256</dt><dd>${esc(m.sha256)}</dd></dl></details>`)}</div>`+panel('Seleção da nova execução','Mesmo alvo, escala log, amostra e período; teste avaliado somente para o vencedor.','<div id="validation-table"></div>')+rankingPanel();
  makeTable('validation-table',m.validacao,[{key:'modelo',label:'Modelo / configuração'},{key:'rmse_log',label:'RMSE log · validação',format:v=>fmt(v,4)},{key:'mae_log',label:'MAE log · validação',format:v=>fmt(v,4)},{key:'r2_log',label:'R² log · validação',format:v=>fmt(v,4)}],{conjunto:'Nova execução '+m.run_id,periodo:'2023–2024',ponderacao:'Não ponderado',versao:m.version,filtros:{}});
  const form=$('#prediction-form');
  const derived=()=>{const p=Object.fromEntries(new FormData(form)),c=+p.Categoria_emprego;$('#derived-features').innerHTML=badge([1,3,5,7].includes(c)||[8,9].includes(c)&&p.Contribuicao_previdencia==='1'?'Formal':'Informal')+badge([5,6,7].includes(c)?'Setor público':'Setor privado')+badge('Idade² = '+fmt(Number(p.Idade)**2,0));};
  form.onchange=derived;derived();
  form.onsubmit=async e=>{e.preventDefault();const btn=$('#predict-button');btn.disabled=true;btn.textContent='Calculando…';try{const r=await api('/api/previsao',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({perfil:Object.fromEntries(new FormData(form))}),signal:controller.signal});predictionData=r;const target=$('#prediction-result');if(!target)return;target.innerHTML=`<div class="prediction-output"><span class="kpi-label">SALÁRIO-HORA PREVISTO · PREÇOS JAN/2026</span><div class="big-value">${money(r.salario_hora)} <span class="kpi-unit">/h</span></div><p>exp da previsão em log · média geométrica condicional aproximada</p></div>${r.aviso_ano?`<p class="explanation">${esc(r.aviso_ano)}</p>`:''}<button class="text-button" id="export-prediction">↓ Exportar previsão e perfil</button>`;$('#export-prediction').onclick=()=>download('pnad_previsao_perfil.json','application/json',JSON.stringify({...r,manifesto:m},null,2));}catch(e){if(e.name!=='AbortError'&&$('#prediction-result'))$('#prediction-result').innerHTML=empty('Previsão indisponível',e.message);}finally{btn.disabled=false;btn.textContent='Estimar salário-hora →';}};
  wireRanking();
}

function forest(rows) {
  rows=rows.filter(r=>r.variavel!=='Intercept');
  if(!rows.length)return empty('Nenhum coeficiente','A busca não retornou coeficientes identificáveis.');
  const visible=rows.slice(0,24),w=1000,left=320,right=880,top=30,h=visible.length*30+65;
  const min=Math.min(...visible.map(r=>r.ic_inferior),0),max=Math.max(...visible.map(r=>r.ic_superior),0),range=max-min || 1;
  const x=v=>left+(v-min)/range*(right-left);
  return `<div class="forest"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Coeficientes em log com intervalos de confiança de 95% pelo desenho amostral"><line x1="${x(0)}" x2="${x(0)}" y1="12" y2="${h-35}" stroke="#9daec4" stroke-dasharray="4 4"/>${[min,min+(max-min)/2,max].map(v=>`<text x="${x(v)}" y="${h-10}" text-anchor="middle">${fmt(v,3)}</text>`).join('')}${visible.map((r,i)=>{const y=top+i*30;return `<g tabindex="0"><title>${esc(r.rotulo)}: β ${fmt(r.coeficiente,5)}; IC95% [${fmt(r.ic_inferior,5)}; ${fmt(r.ic_superior,5)}]. Referência: ${esc(r.referencia)}</title><text x="${left-15}" y="${y+4}" text-anchor="end">${esc(r.rotulo.length>43?r.rotulo.slice(0,42)+'…':r.rotulo)}</text><line x1="${x(r.ic_inferior)}" x2="${x(r.ic_superior)}" y1="${y}" y2="${y}" stroke="#6d94df" stroke-width="2"/><line x1="${x(r.ic_inferior)}" x2="${x(r.ic_inferior)}" y1="${y-4}" y2="${y+4}" stroke="#6d94df"/><line x1="${x(r.ic_superior)}" x2="${x(r.ic_superior)}" y1="${y-4}" y2="${y+4}" stroke="#6d94df"/><circle cx="${x(r.coeficiente)}" cy="${y}" r="4" fill="#2f6df6"/><text x="${right+20}" y="${y+4}">${fmt(r.coeficiente,3)}</text></g>`;}).join('')}</svg></div><p class="note">${visible.length} coeficientes no gráfico${rows.length>24?'; todos os resultados filtrados estão na tabela':''}. Escala: coeficiente β em log. A linha tracejada marca zero.</p>`;
}
const EDU_COLORS=['#2f6df6','#13795b','#a96100','#7450a8','#b33b50'];
const EDU_DASH=['','7 4','2 3','10 3 2 3','5 3'];
function educationChart(rows, contrast='acumulado') {
  const valid=rows.filter(r=>r.identificavel&&Number.isFinite(Number(r.associacao_pct))&&Number.isFinite(Number(r.ic_inferior_pct))&&Number.isFinite(Number(r.ic_superior_pct)));
  if(!valid.length)return empty('Curva indisponível','Não há estimativas identificáveis para este grupo.');
  const groups=[...new Map(valid.map(r=>[r.grupo_codigo,{code:r.grupo_codigo,label:r.grupo}])).values()];
  const values=valid.flatMap(r=>[Number(r.ic_inferior_pct),Number(r.ic_superior_pct)]).concat([0]);
  let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max((hi-lo)*.09,2);lo-=pad;hi+=pad;
  const w=1000,h=430,left=84,right=970,top=24,bottom=350,plotH=bottom-top,plotW=right-left;
  const x=v=>left+(v/16)*plotW,y=v=>bottom-(v-lo)/(hi-lo)*plotH;
  const ticks=Array.from({length:5},(_,i)=>lo+(hi-lo)*i/4);
  const xticks=contrast==='acumulado'?[0,1,4,8,11,12,16]:[1,4,8,11,12,16];
  const lines=groups.map((g,gi)=>{
    const points=[...(contrast==='acumulado'?[{anos_de_estudo_codigo:0,anos_de_estudo_rotulo:'0 anos',contraste_rotulo:'Código 0',associacao_pct:0,ic_inferior_pct:0,ic_superior_pct:0}]:[]),...valid.filter(r=>String(r.grupo_codigo)===String(g.code)).sort((a,b)=>a.anos_de_estudo_codigo-b.anos_de_estudo_codigo)];
    const jitter=(gi-(groups.length-1)/2)*Math.min(.13,.52/groups.length);
    const path=points.map((p,i)=>`${i?'L':'M'}${x(Number(p.anos_de_estudo_codigo)+jitter).toFixed(1)},${y(Number(p.associacao_pct)).toFixed(1)}`).join(' ');
    const marks=points.map((p,i)=>{const xp=x(Number(p.anos_de_estudo_codigo)+jitter),yp=y(Number(p.associacao_pct)),yl=y(Number(p.ic_inferior_pct)),yu=y(Number(p.ic_superior_pct)),base=p.contraste_rotulo==='Código 0';return `<g tabindex="0"><title>${esc(g.label)} · ${esc(p.contraste_rotulo||'Código 0')}: ${fmt(p.associacao_pct,1)}% (IC95% ${fmt(p.ic_inferior_pct,1)}% a ${fmt(p.ic_superior_pct,1)}%)</title>${base?'':`<line x1="${xp}" x2="${xp}" y1="${yu}" y2="${yl}" stroke="${EDU_COLORS[gi%EDU_COLORS.length]}" stroke-width="1.4" opacity=".72"/>`}<circle cx="${xp}" cy="${yp}" r="${i?4:3}" fill="${EDU_COLORS[gi%EDU_COLORS.length]}" stroke="white" stroke-width="1.5"/></g>`;}).join('');
    return `<path d="${path}" fill="none" stroke="${EDU_COLORS[gi%EDU_COLORS.length]}" stroke-width="2" ${EDU_DASH[gi%EDU_DASH.length]?`stroke-dasharray="${EDU_DASH[gi%EDU_DASH.length]}"`:''}/>${marks}`;
  }).join('');
  const baseline=y(0);
  return `<div class="edu-legend">${groups.map((g,i)=>`<span><i style="--legend-color:${EDU_COLORS[i%EDU_COLORS.length]};--legend-dash:${EDU_DASH[i%EDU_DASH.length]||'none'}"></i>${esc(g.label)}</span>`).join('')}</div><div class="edu-chart-wrap"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${contrast==='acumulado'?'Diferença acumulada':'Retorno associado a um ano adicional'} de escolaridade no salário-hora, com intervalos de confiança de 95% pelo desenho amostral.">${ticks.map(v=>`<g><line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" class="grid-line"/><text x="${left-10}" y="${y(v)+4}" text-anchor="end">${fmt(v,0)}%</text></g>`).join('')}<line x1="${left}" x2="${right}" y1="${baseline}" y2="${baseline}" stroke="#718097" stroke-dasharray="4 4"/>${xticks.map(v=>`<g><line x1="${x(v)}" x2="${x(v)}" y1="${bottom}" y2="${bottom+5}" stroke="#8b96a7"/><text x="${x(v)}" y="${bottom+23}" text-anchor="middle">${v===16?'16+':v}</text></g>`).join('')}<line x1="${left}" x2="${right}" y1="${bottom}" y2="${bottom}" stroke="#8b96a7"/><line x1="${left}" x2="${left}" y1="${top}" y2="${bottom}" stroke="#8b96a7"/><text x="${(left+right)/2}" y="${h-20}" text-anchor="middle">${contrast==='marginal'?'Nível final de estudo (VD3005)':'Anos de estudo (VD3005)'}</text><text transform="translate(18 ${(top+bottom)/2}) rotate(-90)" text-anchor="middle">${contrast==='marginal'?'Retorno de 1 ano adicional':'Associação vs código 0'}</text>${lines}</svg></div>`;
}
const EDU_TEST_LABELS={associacao_escolaridade:'Teste conjunto da escolaridade',desvio_da_forma_linear:'Desvio em relação à tendência linear',componentes_nao_lineares:'Componentes não lineares da spline',heterogeneidade_sexo:'Diferença conjunta entre perfis por sexo',heterogeneidade_cor:'Diferença conjunta entre perfis por cor/raça',heterogeneidade_regiao:'Diferença conjunta entre perfis por macrorregião'};
function educationPanel() {
  const data=educationData||{status:'unavailable',message:'Resultados de escolaridade indisponíveis.'};
  const dim=S.educationDimension||'geral',spec=S.educationSpec||'categorias',contrast=S.educationContrast||'acumulado';
  if(data.status!=='ok')return `<section class="panel" id="education-panel"><div class="panel-heading"><div><h2>Retorno associado à educação</h2><p>Comparação ponderada com incerteza pelo desenho amostral</p></div>${badge('V1028 · 2013–2025','green')}</div>${empty('Análise de escolaridade indisponível',data.message)}</section>`;
  const tests=Object.entries(data.tests||{}).map(([key,t])=>`<div class="edu-test"><span>${esc(EDU_TEST_LABELS[key]||key)}</span><strong>${t.p_valor==null?'—':pfmt(t.p_valor)}</strong><small>F(${fmt(t.gl_numerador,0)}, ${fmt(t.gl_denominador,0)}) = ${fmt(t.f,2)} · Wald-F aproximado${t.p_valor_bh==null?'':` · BH ${pfmt(t.p_valor_bh)}`}</small></div>`).join('');
  const specOptions=[['categorias','Categorias de escolaridade'],['linear','Tendência linear em anos'],['spline','Spline linear por segmentos']];
  const dimOptions=[['geral','Brasil · modelo geral'],['sexo','Perfis por sexo'],['cor','Perfis por cor/raça'],['regiao','Perfis por macrorregião']];
  const modelLabel=dim==='geral'?(specOptions.find(x=>x[0]===spec)?.[1]||'Categorias de escolaridade'):'Spline com interação por '+(dim==='regiao'?'macrorregião':dim==='cor'?'cor/raça':'sexo');
  const chartRows=data.rows||[];
  const tableRows=chartRows.map(r=>({...r,rotulo:`${r.grupo} · ${r.contraste_rotulo}`}));
  const contrastLabel=contrast==='marginal'?'Retorno associado a um ano adicional':'Diferença acumulada contra o código 0';
  return `<section class="panel" id="education-panel"><div class="panel-heading"><div><h2>Retorno associado à educação</h2><p>${contrastLabel} no salário-hora, com incerteza amostral</p></div>${badge('Ponderado por V1028','green')}</div><div class="inline-options edu-controls"><label>Forma da escolaridade<select id="education-spec" ${dim!=='geral'?'disabled':''}>${specOptions.map(([v,l])=>`<option value="${v}" ${dim==='geral'&&spec===v?'selected':''}>${l}</option>`).join('')}</select></label><label>Heterogeneidade por<select id="education-dimension">${dimOptions.map(([v,l])=>`<option value="${v}" ${dim===v?'selected':''}>${l}</option>`).join('')}</select></label><label>Medida do retorno<select id="education-contrast"><option value="acumulado" ${contrast==='acumulado'?'selected':''}>Diferença acumulada vs. código 0</option><option value="marginal" ${contrast==='marginal'?'selected':''}>Diferença por ano adicional</option></select></label><span class="badge">${esc(modelLabel)}</span></div><div class="mini-metrics edu-summary"><div><strong>${fmt(data.model.n_observacoes_elegiveis,0)}</strong><span>Observações elegíveis · amostra comum</span></div><div><strong>${fmt(data.model.n_upas,0)}</strong><span>UPAs no desenho</span></div><div><strong>${fmt(data.model.n_estratos,0)}</strong><span>Estratos</span></div></div>${tests?`<div class="edu-tests">${tests}</div>`:''}<div class="edu-plot-heading"><h3>Escolaridade e salário-hora</h3><span>Estimativa pontual e IC95% por nível</span></div>${educationChart(chartRows,contrast)}<p class="note edu-note">${(data.notes||[]).map(esc).join(' ')} ${dim==='geral'&&spec==='categorias'?'Os pontos categóricos são unidos apenas como guia visual.':''} ${contrast==='marginal'?'Na forma categórica, cada ponto compara códigos adjacentes; 16+ é comparado a 15.':''} Percentuais: 100 × (exp(Δ log salário-hora) − 1). Os intervalos são pontuais, não simultâneos.</p><div class="table-tools edu-table-heading"><strong>Estimativas e incerteza</strong><span>${tableRows.length} comparações</span></div><div id="education-table"></div></section>`;
}
function wireEducation() {
  if(!$('#education-spec'))return;
  const update=async()=>{
    S.educationDimension=$('#education-dimension').value;
    S.educationContrast=$('#education-contrast').value;
    if(S.educationDimension==='geral')S.educationSpec=$('#education-spec').value;
    else S.educationSpec='spline';
    save();
    const panelNode=$('#education-panel');if(panelNode)panelNode.innerHTML=loading('Atualizando as estimativas…');
    const q=selection();q.set('especificacao',S.educationSpec);q.set('dimensao',S.educationDimension);q.set('contraste',S.educationContrast);
    const requestId=++educationRevision;
    try{const result=await api('/api/educacao?'+q);if(requestId!==educationRevision)return;educationData=result;}catch(e){if(requestId!==educationRevision)return;educationData={status:'unavailable',message:e.message};}
    if($('#education-panel')){$('#education-panel').outerHTML=educationPanel();wireEducation();}
  };
  $('#education-spec').onchange=update;$('#education-dimension').onchange=update;$('#education-contrast').onchange=update;
  const rows=(educationData.rows||[]).map(r=>({...r,rotulo:`${r.grupo} · ${r.contraste_rotulo}`}));
  makeTable('education-table',rows,[
    {key:'grupo',label:'Grupo'},
    {key:'contraste_rotulo',label:'Comparação'},
    {key:'associacao_pct',label:'Associação %',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'ic_inferior_pct',label:'IC95% inferior',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'ic_superior_pct',label:'IC95% superior',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'p_valor',label:'p-valor',format:pfmt},
    {key:'p_valor_bh',label:'p ajustado BH',format:pfmt},
  ],educationData.provenance,r=>detail(r.rotulo,[['Referência',r.referencia],['β em log',fmt(r.beta_log,5)],['Associação percentual',fmt(r.associacao_pct,2)+'%'],['IC95%',`${fmt(r.ic_inferior_pct,2)}% a ${fmt(r.ic_superior_pct,2)}%`],['p-valor',pfmt(r.p_valor)],['p ajustado BH',pfmt(r.p_valor_bh)],['UPAs',fmt(educationData.model.n_upas,0)],['Estratos',fmt(educationData.model.n_estratos,0)]],'Estimativa condicional entre pessoas ocupadas. Não é um efeito causal.'));
}
function regionalTestCard(label,test) {
  const p=test?.p_valor_apresentacao??(test?.p_valor==null?null:pfmt(test.p_valor));
  const pbh=test?.p_valor_bh_apresentacao??(test?.p_valor_bh==null?null:pfmt(test.p_valor_bh));
  return `<div class="edu-test"><span>${esc(label)}</span><strong>${p==null?'—':esc(p)}</strong><small>F(${fmt(test?.gl_numerador,0)}, ${fmt(test?.gl_denominador,0)}) = ${fmt(test?.f,2)} · Wald-F${pbh==null?'':` · BH ${esc(pbh)}`}</small></div>`;
}
function regionalPanel() {
  const data=regionalData||{status:'unavailable',message:'A análise regional ainda não foi carregada.'};
  if(data.status!=='ok')return `<section class="panel" id="regional-panel"><div class="panel-heading"><div><h2>Heterogeneidade regional</h2><p>Retornos da escolaridade e formalidade por macroregião</p></div>${badge('V1028 · 2013–2025','green')}</div>${empty('Análise regional indisponível',data.message)}</section>`;
  const summary=data.summary,approach=S.regionalApproach||'Pooled com interações',contrast=S.regionalContrast||'acumulado';
  const selected=(data.education||[]).filter(r=>r.abordagem===approach&&r.contraste===contrast);
  const chartRows=selected.map(r=>({...r,grupo_codigo:r.regiao,grupo:r.regiao}));
  const educationLabel=contrast==='marginal'?'Diferença por ano adicional':'Diferença acumulada contra o código 0';
  const tests=summary.testes||{};
  const model=summary.modelos?.pooled_interacoes||{};
  const formalRows=(data.formality||[]).filter(r=>r.abordagem===approach);
  const fitRows=data.comparison||[];
  return `<section class="panel" id="regional-panel"><div class="panel-heading"><div><h2>Heterogeneidade regional</h2><p>Escolaridade e formalidade no salário-hora, com incerteza amostral</p></div>${badge('Ponderado por V1028','green')}</div><div class="mini-metrics edu-summary"><div><strong>${fmt(model.n_observacoes,0)}</strong><span>Observações na amostra comum</span></div><div><strong>${fmt(model.n_upas_no_desenho_completo,0)}</strong><span>UPAs no desenho completo</span></div><div><strong>${fmt(model.n_estratos_no_desenho_completo,0)}</strong><span>Estratos</span></div><div><strong>${fmt(model.graus_liberdade_desenho,0)}</strong><span>Graus de liberdade · desenho</span></div></div><div class="edu-tests regional-tests">${regionalTestCard('Escolaridade × macrorregião',tests.heterogeneidade_escolaridade)}${regionalTestCard('Formalidade × macrorregião',tests.heterogeneidade_formalidade)}${regionalTestCard('Interações em conjunto',tests.interacoes_conjuntas)}</div><div class="inline-options edu-controls regional-controls"><label>Abordagem<select id="regional-approach"><option value="Pooled com interações" ${approach==='Pooled com interações'?'selected':''}>Modelo conjunto com interações</option><option value="Modelos regionais separados" ${approach==='Modelos regionais separados'?'selected':''}>Modelos separados por região</option></select></label><label>Medida do retorno<select id="regional-contrast"><option value="acumulado" ${contrast==='acumulado'?'selected':''}>Diferença acumulada vs. código 0</option><option value="marginal" ${contrast==='marginal'?'selected':''}>Diferença por nível adjacente</option></select></label><span class="badge">Referência das interações pooled: ${esc(summary.referencia_regional)}</span></div><div class="edu-plot-heading"><h3>Escolaridade e salário-hora por macroregião</h3><span>${educationLabel} · estimativa e IC95%</span></div>${educationChart(chartRows,contrast)}<p class="note regional-note">${(data.notes||[]).map(esc).join(' ')} No modelo conjunto, escolaridade e formalidade variam por macroregião; os demais controles são comuns. Os modelos separados estimam os controles dentro de cada macroregião. ${educationLabel}; percentuais calculados por 100 × (exp(Δ log salário-hora) − 1). Intervalos pontuais, não simultâneos.</p><div class="table-tools edu-table-heading"><strong>Retornos da escolaridade</strong><span>${selected.length} estimativas · ${approach}</span></div><div id="regional-education-table"></div><div class="edu-plot-heading edu-table-heading"><h3>Associação da formalidade</h3><span>Formal vs. informal · estimativa e IC95%</span></div><div id="regional-formality-table"></div><div class="table-tools edu-table-heading"><strong>Ajuste pooled e separado</strong><span>RMSE log e R² ponderados, calculados na amostra de estimação</span></div><div id="regional-fit-table"></div></section>`;
}
function wireRegional() {
  if(regionalData?.status!=='ok')return;
  const rerender=()=>{const region=$('#regional-panel');if(region){S.regionalApproach=$('#regional-approach').value;S.regionalContrast=$('#regional-contrast').value;save();region.outerHTML=regionalPanel();wireRegional();}};
  $('#regional-approach').onchange=rerender;$('#regional-contrast').onchange=rerender;
  const approach=S.regionalApproach||'Pooled com interações',contrast=S.regionalContrast||'acumulado';
  const rows=(regionalData.education||[]).filter(r=>r.abordagem===approach&&r.contraste===contrast);
  makeTable('regional-education-table',rows,[
    {key:'regiao',label:'Macrorregião'},
    {key:'anos_de_estudo',label:'Escolaridade'},
    {key:'associacao_pct',label:'Associação %',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'ic_inferior_pct',label:'IC95% inferior',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'ic_superior_pct',label:'IC95% superior',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'p_valor',label:'p-valor',format:pfmt},
    {key:'p_valor_bh',label:'p ajustado BH',format:pfmt},
  ],regionalData.provenance,r=>detail(`${r.regiao} · ${r.contraste_rotulo}`,[['Abordagem',r.abordagem],['Referência',r.referencia],['β em log',fmt(r.beta_log,5)],['Associação',fmt(r.associacao_pct,2)+'%'],['IC95%',`${fmt(r.ic_inferior_pct,2)}% a ${fmt(r.ic_superior_pct,2)}%`],['p-valor',pfmt(r.p_valor)],['p ajustado BH',pfmt(r.p_valor_bh)],['Graus de liberdade',fmt(r.graus_liberdade,0)]],'Associação condicional entre pessoas ocupadas. Não é um efeito causal.'));
  const formalRows=(regionalData.formality||[]).filter(r=>r.abordagem===approach);
  makeTable('regional-formality-table',formalRows,[
    {key:'regiao',label:'Macrorregião'},
    {key:'associacao_pct',label:'Formal vs. informal',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'ic_inferior_pct',label:'IC95% inferior',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'ic_superior_pct',label:'IC95% superior',format:v=>v==null?'—':fmt(v,1)+'%'},
    {key:'p_valor',label:'p-valor',format:pfmt},
    {key:'p_valor_bh',label:'p ajustado BH',format:pfmt},
  ],regionalData.provenance,r=>detail(`${r.regiao} · associação da formalidade`,[['Abordagem',r.abordagem],['β em log',fmt(r.beta_log,5)],['Associação',fmt(r.associacao_pct,2)+'%'],['IC95%',`${fmt(r.ic_inferior_pct,2)}% a ${fmt(r.ic_superior_pct,2)}%`],['p-valor',pfmt(r.p_valor)],['p ajustado BH',pfmt(r.p_valor_bh)],['Graus de liberdade',fmt(r.graus_liberdade,0)]],r.interpretacao));
  makeTable('regional-fit-table',regionalData.comparison||[],[
    {key:'regiao',label:'Macrorregião'},
    {key:'rmse_log_pooled_comum',label:'RMSE · pooled comum',format:v=>fmt(v,4)},
    {key:'rmse_log_pooled_interacoes',label:'RMSE · pooled interações',format:v=>fmt(v,4)},
    {key:'rmse_log_separado',label:'RMSE · separado',format:v=>fmt(v,4)},
    {key:'r2_ponderado_pooled_comum',label:'R² · pooled comum',format:v=>fmt(v,3)},
    {key:'r2_ponderado_pooled_interacoes',label:'R² · pooled interações',format:v=>fmt(v,3)},
    {key:'r2_ponderado_separado',label:'R² · separado',format:v=>fmt(v,3)},
    {key:'n_observacoes',label:'N amostral',format:v=>fmt(v,0)},
  ],regionalData.provenance);
}
function selectionChart(rows) {
  if(!rows.length)return empty('Taxas indisponíveis','Não há grupos para exibir nesta dimensão.');
  const sorted=[...rows].sort((a,b)=>a.dimensao==='escolaridade'?Number(a.grupo_codigo)-Number(b.grupo_codigo):String(a.grupo).localeCompare(String(b.grupo),'pt-BR'));
  const w=1000,h=58+sorted.length*31,left=325,right=930,top=44,plotW=right-left;
  const x=v=>left+(Number(v)/100)*plotW;
  return `<div class="selection-legend"><span><i class="selection-key occupied"></i>Ocupadas</span><span><i class="selection-key observed"></i>Salário-hora válido</span></div><div class="selection-chart-wrap"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Taxas ponderadas de ocupação e de salário-hora válido por grupo">${[0,25,50,75,100].map(v=>`<g><line x1="${x(v)}" x2="${x(v)}" y1="${top-12}" y2="${h-20}" class="grid-line"/><text x="${x(v)}" y="${h-4}" text-anchor="middle">${v}%</text></g>`).join('')}${sorted.map((r,i)=>{const y=top+i*31;return `<g tabindex="0"><title>${esc(r.grupo)}: ocupadas ${fmt(r.taxa_ocupacao_pct,1)}%; salário-hora válido ${fmt(r.taxa_salario_valido_pct,1)}%</title><text x="${left-12}" y="${y+7}" text-anchor="end">${esc(r.grupo)}</text><rect x="${left}" y="${y-8}" width="${plotW}" height="6" rx="2" fill="#eef2f7"/><rect x="${left}" y="${y-8}" width="${Math.max(0,x(r.taxa_ocupacao_pct)-left)}" height="6" rx="2" fill="#2f6df6"/><rect x="${left}" y="${y+2}" width="${plotW}" height="6" rx="2" fill="#eef2f7"/><rect x="${left}" y="${y+2}" width="${Math.max(0,x(r.taxa_salario_valido_pct)-left)}" height="6" rx="2" fill="#13795b"/></g>`;}).join('')}</svg></div>`;
}
function selectionPanel() {
  const data=selectionData||{status:'unavailable',message:'Resultados de seleção ocupacional indisponíveis.'};
  if(data.status!=='ok')return `<section class="panel" id="selection-panel"><div class="panel-heading"><div><h2>Seleção para a amostra salarial</h2><p>Participação no trabalho e sensibilidade da amostra com salário válido</p></div>${badge('V1028 · 2013–2025','green')}</div>${empty('Análise de seleção indisponível',data.message)}</section>`;
  const summary=data.summary,dim=S.selectionDimension||'escolaridade';
  const labels={escolaridade:'Escolaridade',sexo:'Sexo',cor:'Cor/raça',regiao:'Macrorregião'};
  const rows=(data.rates||[]).filter(r=>r.dimensao===dim),effects=data.effects||[];
  return `<section class="panel" id="selection-panel"><div class="panel-heading"><div><h2>Seleção para a amostra salarial</h2><p>Quem está ocupado e quem tem salário-hora observável</p></div>${badge('Sensibilidade ponderada · V1028','green')}</div><div class="mini-metrics selection-summary"><div><strong>${pct(summary.taxa_ocupacao_pct/100)}</strong><span>População ocupada · peso V1028</span></div><div><strong>${pct(summary.taxa_salario_valido_pct/100)}</strong><span>Salário-hora válido · base-alvo</span></div><div><strong>${fmt(summary.n_alvo,0)}</strong><span>Observações no domínio-alvo</span></div><div><strong>${fmt(summary.propensao.p05_ponderado*100,1)}%</strong><span>5º percentil da probabilidade estimada</span></div></div><div class="inline-options selection-controls"><label>Comparar taxas por<select id="selection-dimension">${Object.entries(labels).map(([v,l])=>`<option value="${v}" ${v===dim?'selected':''}>${l}</option>`).join('')}</select></label><span class="badge">${fmt(summary.n_salario_valido,0)} salários válidos</span><span class="badge">${fmt(summary.modelo_salario.n_efetivo_kish_ipw,0)} tamanho efetivo após IPW</span></div><div class="edu-plot-heading"><h3>Participação ponderada por grupo</h3><span>Percentual da população-alvo</span></div>${selectionChart(rows)}<p class="note selection-note">Azul: pessoas ocupadas. Verde: ocupadas com rendimento e horas positivos. As taxas usam V1028. A probabilidade estima a entrada na amostra salarial com escolaridade, sexo, cor/raça, idade, UF e período; ${fmt(summary.propensao.n_abaixo_piso,0)} probabilidades abaixo de 5% foram previstas no domínio-alvo. Na amostra salarial, ${fmt(summary.propensao.n_pesos_salariais_limitados,0)} pesos inversos foram efetivamente limitados a 20. Filtros globais não reestimam esta análise.</p><div class="table-tools edu-table-heading"><strong>Taxas por grupo</strong><span>${rows.length} grupos</span></div><div id="selection-rate-table"></div><div class="table-tools edu-table-heading"><strong>Retorno da escolaridade: base e sensibilidade IPW</strong><span>IC95% da estimativa IPW · diferença em pontos percentuais</span></div><div id="selection-ipw-table"></div><p class="note selection-note">O IPW aumenta associações condicionais entre ${fmt(data.summary.sensibilidade.mudanca_minima_pp,2)} e ${fmt(data.summary.sensibilidade.mudanca_maxima_pp,2)} pontos percentuais nesta amostra. Essa mudança não foi testada como diferença entre estimadores. A seleção por fatores não observados permanece sem identificação; o resultado não é causal nem substitui uma correção de Heckman.</p></section>`;
}
function wireSelection() {
  if(selectionData?.status!=='ok')return;
  const select=$('#selection-dimension');
  if(select)select.onchange=()=>{S.selectionDimension=select.value;save();$('#selection-panel').outerHTML=selectionPanel();wireSelection();};
  const rows=(selectionData.rates||[]).filter(r=>r.dimensao===S.selectionDimension);
  makeTable('selection-rate-table',rows,[
    {key:'grupo',label:'Grupo'},
    {key:'n_alvo',label:'N amostral',format:v=>fmt(v,0)},
    {key:'taxa_ocupacao_pct',label:'Ocupadas %',format:v=>fmt(v,1)+'%'},
    {key:'taxa_salario_valido_pct',label:'Salário-hora válido %',format:v=>fmt(v,1)+'%'},
    {key:'taxa_salario_valido_entre_ocupados_pct',label:'Válido entre ocupados %',format:v=>fmt(v,1)+'%'},
  ],selectionData.provenance);
  makeTable('selection-ipw-table',selectionData.effects||[],[
    {key:'anos_de_estudo',label:'Escolaridade'},
    {key:'associacao_base_pct',label:'Base V1028',format:v=>fmt(v,1)+'%'},
    {key:'associacao_ipw_pct',label:'Sensibilidade IPW',format:v=>fmt(v,1)+'%'},
    {key:'ic_ipw_inferior_pct',label:'IC95% inferior',format:v=>fmt(v,1)+'%'},
    {key:'ic_ipw_superior_pct',label:'IC95% superior',format:v=>fmt(v,1)+'%'},
    {key:'mudanca_associacao_pp',label:'Mudança',format:v=>`${v>0?'+':''}${fmt(v,2)} p.p.`},
    {key:'p_valor_ipw_bh',label:'p ajustado BH',format:pfmt},
  ],selectionData.provenance);
}
function renderInference() {
  if(inferData.status!=='ok'){$('#content').innerHTML=panel('M5 ponderado · associação e inferência','Resultados fixos da execução oficial 2025T4 ou da série histórica',empty('Recorte inferencial indisponível',inferData.message)+`<p class="note" style="margin-top:18px">A estatística descritiva continua disponível. O M5 publicado aceita apenas recortes por macroregião.</p>`)+educationPanel()+regionalPanel()+selectionPanel();wireEducation();wireRegional();wireSelection();return;}
  const m=inferData.model;
  $('#content').innerHTML=`<div class="notice">${esc(inferData.selection)} Os coeficientes descrevem associações condicionais; não são efeitos causais.</div><div class="mini-metrics"><div><strong>${fmt(m.n_observacoes_elegiveis,0)}</strong><span>Observações elegíveis</span></div><div><strong>${fmt(m.r2,3)}</strong><span>R² ponderado · ajuste em log, dentro da amostra</span></div><div><strong>${fmt(m.graus_liberdade_desenho,0)}</strong><span>Graus de liberdade · UPAs − estratos</span></div></div>`+
    panel('Coeficientes e incerteza','Intervalos de confiança de 95% · linearização por UPA dentro de estrato',`<div class="inline-options"><label>Buscar variável<input id="coef-search" type="search" placeholder="Sexo, escolaridade, UF…"></label><label>Família de termos<select id="coef-family"><option value="all">Todos os termos</option><option value="main">Formalidade, sexo, idade e cor</option><option value="education">Escolaridade e interações</option><option value="uf">UF</option><option value="other">Setor e ocupação</option></select></label><label>Significância<select id="coef-significance"><option value="all">Todos os resultados</option><option value="raw">p &lt; 0,05</option><option value="bh">p ajustado BH &lt; 0,05</option></select></label></div><div id="forest-chart"></div>`)+
    educationPanel()+
    regionalPanel()+
    selectionPanel()+
    panel('Tabela inferencial','Selecione uma variável para ler a referência e a interpretação. Busca e ordenação são filtros de visualização.','<div id="coef-table"></div>')+
    panel('Como interpretar','Os termos precisam ser lidos à luz da especificação.',`<div class="method-list"><article><h3>Indicadoras em um modelo log-linear</h3><p>Quando apropriado, a associação percentual é 100 × (exp(β) − 1). Para sexo, formalidade e setor público, esta transformação se refere a 0 anos de estudo, pois existem interações com escolaridade. Para outro nível, combine os coeficientes; o intervalo dessa combinação exige a covariância.</p></article><article><h3>Idade, escolaridade e interações</h3><p>Idade está centrada em 45 anos e aparece também ao quadrado. Interações e termos contínuos não recebem automaticamente uma interpretação percentual. As categorias de anos de estudo comparam-se a 0 anos, no perfil de referência das interações.</p></article><article><h3>Variância e múltiplos testes</h3><p>${esc(inferData.method)} ${esc(inferData.variance)} BH controla a taxa de falsas descobertas na família de coeficientes publicada; não foi recalculado após filtros de visualização.</p></article></div><details style="margin-top:18px"><summary class="note">${inferData.omitted.length} termos constantes ou redundantes omitidos</summary><p class="note">${esc(inferData.omitted.join(', '))}</p></details>`);
  const update=()=>{const q=$('#coef-search').value.toLocaleLowerCase('pt-BR'),f=$('#coef-family').value,s=$('#coef-significance').value;
    const rows=inferData.rows.filter(r=>(r.rotulo+' '+r.variavel).toLocaleLowerCase('pt-BR').includes(q)&&(f==='all'||f==='main'&&/^(is_|Sexo_|Idade|Cor_)/.test(r.variavel)||f==='education'&&r.variavel.startsWith('Estudo')||f==='uf'&&r.variavel.startsWith('UF_')||f==='other'&&/^(Setor_|Ocup_)/.test(r.variavel))&&(s==='all'||s==='raw'&&r.p_valor<.05||s==='bh'&&r.p_bh<.05));
    $('#forest-chart').innerHTML=forest(rows);
    makeTable('coef-table',rows,[{key:'rotulo',label:'Variável',sub:r=>r.referencia},{key:'coeficiente',label:'β',format:v=>fmt(v,5)},{key:'associacao_pct',label:'Associação %',format:v=>v==null?'Interpretação específica':fmt(v)+'%'},{key:'erro_padrao',label:'Erro-padrão',format:v=>fmt(v,5)},{key:'ic_inferior',label:'IC95% inferior',format:v=>fmt(v,5)},{key:'ic_superior',label:'IC95% superior',format:v=>fmt(v,5)},{key:'p_valor',label:'p-valor',format:pfmt},{key:'p_bh',label:'p ajustado BH',format:pfmt}],inferData.provenance,r=>detail(r.rotulo,[['Variável no modelo',r.variavel],['Referência',r.referencia],['Coeficiente β',fmt(r.coeficiente,6)],['Associação percentual',r.associacao_pct==null?'Exige interpretação específica':fmt(r.associacao_pct)+'%'],['Erro-padrão pelo desenho',fmt(r.erro_padrao,6)],['IC95% de β',`[${fmt(r.ic_inferior,6)}; ${fmt(r.ic_superior,6)}]`],['p-valor',pfmt(r.p_valor)],['p ajustado BH',pfmt(r.p_bh)],['Método de variância',inferData.method]],r.tipo==='indicadora_condicional'?'Associação percentual condicional ao perfil de referência das interações. Não representa uma diferença média populacional nem um efeito causal.':'Associação condicional aos controles do M5. Nenhuma interpretação causal é identificada.'));
  };
  $('#coef-search').oninput=update;$('#coef-family').onchange=update;$('#coef-significance').onchange=update;update();wireEducation();wireRegional();wireSelection();
}

function renderMethodology() {
  const h=modelData.historical,p=modelData.prediction;
  const source=catalog.sources.find(s=>s.id==='historico');
  const ponderado=source?.provenance?.ponderacao==='V1028';
  const historicoTitulo=ponderado?'2013–2025 · V1028, Estrato e UPA':'2013–2025 · base histórica não ponderada';
  const historicoTexto=ponderado
    ? 'A base reúne pessoas de 25 a 65 anos, ocupadas ou não, e preserva peso, estrato e UPA. Médias e quantis usam V1028; salário-hora considera ocupados com renda e horas válidas. O M5 histórico está estimado para o Brasil e as macroregiões; o ranking preditivo continua separado e não ponderado.'
    : 'A base histórica local não contém peso e desenho amostral. As descrições representam a amostra disponível, e não totais populacionais. O ranking preditivo segue identificado como exploratório.';
  const historico=panel('Três origens, três leituras','A ponderação é uma propriedade do resultado, não da interface.',
    `<div class="method-list"><article>${badge(ponderado?'Série histórica ponderada':'Histórico exploratório',ponderado?'green':'amber')}<h3 style="margin-top:10px">${historicoTitulo}</h3><p>${historicoTexto} ${fmt(source?.n_registros,0)} registros pessoa-trimestre.</p></article><article>${badge('Execução oficial ponderada','green')}<h3 style="margin-top:10px">2025T4 · V1028, Estrato e UPA</h3><p>O domínio descritivo contém pessoas de 25 a 65 anos. A média de salário-hora usa ocupados com rendimento e horas positivas. A formalidade usa ocupados com classificação válida. Médias, proporções e quantis populacionais usam V1028.</p></article><article>${badge('Nova execução preditiva')}<h3 style="margin-top:10px">${esc(p.modelo || 'Artefato ainda não exportado')}</h3><p>Os modelos e hiperparâmetros são escolhidos pelo RMSE log da validação temporal; o teste é avaliado apenas para o vencedor. Esta execução usa nova amostra reproduzível e não substitui o ranking histórico. A previsão é não ponderada.</p></article></div>`);
  const inferenciaHistorica='O M5 ponderado combina os 52 trimestres de 2013–2025 e não muda ao selecionar um período. O retorno da educação compara categorias, tendência linear e spline na mesma amostra, com perfis por sexo, cor/raça e macrorregião. A heterogeneidade regional compara interações de escolaridade e formalidade com modelos regionais separados. A análise de seleção apresenta taxas de ocupação e uma sensibilidade por pesos inversos para a amostra salarial. Nenhuma dessas análises é reestimada por filtros descritivos. A variância é aproximada e registra uma UPA reclassificada entre dois estratos.';
  const dominio=panel('Domínios e interpretação','Convenções preservadas do pipeline.',
    `<div class="method-list"><article><h3>Salário-hora e deflação</h3><p>Rendimento habitual do trabalho principal / (horas habituais semanais × 4,33). Os rendimentos tratados estão a preços de janeiro/2026 pelo IPCA. Os valores nominais originais não estão disponíveis nesta interface.</p></article><article><h3>Quantis e dispersão</h3><p>Quantis populacionais: inversa da distribuição empírica ponderada, sem interpolação, reutilizando o módulo de amostragem. Desvio-padrão descreve a distribuição; não é o erro-padrão por UPA e estrato.</p></article><article><h3>Recortes e reestimação</h3><p>Filtros recalculam descritivas no servidor. O M5 está disponível para a validação oficial 2025T4 e para a série histórica empilhada. ${inferenciaHistorica} Rankings e o modelo preditivo global não mudam com os filtros descritivos.</p></article><article><h3>Contagem de pessoas</h3><p>Um trimestre usa o domínio populacional daquele trimestre. Ao empilhar períodos, registros e soma de pesos contam pessoas-trimestres; não representam pessoas únicas.</p></article></div>`);
  $('#content').innerHTML=rankingPanel()+`<div class="grid equal">${historico}${dominio}</div>`+
    panel('Modelo inferencial e desenho amostral','Associações salariais e sensibilidade da seleção.',`<div class="method-list"><article><h3>Escolaridade</h3><p>As formas categórica, linear e spline usam a mesma população elegível, V1028, controles e desenho. Testes conjuntos Wald-F aproximados verificam associação, linearidade e diferenças de perfil. Eles não escolhem a forma com base apenas em significância; intervalos por nível são pontuais.</p></article><article><h3>Heterogeneidade regional</h3><p>O modelo conjunto testa diferenças de escolaridade e formalidade entre macroregiões pela mesma covariância por UPA e estrato. Modelos regionais separados e métricas de ajuste ponderadas são apresentados como comparação dentro da amostra.</p></article><article><h3>Seleção para a amostra salarial</h3><p>A ponderação inversa repondera observações com salário-hora válido pela probabilidade estimada a partir de covariáveis observadas. Não há restrição de exclusão defensável nos campos desta série para uma correção de Heckman; a sensibilidade não identifica seleção não observada nem causalidade.</p></article><article><h3>Tratamento amostral</h3><p>A execução do pipeline registra o cadastro completo de UPAs antes de selecionar o domínio. Usa linearização por UPA dentro de estrato, aproximação com reposição, sem FPC e sem recalibrar V1028. A política explícita para estratos com UPA única é ajustar, com centragem na média global de scores. Rotação de domicílios e renovação da Amostra Mestra limitam a variância temporal empilhada.</p></article><article><h3>Interpretação</h3><p>As estimativas descrevem associações condicionais entre escolaridade e salário-hora de pessoas ocupadas. Não identificam causalidade nem salário potencial de pessoas fora da ocupação.</p></article><article><h3>Previsão e importância</h3><p>O ranking preditivo não é uma classificação de modelos inferenciais. Coeficientes Ridge não recebem p-valores OLS. Importância de variáveis de árvores não é apresentada como coeficiente inferencial. Projeções para trimestres futuros não estão disponíveis porque não há modelo temporal validado para esse objetivo.</p></article></div>`)+
    panel('Arquivos e reprodutibilidade','Entradas e saídas locais do projeto.',`<dl class="detail-list"><dt>Benchmark histórico</dt><dd><code>modelos_salario_hora/metricas_modelos.csv</code></dd><dt>Recomendação histórica</dt><dd>${esc(h.melhor_modelo_por_rmse_log_teste || 'Indisponível')} · ${esc(h.criterio || '')}</dd><dt>M5 ponderado 2025T4</dt><dd><code>validacao_real/2025T4/inferencia/</code></dd><dt>M5 ponderado 2013–2025</dt><dd><code>recoleta_ponderada/inferencia_amostral/</code></dd><dt>Retorno da educação</dt><dd><code>recoleta_ponderada/inferencia_amostral/retorno_educacao/</code></dd><dt>Heterogeneidade regional</dt><dd><code>recoleta_ponderada/inferencia_amostral/heterogeneidade_regional/</code></dd><dt>Seleção ocupacional</dt><dd><code>recoleta_ponderada/inferencia_amostral/selecao_ocupacional/</code></dd><dt>Nova previsão</dt><dd><code>previsao_perfis/manifesto.json</code> · ${esc(p.run_id || 'Indisponível')}</dd><dt>Exportar o modelo</dt><dd><code>python pnad_pipeline/scripts/12_treinar_previsao.py</code></dd><dt>Atualizar dados em uso</dt><dd>Reinicie o servidor depois de substituir arquivos processados ou reestimar resultados.</dd></dl><p class="note"><a href="/dashboard_salario_hora.html" target="_blank">Benchmark original ↗</a> · <a href="/inferencia_m5_autonomo.html" target="_blank">M5 ponderado 2025T4 ↗</a> · <a href="/resultados_testes_reais.html" target="_blank">Relatório 2025T4 e testes · snapshot 16/10 ↗</a></p>`);
  wireRanking();
}

async function refresh() {
  if(!catalog)return;
  const token=++revision,educationToken=++educationRevision,selectionToken=++selectionRevision;controller?.abort();controller=new AbortController();tables.clear();comparisonData=undefined;predictionData=undefined;seriesData=undefined;$('#export-all').disabled=true;
  S.view=location.hash.slice(1) in headings?location.hash.slice(1):'visao';save();activeFilters();
  $('#page-title').textContent=headings[S.view][0];$('#page-description').textContent=headings[S.view][1];
  document.querySelectorAll('[data-section]').forEach(a=>{if(a.dataset.section===S.view)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  $('#content').setAttribute('aria-busy','true');$('#content').innerHTML=loading('Consultando o recorte selecionado…');$('#notice').hidden=true;
  try {
    const q=selection();q.set('grupo',S.group);
    const qe=selection();qe.set('dimensao',S.educationDimension);qe.set('especificacao',S.educationDimension==='geral'?S.educationSpec:'spline');qe.set('contraste',S.educationContrast);
    const qs=selection();qs.set('dimensao',S.selectionDimension);
    const results=await Promise.all([api('/api/descritiva?'+q,{signal:controller.signal}),api('/api/modelos',{signal:controller.signal}),api('/api/inferencia?'+q,{signal:controller.signal}),api('/api/educacao?'+qe,{signal:controller.signal}).catch(e=>({status:'unavailable',message:e.message})),api('/api/selecao?'+qs,{signal:controller.signal}).catch(e=>({status:'unavailable',message:e.message})),api('/api/heterogeneidade-regional?'+q,{signal:controller.signal}).catch(e=>({status:'unavailable',message:e.message}))]);
    if(token!==revision)return;
    [current,modelData,inferData]=results;
    if(educationToken===educationRevision)educationData=results[3];
    if(selectionToken===selectionRevision)selectionData=results[4];
    regionalData=results[5];
    provenance(current.provenance);
    if(current.status==='empty'||current.status==='insufficient'){$('#notice').hidden=false;$('#notice').textContent=current.status==='empty'?'Nenhuma observação para este recorte. Altere os filtros ou limpe a seleção.':'Amostra insuficiente: menos de 30 salários-hora válidos. Estatísticas são exploratórias; o comparador não apresentará diferenças.';}
    ({visao:renderOverview,descritiva:renderDescriptive,previsao:renderPrediction,associacao:renderInference,metodologia:renderMethodology})[S.view]();
    $('#export-all').disabled=false;
    if($('#series-chart'))lineChart(token);
  } catch(e) {if(e.name!=='AbortError'&&token===revision)$('#content').innerHTML=empty('Não foi possível carregar os resultados',e.message)+`<button class="button secondary" id="retry" style="margin-top:16px">Tentar novamente</button>`;if($('#retry'))$('#retry').onclick=refresh;}
  finally {if(token===revision)$('#content').setAttribute('aria-busy','false');}
}
window.addEventListener('hashchange',refresh);
async function connect(){
$('#content').innerHTML=loading('Conectando ao pipeline local…');
for(const id of ['source','period','toggle-filters','clear-filters'])$('#'+id).disabled=true;
try {
  if(location.protocol==='file:')throw new Error('Este HTML depende do servidor Python para consultar os dados e executar previsões. Abra abrir_dashboard.cmd na pasta do projeto.');
  catalog=await api('/api/catalogo');
  if(!sourceData()?.available)S.source=catalog.sources.find(x=>x.available)?.id;
  if(!S.source)throw new Error('Nenhuma base processada disponível. Execute o pipeline e reinicie o servidor.');
  for(const [key,val]of Object.entries(S.filters))if(!sourceData().options[key]?.some(o=>o.value===val))delete S.filters[key];
  buildFilters();
  for(const id of ['source','period','toggle-filters','clear-filters'])$('#'+id).disabled=false;
  if(window.matchMedia('(max-width:520px)').matches){$('#toggle-filters').setAttribute('aria-expanded','false');$('#filter-fields').hidden=true;}
  refresh();
} catch(e) {$('#content').innerHTML=empty('Conexão com o pipeline indisponível',e.message)+`<p class="explanation">Dê dois cliques em <strong>abrir_dashboard.cmd</strong> ou execute: <code>python scripts/abrir_dashboard.py</code></p><div class="form-footer"><a class="button" href="http://127.0.0.1:8000">Abrir dashboard local →</a>${location.protocol!=='file:'?'<button class="button secondary" id="connect-retry">Tentar conectar novamente</button>':''}</div>`;$('#content').setAttribute('aria-busy','false');if($('#connect-retry'))$('#connect-retry').onclick=connect;}
}
await connect();
})();
