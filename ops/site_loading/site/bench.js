/* Benchmarks (Jacob 2026-11-01): table at the very top of affine.io (benchmarks = columns from the registry, grouped by suite;
   models = rows: kings newest first, then reference models (queued ones as … rows), Genesis at the bottom; each cell shaded relative to its column best (Jacob 2026-11-01 09:08 CR)).
   Table only: the per-benchmark chart tiles were removed (Jacob 2026-11-01 08:35 CR). Data: /benchmarks.json.
   References: only the Pareto front over (parameters, lower better) x (mean score over the table's columns, higher better) is shown;
   a ref is hidden if Genesis or another fully scored ref is no larger and scores higher. Kings and Genesis always shown; refs without a
   full set of scores (queued/evaluating/failed) or without a PARAMS entry are hidden (Jacob 2026-11-01 18:13 CR). Short names link to HF.
   Avg column (first score column, Jacob 18:26 CR) = mean over the shown benchmark columns (same mean as the Pareto filter; … when a row is incomplete). Shading: per column,
   min..max of the visible rows, red tint at the bottom -> strong green at the top; a small star marks each column max (ties all marked)
   (Jacob 2026-11-01 18:22 CR).
   Sparklines (Jacob pt1046u 2026-11-02): a row of small cards above the table, one per score column (Avg + each benchmark): the kings'
   score over time (x = crowned_at, scored kings only), LATEST = newest scored king; hover shows the king under the cursor.
   Restyled to the site's chart look (Jacob pt1048u): tile frame + Haffer heading/reading, k1-chart gridlines and mono labels, --performance line with
   hollow points, E-epoch x labels, .chart-tip hover tooltip.
   Kings sit in a scroll box ~5 rows deep (sticky header); references + Genesis are a sticky <tfoot> so they stay visible without
   scrolling through the kings (Jacob pt1046u). */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const pct=x=>x==null?'—':(100*x).toFixed(1);
 // parameter counts (safetensors total from the HF API / king index.json, 2026-11-01); add one when adding a reference
 const PARAMS={'genesis':9.41e9,'qwen3.5-9b':9.65e9,'qwen3-8b':8.19e9,'deepseek-r1-0528-qwen3-8b':8.19e9,'r1-distill-qwen-7b':7.62e9,
  'r1-distill-llama-8b':8.03e9,'nemotron-nano-9b-v2':8.89e9,'glm-z1-9b':9.40e9,'qwen3.5-4b':4.66e9,'qwen3-14b':14.77e9,'qwen3-4b':4.02e9,
  'r1-distill-qwen-14b':14.77e9,'openreasoning-nemotron-7b':7.62e9,'acereason-nemotron-1.1-7b':7.62e9,'skywork-or1-7b':7.62e9,'mimo-7b-rl':7.83e9,
  'llama-nemotron-nano-8b-v1':8.03e9,'openthinker3-7b':7.62e9,'phi-4-reasoning':14.66e9,'phi-4-mini-reasoning':3.84e9,'deepscaler-1.5b':1.78e9,
  'r1-distill-qwen-1.5b':1.78e9,'gpt-oss-20b':20.91e9};
 const SHORT={'deepseek-r1-0528-qwen3-8b':'R1-0528-Qwen3-8B','r1-distill-qwen-7b':'R1-Qwen-7B','r1-distill-llama-8b':'R1-Llama-8B',
  'r1-distill-qwen-14b':'R1-Qwen-14B','r1-distill-qwen-1.5b':'R1-Qwen-1.5B','nemotron-nano-9b-v2':'Nemotron-Nano-9B','glm-z1-9b':'GLM-Z1-9B',
  'openreasoning-nemotron-7b':'OR-Nemotron-7B','acereason-nemotron-1.1-7b':'AceReason-7B','llama-nemotron-nano-8b-v1':'Nemotron-Nano-8B',
  'phi-4-reasoning':'Phi-4-R-14B','phi-4-mini-reasoning':'Phi-4-mini-R','deepscaler-1.5b':'DeepScaleR-1.5B'};
 const gb=p=>p==null?'':(p/1e9).toFixed(1)+'B';
 const name=m=>{const n=esc(SHORT[m.model_id]||m.label);const repo=m.hf&&String(m.hf).split('@')[0];
  const t=esc((repo||m.label)+(PARAMS[m.model_id]?' · '+gb(PARAMS[m.model_id])+' params':''));
  return (repo?`<a href="https://huggingface.co/${esc(repo)}" target="_blank" rel="noopener" title="${t}">${n}</a>`:n)+(PARAMS[m.model_id]?` <span>${gb(PARAMS[m.model_id])}</span>`:'');};
 let box=document.getElementById('bench-table');
 if(!box){box=document.createElement('section');box.id='bench-table';box.setAttribute('aria-label','Benchmarks');main.parentNode.insertBefore(box,main);}
 function table(d){
  const B=(d.benchmarks||[]).filter(b=>b.enabled).sort((a,b)=>(a.suite===b.suite?0:d.suites.indexOf(a.suite)-d.suites.indexOf(b.suite))||a.order-b.order);
  const R={};for(const r of d.results||[])(R[r.model_id]||(R[r.model_id]={}))[r.benchmark_id]=r;
  const ms=d.models||[],gen=ms.find(m=>m.kind==='genesis');
  const kings=ms.filter(m=>m.kind==='king'&&(R[m.model_id]||m.status==='evaluating')).sort((a,b)=>(b.epoch??0)-(a.epoch??0));
  const full=m=>R[m.model_id]&&B.every(b=>R[m.model_id][b.id])&&PARAMS[m.model_id]!=null;
  const avg=m=>B.reduce((t,b)=>t+R[m.model_id][b.id].score,0)/B.length;
  const cand=ms.filter(m=>(m.kind==='reference'||m.kind==='genesis')&&full(m));
  const refs=cand.filter(m=>m.kind==='reference'&&!cand.some(o=>o!==m&&PARAMS[o.model_id]<=PARAMS[m.model_id]&&avg(o)>avg(m)))
   .sort((a,b)=>PARAMS[b.model_id]-PARAMS[a.model_id]||avg(b)-avg(a));
  const G=gen?R[gen.model_id]||{}:{};
  const all=[...kings,...refs,...(gen?[gen]:[])];
  const AVG={id:'_avg'};const val=(m,b)=>b===AVG?(R[m.model_id]&&B.every(x=>R[m.model_id][x.id])?avg(m):null):((R[m.model_id]||{})[b.id]||{}).score??null;
  const r1=x=>Math.round(1000*x);  // compare at display precision (0.1 pp) so visible ties are all starred
  const span={};for(const b of [...B,AVG]){const v=all.map(m=>val(m,b)).filter(x=>x!=null);if(v.length)span[b.id]=[Math.min(...v),Math.max(...v)];}
  const STAR='<svg class="max-ic" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 .6l1.3 2.8 3 .3-2.3 2 .7 3L5 7.2 2.3 8.7l.7-3-2.3-2 3-.3z"/></svg>';
  // d=(max-v)/(max-min): green to d~.2, amber d~.4-.5, strong red near d~.85+ (Jacob 19:50 CR)
  const shade=f=>{const d=1-f,k=Math.min(1,Math.max(0,(d-.1)/.75)),h=Math.round(130*(1-Math.pow(k,.75)));return`hsl(${h} 65% 45% / ${(.30+.14*Math.abs(.5-k)*2).toFixed(2)})`;};
  const cell=(m,b)=>{const x=val(m,b),c=b===AVG?' avg':'';if(x==null)return`<td class="na${c}">${b===AVG||m.status==='evaluating'||m.status==='queued'?'…':'—'}</td>`;
   const [lo,hi]=span[b.id];const f=hi>lo?(x-lo)/(hi-lo):1;const top=r1(x)===r1(hi);
   return`<td class="${top?'best':''}${c}" style="background:linear-gradient(${shade(f)},${shade(f)}),var(--bg)"${top?' title="column best"':''}>${top?STAR:''}${pct(x)}</td>`;};
  const row=(m,cls='')=>`<tr class="${cls}"><th scope="row">${m.kind==='king'?`E${esc(m.epoch)} <span>${esc(String(m.king).slice(0,8))}</span>`:name(m)}</th>${cell(m,AVG)}${B.map(b=>cell(m,b)).join('')}</tr>`;
  const suites=[...new Set(B.map(b=>b.suite))];
  const head=suites.length>1?`<tr><th></th><th class="avg"></th>${suites.map(s=>`<th colspan="${B.filter(b=>b.suite===s).length}" class="suite">${esc(s)}</th>`).join('')}</tr>`:'';
  const prev=box.querySelector('.bench-scroll'),keep=prev?[prev.scrollTop,prev.scrollLeft]:null;
  const cols=[AVG,...B],lab=b=>b===AVG?'Avg':b.label;
  const hist=kings.filter(m=>R[m.model_id]).sort((a,b)=>(a.crowned_at??a.epoch??0)-(b.crowned_at??b.epoch??0));
  SP=cols.map(b=>({b,lab:lab(b),pts:hist.map(m=>({t:m.crowned_at??m.epoch,v:val(m,b),se:b===AVG?null:((R[m.model_id]||{})[b.id]||{}).se??null,m})).filter(p=>p.v!=null&&p.t!=null)}));
  const sparks=SP.map((s,i)=>{const l=s.pts[s.pts.length-1];
   return`<div class="bench-spark" data-i="${i}"><div class="section-heading"><h2 title="${esc(s.lab)}: score of each scored king, by crowning time (${s.pts.length} kings)">${esc(s.lab)}</h2><span class="reading">${l?pct(l.v)+'%':'—'}</span></div><div class="chart-wrap k1-chart bs-plot" role="img" aria-label="${esc(s.lab)} king score history"></div></div>`;}).join('');
  box.innerHTML=`<div class="bench-sparks" aria-label="King score history per benchmark">${sparks}</div><div class="bench-scroll"><table><thead>${head}<tr><th scope="col">Model</th><th scope="col" class="avg" title="mean of the benchmark columns">Avg</th>${B.map(b=>`<th scope="col">${esc(b.label)}</th>`).join('')}</tr></thead><tbody>`+
   kings.map(m=>row(m)).join('')+`</tbody><tfoot>`+refs.map(m=>row(m)).join('')+(gen?row(gen,'genesis'):'')+`</tfoot></table></div>`;
  layout();
  const sc=box.querySelector('.bench-scroll');if(keep){sc.scrollTop=keep[0];sc.scrollLeft=keep[1];}
 }
 let SP=[];
 const dt=t=>new Date(t*1000).toLocaleDateString('en-US',{month:'short',day:'2-digit'});
 const NS='http://www.w3.org/2000/svg',el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 function spark(card,s){
  const plot=card.querySelector('.bs-plot');plot.textContent='';const ps=s.pts;
  const W=plot.clientWidth||200,H=plot.clientHeight||90,P={l:30,r:6,t:6,b:16};
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});plot.append(svg);
  if(!ps.length){const e=document.createElement('p');e.className='empty';e.textContent='No king scored yet.';plot.append(e);return;}
  const t0=ps[0].t,t1=ps[ps.length-1].t;let lo=Math.min(...ps.map(p=>p.v)),hi=Math.max(...ps.map(p=>p.v));
  const st=hi-lo>.2?.1:hi-lo>.08?.05:hi-lo>.03?.02:.01;lo=Math.max(0,Math.floor(lo/st)*st);hi=Math.min(1,Math.ceil(hi/st)*st);if(hi<=lo)hi=Math.min(1,lo+st);
  const x=t=>t1>t0?P.l+(W-P.l-P.r)*(t-t0)/(t1-t0):(P.l+W-P.r)/2,y=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/(hi-lo));
  for(const g of [lo,(lo+hi)/2,hi]){svg.append(el('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:'k1-grid'}));
   const tx=el('text',{x:P.l-5,y:y(g)+3,'text-anchor':'end'});{const v=100*g;tx.textContent=(Math.abs(v-Math.round(v))<1e-6?v.toFixed(0):v.toFixed(1))+'%';}svg.append(tx);}
  const C='var(--performance)';
  if(ps.length>1)svg.append(el('polyline',{points:ps.map(p=>x(p.t).toFixed(1)+','+y(p.v).toFixed(1)).join(' '),fill:'none',stroke:C,'stroke-width':1.5,'stroke-linejoin':'round'}));
  const r=ps.length>40?1.6:2.2;
  ps.forEach((p,i)=>svg.append(el('circle',{cx:x(p.t),cy:y(p.v),r:i===ps.length-1?r+.6:r,fill:i===ps.length-1?C:'var(--bg)',stroke:C,'stroke-width':1.2})));
  const at=(e,a,xx)=>{const tx=el('text',{x:xx,y:H-3,'text-anchor':a});tx.textContent='E'+e;svg.append(tx);};
  at(ps[0].m.epoch,'start',P.l);if(ps.length>1)at(ps[ps.length-1].m.epoch,'end',W-P.r);
  const hl=el('circle',{r:r+1.6,fill:C,stroke:'var(--bg)','stroke-width':1,'pointer-events':'none',visibility:'hidden'});svg.append(hl);
  let tp=document.createElement('output');tp.className='chart-tip';tp.hidden=true;plot.append(tp);
  const xs=ps.map(p=>x(p.t));
  ps.forEach((p,i)=>{const l=i?(xs[i-1]+xs[i])/2:P.l,rr=i<ps.length-1?(xs[i]+xs[i+1])/2:W-P.r;
   const h=el('rect',{x:l,y:P.t,width:Math.max(1,rr-l),height:H-P.t-P.b,fill:'transparent'});
   h.addEventListener('mouseenter',()=>{hl.setAttribute('cx',xs[i]);hl.setAttribute('cy',y(p.v));hl.setAttribute('visibility','visible');
    tp.innerHTML=`<strong>E${esc(p.m.epoch)} · king ${esc(String(p.m.king).slice(0,8))}</strong><span>${esc(s.lab)}</span> ${pct(p.v)}%${p.se!=null?' ±'+(100*p.se).toFixed(1):''}\n<span>crowned</span> ${esc(new Date(p.t*1000).toLocaleString('en-US',{month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'}))}`;
    tp.hidden=false;const tw=tp.offsetWidth;tp.style.left=Math.min(Math.max(0,xs[i]-tw/2),Math.max(0,W-tw))+'px';tp.style.top=(y(p.v)>H/2?2:H/2)+'px';});
   h.addEventListener('mouseleave',()=>{tp.hidden=true;hl.setAttribute('visibility','hidden');});svg.append(h);});
 }
 // kings box: header + 5 king rows + the sticky reference/Genesis footer
 function fit(){const sc=box.querySelector('.bench-scroll'),t=sc&&sc.querySelector('table');if(!t)return;/* pt1070u: the skeleton box has no table yet */const rs=t.tBodies[0]?t.tBodies[0].rows:[];
  if(rs.length<=5){sc.style.maxHeight='none';return;}
  let h=t.tHead.offsetHeight+(t.tFoot?t.tFoot.offsetHeight:0)+2;for(let i=0;i<5;i++)h+=rs[i].offsetHeight;sc.style.maxHeight=h+'px';}
 function layout(){box.querySelectorAll('.bench-spark').forEach(c=>{const s=SP[+c.dataset.i];if(s)spark(c,s);});fit();}
 let rt=0;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(layout,150);});
 let data=null;
 async function load(){try{const r=await (window.AFX&&AFX.take('/benchmarks.json')||fetch('/benchmarks.json',{cache:'no-store'}));/* pt1070u */if(!r.ok)throw 0;data=await r.json();table(data);}catch(e){}}
 load();setInterval(load,120000);
})();
