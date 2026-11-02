/* Benchmarks (Jacob 2026-11-01): table at the very top of affine.io (benchmarks = columns from the registry, grouped by suite;
   models = rows: kings newest first, then reference models (queued ones as … rows), Genesis at the bottom; each cell shaded relative to its column best (Jacob 2026-11-01 09:08 CR)).
   Table only: the per-benchmark chart tiles were removed (Jacob 2026-11-01 08:35 CR). Data: /benchmarks.json.
   References: only the Pareto front over (parameters, lower better) x (mean score over the table's columns, higher better) is shown;
   a ref is hidden if Genesis or another fully scored ref is no larger and scores higher. Kings and Genesis always shown; refs without a
   full set of scores (queued/evaluating/failed) or without a PARAMS entry are hidden (Jacob 2026-11-01 18:13 CR). Short names link to HF.
   Avg column (first score column, Jacob 18:26 CR) = mean over the shown benchmark columns (same mean as the Pareto filter; … when a row is incomplete). Shading: per column,
   min..max of the visible rows, red tint at the bottom -> strong green at the top; a small star marks each column max (ties all marked)
   (Jacob 2026-11-01 18:22 CR). */
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
  const shade=f=>f<.3?`color-mix(in srgb,#d64545 ${Math.round(26*(1-f/.3))}%,transparent)`:`color-mix(in srgb,#2fa84f ${Math.round(6+60*Math.pow((f-.3)/.7,1.4))}%,transparent)`;
  const cell=(m,b)=>{const x=val(m,b),c=b===AVG?' avg':'';if(x==null)return`<td class="na${c}">${b===AVG||m.status==='evaluating'||m.status==='queued'?'…':'—'}</td>`;
   const [lo,hi]=span[b.id];const f=hi>lo?(x-lo)/(hi-lo):1;const top=r1(x)===r1(hi);
   return`<td class="${top?'best':''}${c}" style="background:${shade(f)}"${top?' title="column best"':''}>${top?STAR:''}${pct(x)}</td>`;};
  const row=(m,cls='')=>`<tr class="${cls}"><th scope="row">${m.kind==='king'?`E${esc(m.epoch)} <span>${esc(String(m.king).slice(0,8))}</span>`:name(m)}</th>${cell(m,AVG)}${B.map(b=>cell(m,b)).join('')}</tr>`;
  const suites=[...new Set(B.map(b=>b.suite))];
  const head=suites.length>1?`<tr><th></th><th class="avg"></th>${suites.map(s=>`<th colspan="${B.filter(b=>b.suite===s).length}" class="suite">${esc(s)}</th>`).join('')}</tr>`:'';
  box.innerHTML=`<div class="bench-scroll"><table><thead>${head}<tr><th scope="col">Model</th><th scope="col" class="avg" title="mean of the benchmark columns">Avg</th>${B.map(b=>`<th scope="col">${esc(b.label)}</th>`).join('')}</tr></thead><tbody>`+
   kings.map((m,i)=>row(m,i===kings.length-1?'last-king':'')).join('')+refs.map(m=>row(m)).join('')+(gen?row(gen,'genesis'):'')+`</tbody></table></div>`;
 }
 let data=null;
 async function load(){try{const r=await fetch('/benchmarks.json',{cache:'no-store'});if(!r.ok)throw 0;data=await r.json();table(data);}catch(e){}}
 load();setInterval(load,120000);
})();
