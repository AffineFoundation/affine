/* Benchmarks (Jacob 2026-11-01): table at the very top of affine.io (benchmarks = columns from the registry, grouped by suite;
   models = rows: kings newest first, then reference models (queued ones as … rows), Genesis at the bottom; each cell shaded relative to its column best (Jacob 2026-11-01 09:08 CR)).
   Table only: the per-benchmark chart tiles were removed (Jacob 2026-11-01 08:35 CR). Data: /benchmarks.json. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const pct=x=>x==null?'—':(100*x).toFixed(1);
 let box=document.getElementById('bench-table');
 if(!box){box=document.createElement('section');box.id='bench-table';box.setAttribute('aria-label','Benchmarks');main.parentNode.insertBefore(box,main);}
 function table(d){
  const B=(d.benchmarks||[]).filter(b=>b.enabled).sort((a,b)=>(a.suite===b.suite?0:d.suites.indexOf(a.suite)-d.suites.indexOf(b.suite))||a.order-b.order);
  const R={};for(const r of d.results||[])(R[r.model_id]||(R[r.model_id]={}))[r.benchmark_id]=r;
  const ms=d.models||[],gen=ms.find(m=>m.kind==='genesis');
  const kings=ms.filter(m=>m.kind==='king'&&(R[m.model_id]||m.status==='evaluating')).sort((a,b)=>(b.epoch??0)-(a.epoch??0));
  const refs=ms.filter(m=>m.kind==='reference'&&(R[m.model_id]||m.status==='evaluating'||m.status==='queued'));
  const G=gen?R[gen.model_id]||{}:{};
  const all=[...kings,...refs,...(gen?[gen]:[])];
  const span={};for(const b of B){const v=all.map(m=>(R[m.model_id]||{})[b.id]).filter(Boolean).map(r=>r.score);if(v.length)span[b.id]=[Math.min(...v),Math.max(...v)];}
  const cell=(m,b)=>{const r=(R[m.model_id]||{})[b.id];if(!r)return`<td class="na">${m.status==='evaluating'||m.status==='queued'?'…':'—'}</td>`;
   const [lo,hi]=span[b.id];const f=hi>lo?(r.score-lo)/(hi-lo):1;const a=Math.round(4+28*f*f);
   return`<td class="${r.score===hi?'best':''}" style="background:color-mix(in srgb,#2fa84f ${a}%,transparent)">${pct(r.score)}</td>`;};
  const row=(m,cls='')=>`<tr class="${cls}"><th scope="row">${m.kind==='king'?`E${esc(m.epoch)} <span>${esc(String(m.king).slice(0,8))}</span>`:esc(m.label)}</th>${B.map(b=>cell(m,b)).join('')}</tr>`;
  const suites=[...new Set(B.map(b=>b.suite))];
  const head=suites.length>1?`<tr><th></th>${suites.map(s=>`<th colspan="${B.filter(b=>b.suite===s).length}" class="suite">${esc(s)}</th>`).join('')}</tr>`:'';
  box.innerHTML=`<div class="bench-scroll"><table><thead>${head}<tr><th scope="col">Model</th>${B.map(b=>`<th scope="col">${esc(b.label)}</th>`).join('')}</tr></thead><tbody>`+
   kings.map((m,i)=>row(m,i===kings.length-1?'last-king':'')).join('')+refs.map(m=>row(m)).join('')+(gen?row(gen,'genesis'):'')+`</tbody></table></div>`;
 }
 let data=null;
 async function load(){try{const r=await fetch('/benchmarks.json',{cache:'no-store'});if(!r.ok)throw 0;data=await r.json();table(data);}catch(e){}}
 load();setInterval(load,120000);
})();
