/* Benchmarks (Jacob 2026-11-01): table at the very top of affine.io (benchmarks = columns from the registry, grouped by suite;
   models = rows: kings newest first, then reference models, Genesis at the bottom; each cell shaded against Genesis) and one
   chart tile per benchmark (score of each king over crowned epoch). Data: /benchmarks.json. Minimal: title, number, chart. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const NS='http://www.w3.org/2000/svg',svgEl=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const pct=x=>x==null?'—':(100*x).toFixed(1);
 let box=document.getElementById('bench-table');
 if(!box){box=document.createElement('section');box.id='bench-table';box.setAttribute('aria-label','Benchmarks');main.parentNode.insertBefore(box,main);}
 const tiles={};
 function tile(b){
  let s=tiles[b.id];if(s)return s;
  s=document.createElement('section');s.id='panel-bench-'+b.id;s.className='site-panel k1-tile bench-tile';
  s.innerHTML=`<div class="section-heading"><h2>${esc(b.label)}</h2><div class="measurement-reading"><span class="reading">—</span></div></div><div class="chart-wrap k1-chart bench-chart"></div>`;
  tiles[b.id]=s;return s;}
 function place(){const perf=document.getElementById('panel-performance');let at=perf?perf.nextSibling:main.firstChild;
  for(const s of Object.values(tiles)){if(s.parentNode!==main||s.previousSibling!==(at?at.previousSibling:null)){main.insertBefore(s,at);}at=s.nextSibling;}}
 function table(d){
  const B=(d.benchmarks||[]).filter(b=>b.enabled).sort((a,b)=>(a.suite===b.suite?0:d.suites.indexOf(a.suite)-d.suites.indexOf(b.suite))||a.order-b.order);
  const R={};for(const r of d.results||[])(R[r.model_id]||(R[r.model_id]={}))[r.benchmark_id]=r;
  const ms=d.models||[],gen=ms.find(m=>m.kind==='genesis');
  const kings=ms.filter(m=>m.kind==='king'&&(R[m.model_id]||m.status==='evaluating')).sort((a,b)=>(b.epoch??0)-(a.epoch??0));
  const refs=ms.filter(m=>m.kind==='reference'&&(R[m.model_id]||m.status==='evaluating'));
  const G=gen?R[gen.model_id]||{}:{};
  const cell=(m,b)=>{const r=(R[m.model_id]||{})[b.id];if(!r)return`<td class="na">${m.status==='evaluating'?'…':'—'}</td>`;
   let c='';const g=G[b.id];
   if(m.kind!=='genesis'&&g){const dlt=r.score-g.score,s=Math.hypot(r.se||0,g.se||0);c=dlt>s?'up':dlt<-s?'dn':'eq';}
   return`<td class="${c}">${pct(r.score)}</td>`;};
  const row=(m,cls='')=>`<tr class="${cls}"><th scope="row">${m.kind==='king'?`E${esc(m.epoch)} <span>${esc(String(m.king).slice(0,8))}</span>`:esc(m.label)}</th>${B.map(b=>cell(m,b)).join('')}</tr>`;
  const suites=[...new Set(B.map(b=>b.suite))];
  const head=suites.length>1?`<tr><th></th>${suites.map(s=>`<th colspan="${B.filter(b=>b.suite===s).length}" class="suite">${esc(s)}</th>`).join('')}</tr>`:'';
  box.innerHTML=`<div class="bench-scroll"><table><thead>${head}<tr><th scope="col">Model</th>${B.map(b=>`<th scope="col">${esc(b.label)}</th>`).join('')}</tr></thead><tbody>`+
   kings.map((m,i)=>row(m,i===kings.length-1?'last-king':'')).join('')+refs.map(m=>row(m)).join('')+(gen?row(gen,'genesis'):'')+`</tbody></table></div>`;
 }
 function charts(d){
  const B=(d.benchmarks||[]).filter(b=>b.enabled&&!/^aime2\d$/.test(b.id)).sort((a,b)=>a.order-b.order);
  const R={};for(const r of d.results||[])(R[r.benchmark_id]||(R[r.benchmark_id]={}))[r.model_id]=r;
  const ks=(d.models||[]).filter(m=>m.kind==='king');
  for(const b of B){const s=tile(b);const pts=ks.map(m=>[m,(R[b.id]||{})[m.model_id]]).filter(x=>x[1]).sort((a,b)=>(a[0].epoch??0)-(b[0].epoch??0));
   const latest=pts.length?pts[pts.length-1][1]:null;s.querySelector('.reading').textContent=latest?pct(latest.score)+'%':'—';
   const wrap=s.querySelector('.bench-chart');wrap.querySelector('svg')?.remove();
   const W=wrap.clientWidth||300,H=wrap.clientHeight||160,P={l:34,r:8,t:8,b:18};const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});wrap.prepend(svg);
   if(!pts.length)continue;
   const v=pts.map(p=>p[1].score);let lo=Math.max(0,Math.floor((Math.min(...v)-.02)*20)/20),hi=Math.min(1,Math.ceil((Math.max(...v)+.02)*20)/20);if(hi-lo<.1)lo=Math.max(0,hi-.1);
   const y=x=>P.t+(H-P.t-P.b)*(1-(x-lo)/((hi-lo)||1)),st=(hi-lo)>.3?.1:.05;
   for(let g=lo;g<=hi+1e-9;g+=st){svg.append(svgEl('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:'k1-grid'}));const t=svgEl('text',{x:P.l-5,y:y(g)+3,'text-anchor':'end'});t.textContent=Math.round(g*100);svg.append(t);}
   const e0=Math.min(...pts.map(p=>p[0].epoch)),e1=Math.max(...pts.map(p=>p[0].epoch),e0+1),X=e=>P.l+(W-P.l-P.r)*((e-e0)/((e1-e0)||1));
   if(pts.length>1)svg.append(svgEl('polyline',{points:pts.map(p=>X(p[0].epoch)+','+y(p[1].score)).join(' '),fill:'none',stroke:'var(--performance)','stroke-width':1.5}));
   for(const p of pts)svg.append(svgEl('circle',{cx:X(p[0].epoch),cy:y(p[1].score),r:pts.length>40?1.8:2.6,fill:'var(--bg)',stroke:'var(--performance)','stroke-width':1.4}));
   const at=(e,a,x)=>{const t=svgEl('text',{x,y:H-5,'text-anchor':a});t.textContent='E'+e;svg.append(t);};at(e0,'start',P.l);if(e1!==e0&&pts.length>1)at(e1,'end',W-P.r);}
  place();
 }
 let data=null;
 async function load(){try{const r=await fetch('/benchmarks.json',{cache:'no-store'});if(!r.ok)throw 0;data=await r.json();table(data);requestAnimationFrame(()=>charts(data));}catch(e){}}
 load();setInterval(load,120000);
 new MutationObserver(()=>{if(data&&Object.values(tiles).some(s=>s.parentNode!==main))place();}).observe(main,{childList:true});
 let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(()=>data&&charts(data),150);});
})();
