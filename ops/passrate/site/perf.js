/* Performance tile = pass rate across all envs (Jacob pt1130u 2026-11-02, replaces the affine_math full set / holdout chart of pt911u).
   One point per king (x = crowned epoch): its pass rate over the whole gate pool (affine_math + dapo, 19,305 tasks), measured from
   the prior sampler's complete passes (filled), or until a king has one, the gate's head-baseline draw re-weighted by 1/pi (hollow,
   with a 95% CI bar). Data: /passrate.json (aggregates only). Kept as the first tile. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const NS='http://www.w3.org/2000/svg',svgEl=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const pct=x=>x==null?'—':(100*x).toFixed(1)+'%',n=x=>x==null?'—':Number(x).toLocaleString('en-US');
 const C='var(--performance)',SRC={'prior-pass':'prior, every pool task','gate-ipw':'gate draw, 1/π-weighted'};
 let sec=document.getElementById('panel-performance');
 if(!sec){sec=document.createElement('section');sec.id='panel-performance';sec.className='site-panel k1-tile perf-tile';sec.setAttribute('aria-labelledby','panel-performance-t');
  sec.innerHTML='<div class="section-heading"><h2 id="panel-performance-t">Pass rate, all envs</h2><div class="measurement-reading"><span class="reading" id="panel-performance-r">—</span></div></div>'+
   '<div class="perf-body"><div class="chart-wrap k1-chart perf-chart" role="img" aria-label="Pass rate of each king over the whole gate pool, all envs"></div><dl class="perf-latest"></dl></div>'+
   '<div class="legend chart-legend"><span><i class="swatch" style="background:var(--performance)"></i>Prior, full pool</span><span><i class="swatch" style="border:1px solid var(--performance)"></i>Gate draw, 1/π</span></div><p class="perf-note"></p>';}
 const keepFirst=()=>{if(main.firstElementChild!==sec)main.prepend(sec);};
 keepFirst();new MutationObserver(keepFirst).observe(main,{childList:true});
 const wrap=sec.querySelector('.perf-chart'),latest=sec.querySelector('.perf-latest'),note=sec.querySelector('.perf-note');
 function tip(){let t=wrap.querySelector('.chart-tip');if(!t){t=document.createElement('output');t.className='chart-tip';t.hidden=true;wrap.append(t);}return t;}
 function draw(d){
  const pts=(d.kings||[]).filter(k=>k.pass_rate!=null).sort((a,b)=>a.epoch-b.epoch);
  const L=pts[pts.length-1],P0=d.pool||{};
  sec.querySelector('.reading').textContent=L?pct(L.pass_rate):'—';
  latest.innerHTML=`<div><dt><i class="swatch" style="background:${C}"></i>All envs <small>${n(P0.tasks)} tasks</small></dt><dd>${L?pct(L.pass_rate):'—'}<small>${L?'±'+(196*L.se).toFixed(1):''}</small></dd><dd class="perf-king">${L?'E'+esc(L.epoch)+' · '+esc(String(L.king).slice(0,8)):'pending'}</dd></div>`;
  const envs=P0.envs?Object.entries(P0.envs).map(([e,c])=>n(c)+' '+esc(e)).join(' + '):'';
  note.innerHTML=`Pass rate of each king over the whole gate pool (${envs}), measured from the prior: every pool task, 4 rollouts per pass, complete passes only. `+
   `Hollow: no complete pass yet, so the gate's ~500-task head baseline re-weighted to the pool by 1/π (Hajek), bar = 95% CI. Capped replies count as fails; `+
   `8,192 output tokens before the 12K switch (E567), 12,288 after. `+(d.pending?`Measuring ${esc(String(d.pending).slice(0,8))}. `:'')+
   `Informational, never used for promotions. <a href="/passrate.json">passrate.json</a>`;
  wrap.querySelector('svg')?.remove();wrap.querySelector('.empty')?.remove();
  const W=wrap.clientWidth||400,H=wrap.clientHeight||200,P={l:40,r:10,t:10,b:22};
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});wrap.prepend(svg);
  if(!pts.length){const e=document.createElement('p');e.className='empty';e.textContent='No king measured yet.';wrap.append(e);return;}
  const vals=pts.flatMap(k=>[k.ci95?k.ci95[0]:k.pass_rate,k.ci95?k.ci95[1]:k.pass_rate]);
  let lo=Math.max(0,Math.floor((Math.min(...vals)-.01)*20)/20),hi=Math.min(1,Math.ceil((Math.max(...vals)+.01)*20)/20);if(hi-lo<.1){hi=Math.min(1,lo+.1);}
  const y=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/((hi-lo)||1));const st=(hi-lo)>.3?.1:.05;
  for(let g=lo;g<=hi+1e-9;g+=st){svg.append(svgEl('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:'k1-grid'}));const t=svgEl('text',{x:P.l-6,y:y(g)+3,'text-anchor':'end'});t.textContent=Math.round(g*100)+'%';svg.append(t);}
  const e0=Math.min(...pts.map(k=>k.epoch)),e1=Math.max(...pts.map(k=>k.epoch),e0+1),X=e=>P.l+(W-P.l-P.r)*((e-e0)/((e1-e0)||1));
  const xs=pts.map(k=>X(k.epoch));
  if(pts.length>1)svg.append(svgEl('polyline',{points:pts.map((k,i)=>xs[i]+','+y(k.pass_rate)).join(' '),fill:'none',stroke:C,'stroke-width':1.5}));
  pts.forEach((k,i)=>{const g=k.source!=='prior-pass';
   if(g&&k.ci95)svg.append(svgEl('line',{x1:xs[i],x2:xs[i],y1:y(k.ci95[0]),y2:y(k.ci95[1]),stroke:C,'stroke-width':1,opacity:.6}));
   svg.append(svgEl('circle',{cx:xs[i],cy:y(k.pass_rate),r:3,fill:g?'var(--bg)':C,stroke:C,'stroke-width':1.5}));});
  const tp=tip();
  pts.forEach((k,i)=>{const l=i?(xs[i-1]+xs[i])/2:P.l,r=i<pts.length-1?(xs[i]+xs[i+1])/2:W-P.r;const h=svgEl('rect',{x:l,y:P.t,width:Math.max(1,r-l),height:H-P.t-P.b,fill:'transparent'});
   h.addEventListener('mouseenter',()=>{tp.innerHTML=`<strong>E${esc(k.epoch)} · king ${esc(String(k.king).slice(0,8))}</strong>`+
    `<span>pass</span> ${pct(k.pass_rate)} (95% CI ${pct(k.ci95&&k.ci95[0])}–${pct(k.ci95&&k.ci95[1])})\n<span>source</span> ${esc(SRC[k.source]||k.source)}`+
    (k.source==='prior-pass'?` · ${n(k.passes)} pass${k.passes>1?'es':''}`:` · ${n(k.rounds)} round${k.rounds>1?'s':''}, ${n(k.tasks)} tasks`)+
    `\n<span>rollouts</span> ${n(k.rollouts)} · max ${n(k.max_tokens)} tokens\n<span>crowned</span> ${esc(k.crowned_at_cr||'')}`;
    tp.hidden=false;tp.style.left=Math.min(Math.max(0,xs[i]-60),W-220)+'px';tp.style.top='4px';});h.addEventListener('mouseleave',()=>tp.hidden=true);svg.append(h);});
  const at=(e,a,x)=>{const t=svgEl('text',{x,y:H-6,'text-anchor':a});t.textContent='E'+e;svg.append(t);};
  at(e0,'start',P.l);if(e1!==e0&&pts.length>1)at(e1,'end',W-P.r);
 }
 let data=null;
 async function load(){try{const r=await (window.AFX&&AFX.take('/passrate.json')||fetch('/passrate.json',{cache:'no-store'}));/* first load uses the fetch started in <head> */if(!r.ok)throw 0;data=await r.json();requestAnimationFrame(()=>draw(data));}
  catch(e){sec.querySelector('.reading').textContent='unavailable';}}
 load();setInterval(load,120000);
 let rt;const re=()=>{clearTimeout(rt);rt=setTimeout(()=>data&&draw(data),150);};window.addEventListener('resize',re);
 let lw=0,lh=0;new ResizeObserver(()=>{const w=wrap.clientWidth,h=wrap.clientHeight;if(w!==lw||h!==lh){lw=w;lh=h;re();}}).observe(wrap);
})();
