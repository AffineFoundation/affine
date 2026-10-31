/* Performance tile (Jacob pt911u 2026-10-31): pass rate of every king on the full 7,496-task pool and on the private holdout, one chart,
   shared x (king, crowned order) and y; latest value of each line next to the chart. Data: /performance.json. Kept as the first tile. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const NS='http://www.w3.org/2000/svg',svgEl=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const pct=x=>x==null?'—':(100*x).toFixed(1)+'%',se=x=>x==null?'':'±'+(100*x).toFixed(1),n=x=>x==null?'—':Number(x).toLocaleString('en-US');
 const C={full:'var(--guide)',holdout:'var(--performance)'};
 let sec=document.getElementById('panel-performance');
 if(!sec){sec=document.createElement('section');sec.id='panel-performance';sec.className='site-panel k1-tile perf-tile';sec.setAttribute('aria-labelledby','panel-performance-t');
  sec.innerHTML='<div class="section-heading"><h2 id="panel-performance-t">Performance</h2><div class="measurement-reading"><span class="reading" id="panel-performance-r">—</span></div></div>'+
   '<div class="perf-body"><div class="chart-wrap k1-chart perf-chart" role="img" aria-label="Pass rate per king on the full set and the holdout"></div><dl class="perf-latest"></dl></div>'+
   '<p class="perf-note"></p>';}
 const keepFirst=()=>{if(main.firstElementChild!==sec)main.prepend(sec);};
 keepFirst();new MutationObserver(keepFirst).observe(main,{childList:true});
 const wrap=sec.querySelector('.perf-chart'),latest=sec.querySelector('.perf-latest'),note=sec.querySelector('.perf-note');
 function tip(){let t=wrap.querySelector('.chart-tip');if(!t){t=document.createElement('output');t.className='chart-tip';t.hidden=true;wrap.append(t);}return t;}
 function draw(d){
  const ks=(d.kings||[]).slice().sort((a,b)=>a.epoch-b.epoch);
  const first=ks.findIndex(k=>k.status==='scored'||k.status==='evaluating');
  const slots=first<0?[]:ks.slice(first);
  const sc=slots.filter(k=>k.status==='scored');
  const last=s=>{for(let i=sc.length-1;i>=0;i--)if(sc[i][s]&&sc[i][s].pass_rate!=null)return sc[i];return null;};
  const S=d.sets||{},lf=last('full'),lh=last('holdout');
  sec.querySelector('.reading').textContent=lf?pct(lf.full.pass_rate):'—';
  latest.innerHTML=[['full',(S.full&&S.full.name)||'Full set',S.full&&S.full.tasks,lf],['holdout',(S.holdout&&S.holdout.name)||'Holdout',S.holdout&&S.holdout.tasks,lh]].map(([k,name,t,r])=>
   `<div><dt><i class="swatch" style="background:${C[k]}"></i>${esc(name)} <small>${n(t)} tasks</small></dt><dd>${r?pct(r[k].pass_rate):'—'}<small>${r?se(r[k].se):''}</small></dd><dd class="perf-king">${r?'E'+esc(r.epoch)+' · '+esc(String(r.king).slice(0,8)):'pending'}</dd></div>`).join('');
  const ev=d.evaluating&&d.evaluating.king;
  note.innerHTML=`Same fixed sets for every king (the gate scores ~150 head-mixed tasks per round). Holdout: MATH test minus pool near-duplicates, list private, sha256 ${esc(String((S.holdout||{}).sha256||'').slice(0,12))}… · `+
   `${esc((d.method||{}).samples_per_task?Object.entries(d.method.samples_per_task).map(([a,b])=>b+'/task '+a).join(', '):'')} · ±SE · informational, never used for promotions. `+
   (ev?`Evaluating ${esc(String(ev).slice(0,8))}. `:'')+(d.backfill?`History: ${n(d.backfill.scored)} of ${n(d.backfill.recoverable)} kings scored. `:'')+`<a href="/performance.json">performance.json</a>`;
  wrap.querySelector('svg')?.remove();wrap.querySelector('.empty')?.remove();
  const W=wrap.clientWidth||400,H=wrap.clientHeight||200,P={l:40,r:10,t:10,b:22};
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});wrap.prepend(svg);
  if(!sc.length){const e=document.createElement('p');e.className='empty';e.textContent=ev?'First evaluation running.':'No king scored yet.';wrap.append(e);return;}
  const vals=sc.flatMap(k=>['full','holdout'].flatMap(s=>k[s]&&k[s].pass_rate!=null?[k[s].pass_rate-(k[s].se||0),k[s].pass_rate+(k[s].se||0)]:[]));
  let lo=Math.max(0,Math.floor((Math.min(...vals)-.02)*20)/20),hi=Math.min(1,Math.ceil((Math.max(...vals)+.02)*20)/20);if(hi-lo<.1){hi=Math.min(1,lo+.1);}
  const y=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/((hi-lo)||1));const st=(hi-lo)>.3?.1:.05;
  for(let g=lo;g<=hi+1e-9;g+=st){svg.append(svgEl('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:'k1-grid'}));const t=svgEl('text',{x:P.l-6,y:y(g)+3,'text-anchor':'end'});t.textContent=Math.round(g*100)+'%';svg.append(t);}
  // x = crowned epoch (linear), so sparse backfill still spans the history; lines join measured kings only
  const pts=sc,e0=Math.min(...pts.map(k=>k.epoch)),e1=Math.max(...pts.map(k=>k.epoch),e0+1),X=e=>P.l+(W-P.l-P.r)*((e-e0)/((e1-e0)||1));
  for(const s of ['full','holdout']){const g=pts.filter(k=>k[s]&&k[s].pass_rate!=null).map(k=>[X(k.epoch),k[s].pass_rate,k[s].se||0]);
   if(g.length>1)svg.append(svgEl('polyline',{points:g.map(p=>p[0]+','+y(p[1])).join(' '),fill:'none',stroke:C[s],'stroke-width':1.5}));
   for(const p of g){if(p[2])svg.append(svgEl('line',{x1:p[0],x2:p[0],y1:y(p[1]-p[2]),y2:y(p[1]+p[2]),stroke:C[s],'stroke-width':1,opacity:.6}));
    svg.append(svgEl('circle',{cx:p[0],cy:y(p[1]),r:pts.length>40?2:3,fill:'var(--bg)',stroke:C[s],'stroke-width':1.5}));}}
  const tp=tip();
  const xs=pts.map(k=>X(k.epoch));
  pts.forEach((k,i)=>{const l=i?(xs[i-1]+xs[i])/2:P.l,r=i<pts.length-1?(xs[i]+xs[i+1])/2:W-P.r;const h=svgEl('rect',{x:l,y:P.t,width:Math.max(1,r-l),height:H-P.t-P.b,fill:'transparent'});
   h.addEventListener('mouseenter',()=>{tp.innerHTML=`<strong>E${esc(k.epoch)} · king ${esc(String(k.king).slice(0,8))}</strong>`+
    `<span>full</span> ${pct(k.full.pass_rate)} ${se(k.full.se)} (${n(k.full.samples)} samples)\n<span>holdout</span> ${pct(k.holdout.pass_rate)} ${se(k.holdout.se)} (${n(k.holdout.samples)} samples)\n<span>cap-hit</span> ${pct(k.full.cap_rate)} / ${pct(k.holdout.cap_rate)}\n<span>crowned</span> ${esc(k.crowned_at_cr||'')} (${esc(k.era||'')})`;
    tp.hidden=false;tp.style.left=Math.min(Math.max(0,xs[i]-60),W-220)+'px';tp.style.top='4px';});h.addEventListener('mouseleave',()=>tp.hidden=true);svg.append(h);});
  const at=(e,a,x)=>{const t=svgEl('text',{x,y:H-6,'text-anchor':a});t.textContent='E'+e;svg.append(t);};
  at(e0,'start',P.l);if(e1!==e0&&pts.length>1)at(e1,'end',W-P.r);
 }
 let data=null;
 async function load(){try{const r=await fetch('/performance.json',{cache:'no-store'});if(!r.ok)throw 0;data=await r.json();requestAnimationFrame(()=>draw(data));}
  catch(e){sec.querySelector('.reading').textContent='unavailable';}}
 load();setInterval(load,120000);
 let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(()=>data&&draw(data),150);});
})();
