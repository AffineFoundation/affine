/* affine.io/baseline (Jacob 2026-11-02): shape of the live prior-v1 baseline.
   Data: /baseline/curves.json (site-tools/build_baseline_curves.py): tasks sorted by the head's pass rate, averaged over ~160
   equal-count bins and rounded - no task ids, no per-task values. Two charts share the same x (percentile of tasks by pass rate):
   pass rate (--performance) and per-round selection probability of the Neyman/Hajek draw (--incentive). Chart look as the site's
   k1 tiles / bench cards (b969182): k1-grid gridlines, mono tick labels, line with hollow points, .chart-tip hover.
   pt1080u: dashed --accepted overlays from the static /baseline/irt-curves.json (site-tools/build_baseline_irt_curves.py, a 2PL
   item-response fit, PROPOSED, NOT LIVE): IRT selection probability over the live one, a^2 P(1-P) vs E[p(1-p)] (both / mean) in
   their own tile, and the binned median discrimination a (relative to its median, right axis) on the pass-rate chart. */
(()=>{
 const root=document.documentElement,button=document.getElementById('theme-toggle');
 function sync(){const light=root.dataset.theme==='light';button.setAttribute('aria-pressed',String(light));button.setAttribute('aria-label',light?'Switch to dark mode':'Switch to light mode');button.title=button.getAttribute('aria-label');document.querySelector('meta[name="theme-color"]').content=light?'#ffffff':'#000000';}
 function apply(theme){if(theme==='light')root.dataset.theme='light';else delete root.dataset.theme;sync();window.dispatchEvent(new Event('affine:theme'));}
 button.addEventListener('click',()=>{const theme=root.dataset.theme==='light'?'dark':'light';try{localStorage.setItem('affine:theme',theme);}catch{}apply(theme);});
 window.addEventListener('storage',e=>{if(e.key==='affine:theme')apply(e.newValue);});sync();
})();
(()=>{
 const NS='http://www.w3.org/2000/svg',el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const $=id=>document.getElementById(id),fmt=n=>Number(n).toLocaleString('en-US');
 let D=null,R=null;const charts=[];
 function chart(box,cfg){
  box.textContent='';const n=cfg.v.length;
  const ov=(cfg.over||[]).filter(o=>o&&o.v&&o.v.length),rax=ov.find(o=>o.ticks);
  const W=box.clientWidth||600,H=box.clientHeight||280,P={l:44,r:rax?40:10,t:8,b:22};
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});box.append(svg);
  const xc=i=>P.l+(W-P.l-P.r)*((i+.5)/n),y=v=>P.t+(H-P.t-P.b)*(1-(v-cfg.lo)/(cfg.hi-cfg.lo));
  for(const g of cfg.ticks){svg.append(el('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:'k1-grid'}));
   const tx=el('text',{x:P.l-6,y:y(g)+3,'text-anchor':'end'});tx.textContent=cfg.tick(g);svg.append(tx);}
  for(const q of [0,25,50,75,100]){const xx=P.l+(W-P.l-P.r)*q/100;
   const tx=el('text',{x:xx,y:H-5,'text-anchor':q===0?'start':q===100?'end':'middle'});tx.textContent=String(q);svg.append(tx);}
  if(cfg.uniform!=null)svg.append(el('line',{x1:P.l,x2:W-P.r,y1:y(cfg.uniform),y2:y(cfg.uniform),class:'bl-uniform'}));
  if(cfg.mid!=null){const xm=P.l+(W-P.l-P.r)*cfg.mid;svg.append(el('line',{x1:xm,x2:xm,y1:P.t,y2:H-P.b,class:'bl-guide'}));}
  if(rax)for(const g of rax.ticks){const yy=P.t+(H-P.t-P.b)*(1-(g-rax.lo)/(rax.hi-rax.lo));const tx=el('text',{x:W-P.r+6,y:yy+3,'text-anchor':'start',class:'bl-rtick'});tx.textContent=rax.tick(g);svg.append(tx);}
  const C=cfg.color;
  svg.append(el('polyline',{points:cfg.v.map((v,i)=>xc(i).toFixed(1)+','+y(v).toFixed(1)).join(' '),fill:'none',stroke:C,'stroke-width':1.5,'stroke-linejoin':'round'}));
  for(const o of ov){const m=o.v.length,lo=o.lo!=null?o.lo:cfg.lo,hi=o.hi!=null?o.hi:cfg.hi,xo=j=>P.l+(W-P.l-P.r)*((j+.5)/m),yo=v=>P.t+(H-P.t-P.b)*(1-(Math.min(Math.max(v,lo),hi)-lo)/(hi-lo));
   svg.append(el('polyline',{points:o.v.map((v,j)=>xo(j).toFixed(1)+','+yo(v).toFixed(1)).join(' '),fill:'none',stroke:o.color,'stroke-width':o.width||1.5,'stroke-dasharray':o.dash||'5 3','stroke-linejoin':'round',class:'bl-over'}));}
  const r=n>40?1.6:2.2;
  cfg.v.forEach((v,i)=>{if(i%2===0||i===n-1)svg.append(el('circle',{cx:xc(i),cy:y(v),r,fill:'var(--bg)',stroke:C,'stroke-width':1.2}));});
  const hl=el('circle',{r:r+1.6,fill:C,stroke:'var(--bg)','stroke-width':1,'pointer-events':'none',visibility:'hidden'});svg.append(hl);
  const vl=el('line',{y1:P.t,y2:H-P.b,class:'bl-guide','pointer-events':'none',visibility:'hidden'});svg.append(vl);
  const tp=document.createElement('output');tp.className='chart-tip';tp.hidden=true;box.append(tp);
  const step=(W-P.l-P.r)/n;
  const c={mark(i,tip){if(i!=null&&i>n-1)i=n-1;if(i==null){hl.setAttribute('visibility','hidden');vl.setAttribute('visibility','hidden');tp.hidden=true;return;}
    hl.setAttribute('cx',xc(i));hl.setAttribute('cy',y(cfg.v[i]));hl.setAttribute('visibility','visible');
    vl.setAttribute('x1',xc(i));vl.setAttribute('x2',xc(i));vl.setAttribute('visibility','visible');
    if(!tip){tp.hidden=true;return;}
    tp.innerHTML=tipText(i);tp.hidden=false;const tw=tp.offsetWidth;tp.style.left=Math.min(Math.max(0,xc(i)-tw/2),Math.max(0,W-tw))+'px';tp.style.top=(y(cfg.v[i])>H/2?4:H/2)+'px';}};
  const hit=el('rect',{x:P.l,y:P.t,width:W-P.l-P.r,height:H-P.t-P.b,fill:'transparent'});svg.append(hit);
  const at=e=>{const b=svg.getBoundingClientRect(),mx=(e.clientX-b.left)*W/b.width;return Math.max(0,Math.min(n-1,Math.floor((mx-P.l)/step)));};
  hit.addEventListener('mousemove',e=>{const i=at(e)/n;charts.forEach(o=>o.mark(o.idx(i),o===c));});
  hit.addEventListener('mouseleave',()=>charts.forEach(o=>o.mark(null)));
  c.idx=f=>Math.min(n-1,Math.floor(f*n+1e-9));c.n=n;
  return c;
 }
 const rj=(k,i,n)=>{const v=R&&R[k];if(!v||!v.length)return null;return v[Math.min(v.length-1,Math.floor(i*v.length/n))];};
 function tipText(i){const n=D?D.pass_rate.length:R.info_irt.length,a=Math.round(100*i/n),b=Math.round(100*(i+1)/n);
  let t=`<strong>Tasks at percentile ${a}${b>a?'&ndash;'+b:''}</strong>`;
  if(D)t+=`<span>pass rate</span> &asymp; ${D.pass_rate[i].toFixed(2)}\n<span>selection per round</span> &asymp; ${D.select_pct[i].toFixed(1)}%`;
  const s=rj('select_pct_irt',i,n),ii=rj('info_irt',i,n),ic=rj('info_current',i,n),ar=rj('a_rel_median',i,n);
  if(s!=null)t+=`\n<span>IRT selection (proposed, not live)</span> &asymp; ${s.toFixed(1)}%\n<span>a&sup2;P(1&minus;P) / E[p(1&minus;p)], each / mean</span> &asymp; ${ii.toFixed(1)} / ${ic.toFixed(1)}\n<span>discrimination a / median</span> &asymp; ${ar.toFixed(1)} (model-dependent)`;
  return t;}
 function draw(){
  charts.length=0;
  if(R&&R.info_irt){const m=R.info_irt.length,top=Math.ceil(Math.max(...R.info_irt,...R.info_current)+.25),st=top>6?2:1,tk=[];for(let g=0;g<=top+1e-9;g+=st)tk.push(g);
   let k=D?D.pass_rate.findIndex(v=>v>=.5):-1;
   charts.push(chart($('bl-info-c'),{v:R.info_current,lo:0,hi:tk[tk.length-1],ticks:tk,tick:g=>String(g),color:'var(--incentive)',mid:k<0?null:k/D.pass_rate.length,uniform:1,
    over:[{v:R.info_irt,color:'var(--accepted)'}]}));$('bl-info-c').classList.remove('afx-wait');}
  if(!D)return;const n=D.pass_rate.length;
  let k=D.pass_rate.findIndex(v=>v>=.5);const mid=k<0?null:k/n;
  const ar=R&&R.a_rel_median,atop=ar?Math.max(2,Math.ceil(Math.max(...ar)*2)/2):2;
  charts.unshift(chart($('bl-pass-c'),{v:D.pass_rate,lo:0,hi:1,ticks:[0,.25,.5,.75,1],tick:g=>g===0||g===1?String(g):g.toFixed(2),color:'var(--performance)',mid,
   over:ar?[{v:ar,color:'var(--accepted)',dash:'2 3',width:1.3,lo:0,hi:atop,ticks:[0,atop/2,atop],tick:g=>g+'\u00d7'}]:[]}));
  const si=R&&R.select_pct_irt,top=Math.ceil(Math.max(...D.select_pct,D.uniform_pct,...(si||[0]))+.25),st=top>6?2:1,tk=[];for(let g=0;g<=top+1e-9;g+=st)tk.push(g);
  charts.splice(1,0,chart($('bl-select-c'),{v:D.select_pct,lo:0,hi:tk[tk.length-1],ticks:tk,tick:g=>g+'%',color:'var(--incentive)',uniform:D.uniform_pct,mid,
   over:si?[{v:si,color:'var(--accepted)'}]:[]}));
 }
 function fill(){
  if(D.cfg)document.querySelectorAll('[data-cfg]').forEach(e=>{const v=D.cfg[e.dataset.cfg];if(typeof v==='number')e.textContent=e.dataset.cfg==='floor'||e.dataset.cfg==='cap'?v.toFixed(2):String(v);});
  document.querySelectorAll('[data-bl]').forEach(e=>{const v=D[e.dataset.bl];if(v!=null)e.textContent=fmt(v);});
  $('bl-pass-r').textContent='mean '+D.mean_pass.toFixed(2);
  $('bl-select-r').textContent='~'+fmt(D.per_round)+' / '+fmt(D.n_tasks)+' tasks';
  $('bl-stats').textContent=`${D.baseline} · live since E${D.since_epoch} · ${fmt(D.n_tasks)} tasks · mean pass rate ${D.mean_pass.toFixed(2)} · ~${fmt(D.per_round)} drawn per round (uniform would be ${D.uniform_pct.toFixed(1)}% each)`;
  if(D.updated)$('bl-updated').textContent='Updated '+new Date(D.updated.replace('Z',':00Z')).toLocaleString('en-US',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',timeZoneName:'short'});
 }
 (window.AFX&&AFX.take('/baseline/irt-curves.json')||fetch('/baseline/irt-curves.json',{cache:'no-store'})).then(r=>{if(!r.ok)throw new Error(r.status);return r.json();}).then(d=>{R=d;
   if(R.generated)document.querySelectorAll('[data-irt="generated"]').forEach(e=>{e.textContent=new Date(R.generated.replace('Z',':00Z')).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'});});
   draw();})
  .catch(()=>{document.querySelectorAll('.bl-irt-only').forEach(e=>{e.hidden=true;});const p=document.createElement('p');p.className='empty';p.textContent='IRT overlay unavailable.';const b=$('bl-info-c');if(b){b.classList.remove('afx-wait');b.append(p);}});
 (window.AFX&&AFX.take('/baseline/curves.json')||fetch('/baseline/curves.json',{cache:'no-store'})).then(r=>{if(!r.ok)throw new Error(r.status);return r.json();}).then(d=>{D=d;fill();draw();$('bl-main').dataset.state='ready';})
  .catch(()=>{$('bl-main').dataset.state='unavailable';for(const id of ['bl-pass-c','bl-select-c']){const p=document.createElement('p');p.className='empty';p.textContent='Baseline data unavailable.';$(id).append(p);}});
 let t;window.addEventListener('resize',()=>{clearTimeout(t);t=setTimeout(draw,120);});
 window.addEventListener('affine:theme',draw);
})();
