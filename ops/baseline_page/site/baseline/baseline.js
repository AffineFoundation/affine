/* affine.io/baseline (Jacob 2026-11-02): shape of the live prior-v1 baseline.
   Data: /baseline/curves.json (site-tools/build_baseline_curves.py): tasks sorted by the head's pass rate, averaged over ~160
   equal-count bins and rounded - no task ids, no per-task values. Two charts share the same x (percentile of tasks by pass rate):
   pass rate (--performance) and per-round selection probability of the Neyman/Hajek draw (--incentive). Chart look as the site's
   k1 tiles / bench cards (b969182): k1-grid gridlines, mono tick labels, line with hollow points, .chart-tip hover. */
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
 let D=null;const charts=[];
 function chart(box,cfg){
  box.textContent='';const n=cfg.v.length;
  const W=box.clientWidth||600,H=box.clientHeight||280,P={l:44,r:10,t:8,b:22};
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});box.append(svg);
  const xc=i=>P.l+(W-P.l-P.r)*((i+.5)/n),y=v=>P.t+(H-P.t-P.b)*(1-(v-cfg.lo)/(cfg.hi-cfg.lo));
  for(const g of cfg.ticks){svg.append(el('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:'k1-grid'}));
   const tx=el('text',{x:P.l-6,y:y(g)+3,'text-anchor':'end'});tx.textContent=cfg.tick(g);svg.append(tx);}
  for(const q of [0,25,50,75,100]){const xx=P.l+(W-P.l-P.r)*q/100;
   const tx=el('text',{x:xx,y:H-5,'text-anchor':q===0?'start':q===100?'end':'middle'});tx.textContent=String(q);svg.append(tx);}
  if(cfg.uniform!=null)svg.append(el('line',{x1:P.l,x2:W-P.r,y1:y(cfg.uniform),y2:y(cfg.uniform),class:'bl-uniform'}));
  if(cfg.mid!=null){const xm=P.l+(W-P.l-P.r)*cfg.mid;svg.append(el('line',{x1:xm,x2:xm,y1:P.t,y2:H-P.b,class:'bl-guide'}));}
  const C=cfg.color;
  svg.append(el('polyline',{points:cfg.v.map((v,i)=>xc(i).toFixed(1)+','+y(v).toFixed(1)).join(' '),fill:'none',stroke:C,'stroke-width':1.5,'stroke-linejoin':'round'}));
  const r=n>40?1.6:2.2;
  cfg.v.forEach((v,i)=>{if(i%2===0||i===n-1)svg.append(el('circle',{cx:xc(i),cy:y(v),r,fill:'var(--bg)',stroke:C,'stroke-width':1.2}));});
  const hl=el('circle',{r:r+1.6,fill:C,stroke:'var(--bg)','stroke-width':1,'pointer-events':'none',visibility:'hidden'});svg.append(hl);
  const vl=el('line',{y1:P.t,y2:H-P.b,class:'bl-guide','pointer-events':'none',visibility:'hidden'});svg.append(vl);
  const tp=document.createElement('output');tp.className='chart-tip';tp.hidden=true;box.append(tp);
  const step=(W-P.l-P.r)/n;
  const c={mark(i,tip){if(i==null){hl.setAttribute('visibility','hidden');vl.setAttribute('visibility','hidden');tp.hidden=true;return;}
    hl.setAttribute('cx',xc(i));hl.setAttribute('cy',y(cfg.v[i]));hl.setAttribute('visibility','visible');
    vl.setAttribute('x1',xc(i));vl.setAttribute('x2',xc(i));vl.setAttribute('visibility','visible');
    if(!tip){tp.hidden=true;return;}
    tp.innerHTML=tipText(i);tp.hidden=false;const tw=tp.offsetWidth;tp.style.left=Math.min(Math.max(0,xc(i)-tw/2),Math.max(0,W-tw))+'px';tp.style.top=(y(cfg.v[i])>H/2?4:H/2)+'px';}};
  const hit=el('rect',{x:P.l,y:P.t,width:W-P.l-P.r,height:H-P.t-P.b,fill:'transparent'});svg.append(hit);
  const at=e=>{const b=svg.getBoundingClientRect(),mx=(e.clientX-b.left)*W/b.width;return Math.max(0,Math.min(n-1,Math.floor((mx-P.l)/step)));};
  hit.addEventListener('mousemove',e=>{const i=at(e);charts.forEach(o=>o.mark(i,o===c));});
  hit.addEventListener('mouseleave',()=>charts.forEach(o=>o.mark(null)));
  return c;
 }
 function tipText(i){const n=D.pass_rate.length,a=Math.round(100*i/n),b=Math.round(100*(i+1)/n);
  return`<strong>Tasks at percentile ${a}${b>a?'&ndash;'+b:''}</strong><span>pass rate</span> &asymp; ${D.pass_rate[i].toFixed(2)}\n<span>selection per round</span> &asymp; ${D.select_pct[i].toFixed(1)}%`;}
 function draw(){
  if(!D)return;charts.length=0;const n=D.pass_rate.length;
  let k=D.pass_rate.findIndex(v=>v>=.5);const mid=k<0?null:k/n;
  charts.push(chart($('bl-pass-c'),{v:D.pass_rate,lo:0,hi:1,ticks:[0,.25,.5,.75,1],tick:g=>g===0||g===1?String(g):g.toFixed(2),color:'var(--performance)',mid}));
  const top=Math.ceil(Math.max(...D.select_pct,D.uniform_pct)+.25),st=top>6?2:1,tk=[];for(let g=0;g<=top+1e-9;g+=st)tk.push(g);
  charts.push(chart($('bl-select-c'),{v:D.select_pct,lo:0,hi:tk[tk.length-1],ticks:tk,tick:g=>g+'%',color:'var(--incentive)',uniform:D.uniform_pct,mid}));
 }
 function fill(){
  if(D.cfg)document.querySelectorAll('[data-cfg]').forEach(e=>{const v=D.cfg[e.dataset.cfg];if(typeof v==='number')e.textContent=e.dataset.cfg==='floor'||e.dataset.cfg==='cap'?v.toFixed(2):String(v);});
  document.querySelectorAll('[data-bl]').forEach(e=>{const v=D[e.dataset.bl];if(v!=null)e.textContent=fmt(v);});
  $('bl-pass-r').textContent='mean '+D.mean_pass.toFixed(2);
  $('bl-select-r').textContent='~'+fmt(D.per_round)+' / '+fmt(D.n_tasks)+' tasks';
  $('bl-stats').textContent=`${D.baseline} · live since E${D.since_epoch} · ${fmt(D.n_tasks)} tasks · mean pass rate ${D.mean_pass.toFixed(2)} · ~${fmt(D.per_round)} drawn per round (uniform would be ${D.uniform_pct.toFixed(1)}% each)`;
  if(D.updated)$('bl-updated').textContent='Updated '+new Date(D.updated.replace('Z',':00Z')).toLocaleString('en-US',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',timeZoneName:'short'});
 }
 fetch('/baseline/curves.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(r.status);return r.json();}).then(d=>{D=d;fill();draw();$('bl-main').dataset.state='ready';})
  .catch(()=>{$('bl-main').dataset.state='unavailable';for(const id of ['bl-pass-c','bl-select-c']){const p=document.createElement('p');p.className='empty';p.textContent='Baseline data unavailable.';$(id).append(p);}});
 let t;window.addEventListener('resize',()=>{clearTimeout(t);t=setTimeout(draw,120);});
 window.addEventListener('affine:theme',draw);
})();
