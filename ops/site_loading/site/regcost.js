/* Registration cost tiles (2026-10-30, Jacob pt809u). Data: /reg-cost.json, published hourly by site-tools/build_reg_cost.py
   from the TaoMarketCap public API (key stays on the server). One square tile in the uniform grid:
   - SN120 UID registration burn (TAO): hourly close line, faint hourly high-low band (burn jumps after registrations, then decays).
   Span buttons in the legend row (7D / 30D / 1Y / All). The y axis turns logarithmic when the visible range spans more than ~2 decades.
   Points are omitted for hours with no change, so the line is drawn as a step that holds the last value up to now. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const NS='http://www.w3.org/2000/svg',fin=n=>typeof n==='number'&&Number.isFinite(n);
 const S=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const af=v=>v>=1000?+(v/1000).toPrecision(3)+'k':String(+v.toPrecision(3));
 const tf=v=>!fin(v)?'—':v>=100?v.toFixed(1):v>=10?v.toFixed(2):v>=1?v.toFixed(3):v>=.01?v.toFixed(4):v.toPrecision(2);
 const dt=(t,y)=>new Date(t*1000).toLocaleDateString('en-GB',{day:'numeric',month:'short',...(y?{year:'2-digit'}:{}),timeZone:'America/Costa_Rica'});
 const dtt=t=>new Date(t*1000).toLocaleString('en-GB',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'America/Costa_Rica'})+' CR';
 const SPANS=[['7D',7],['30D',30],['1Y',365],['All',0]];
 const DEF=[
  {id:'panel-regcost',key:'sn120',title:'SN120 registration cost',color:'--incentive',span:7,band:true,
   tip:'Burn in TAO to register one UID (hotkey) on subnet 120, over time. Line: value at the end of each hour; faint band: the hourly high-low range (the burn steps up after registrations and decays back toward the floor). Source: TaoMarketCap public API (subnet burn chart + current subnet snapshot), refreshed hourly into /reg-cost.json. Times in Costa Rica time.'}];
 let doc=null;
 const tiles=DEF.map(d=>{const s=document.getElementById(d.id)||document.createElement('section');/* pt1070u: adopt the shell in index.html */s.id=d.id;s.className='site-panel k1-tile rc-tile';s.setAttribute('aria-labelledby',d.id+'-t');
  s.innerHTML=`<div class="section-heading"><h2 id="${d.id}-t" title="${esc(d.tip)}">${esc(d.title)}</h2><div class="measurement-reading"><span class="reading">—</span></div></div><div class="chart-wrap k1-chart"></div><div class="legend chart-legend rc-legend"></div>`;
  const lg=s.querySelector('.legend');lg.innerHTML=`<span class="grid-legend-static"><i class="swatch" style="background:var(${d.color})"></i>TAO</span>`+SPANS.map(([l,n])=>`<button type="button" class="rc-span" data-days="${n}" aria-pressed="${n===d.span}">${l}</button>`).join('');
  lg.addEventListener('click',e=>{const b=e.target.closest('.rc-span');if(!b)return;d.span=+b.dataset.days;lg.querySelectorAll('.rc-span').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));draw(d,s);});
  return s;});
 function place(){
  if(tiles.every(t=>t.isConnected))return true;
  const ref=document.getElementById('panel-timing')||document.getElementById('panel-k1-batch');if(!ref)return false;
  ref.after(tiles[0]);return true;
 }
 function series(d){
  const o=doc?.[d.key];if(!o||!Array.isArray(o.points))return null;
  const P=o.points.filter(p=>Array.isArray(p)&&fin(p[0])&&fin(p[1])&&p[1]>0).map(p=>({t:p[0],c:p[1],h:fin(p[2])?Math.max(p[2],p[1]):p[1],l:fin(p[3])&&p[3]>0?Math.min(p[3],p[1]):p[1]}));
  const cur=o.current&&fin(o.current.tao)?o.current:null;
  return {P,cur,o};
 }
 function draw(d,tile){
  const wrap=tile.querySelector('.chart-wrap');wrap.querySelector('svg')?.remove();
  let tp=wrap.querySelector('.chart-tip');if(!tp){tp=document.createElement('output');tp.className='chart-tip';tp.hidden=true;wrap.append(tp);}
  const sr=series(d),rd=tile.querySelector('.reading');
  if(!sr||!sr.P.length){rd.textContent='—';return;}
  const now=Math.max(doc.generated_at||0,sr.P[sr.P.length-1].t);
  const curV=sr.cur?sr.cur.tao:sr.P[sr.P.length-1].c;
  rd.textContent=tf(curV)+' τ';rd.title='Current: '+tf(curV)+' TAO'+(sr.cur&&fin(sr.cur.block)?' at block '+sr.cur.block:'')+(fin(sr.o.min_burn_tao)?' · floor (min burn) '+tf(sr.o.min_burn_tao)+' TAO':'')+' · updated '+dtt(doc.generated_at);
  const t0=d.span?now-d.span*86400:sr.P[0].t;
  // value in force at t0 = last point before t0 (step series)
  let i0=sr.P.findIndex(p=>p.t>=t0);if(i0<0)i0=sr.P.length;
  const vis=sr.P.slice(i0),prev=i0>0?sr.P[i0-1]:null;
  const pts=(prev?[{t:t0,c:prev.c,h:prev.c,l:prev.c,carry:1}]:[]).concat(vis);
  pts.push({t:now,c:curV,h:curV,l:curV,end:1});
  const W=wrap.clientWidth||400,H=wrap.clientHeight||300,Pd={l:44,r:10,t:10,b:22};
  const xa=pts[0].t,xb=now>xa?now:xa+1,x=t=>Pd.l+(W-Pd.l-Pd.r)*(t-xa)/(xb-xa);
  let lo=Infinity,hi=0;for(const p of pts){const L=d.band?p.l:p.c,Hh=d.band?p.h:p.c;if(L<lo)lo=L;if(Hh>hi)hi=Hh;}
  const log=hi/lo>150;
  let y0,y1,ticks=[];
  if(log){y0=Math.pow(10,Math.floor(Math.log10(lo)));y1=Math.pow(10,Math.ceil(Math.log10(hi)));for(let v=y0;v<=y1*1.0001;v*=10)ticks.push(v);}
  else{const span=(hi-lo)||hi||1,raw=span*1.15/4,mag=Math.pow(10,Math.floor(Math.log10(raw))),step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>s>=raw);
   y0=Math.max(0,Math.floor((lo-span*.08)/step)*step);y1=Math.ceil((hi+span*.08)/step)*step;if(y1<=y0)y1=y0+step;for(let v=y0;v<=y1+step*1e-6;v+=step)ticks.push(+v.toPrecision(12));}
  const yv=v=>log?Pd.t+(H-Pd.t-Pd.b)*(1-(Math.log10(v)-Math.log10(y0))/(Math.log10(y1)-Math.log10(y0))):Pd.t+(H-Pd.t-Pd.b)*(1-(v-y0)/(y1-y0));
  const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':d.title+' in TAO over time'});wrap.prepend(svg);
  for(const v of ticks){svg.append(S('line',{x1:Pd.l,x2:W-Pd.r,y1:yv(v),y2:yv(v),class:v===ticks[0]?'k1-axis':'k1-grid'}));const t=S('text',{x:Pd.l-6,y:yv(v)+3,'text-anchor':'end'});t.textContent=af(v);svg.append(t);}
  // per-pixel decimation: one column per px keeps last close, max high, min low
  const cols=[];for(const p of pts){const k=Math.round(x(p.t));const c=cols[cols.length-1];if(c&&c.k===k){c.c=p.c;c.h=Math.max(c.h,p.h);c.l=Math.min(c.l,p.l);c.p=p;}else cols.push({k,c:p.c,h:p.h,l:p.l,p});}
  if(d.band&&cols.length>1){let up='',dn='';cols.forEach((c,j)=>{const nx=j+1<cols.length?cols[j+1].k:c.k;up+=(j?'L':'M')+c.k+' '+yv(c.h).toFixed(1)+'L'+nx+' '+yv(c.h).toFixed(1);dn=`L${nx} ${yv(c.l).toFixed(1)}L${c.k} ${yv(c.l).toFixed(1)}`+dn;});
   svg.append(S('path',{d:up+dn+'Z',fill:`var(${d.color})`,'fill-opacity':.18,stroke:'none'}));}
  let line='';cols.forEach((c,j)=>{const yy=yv(c.c).toFixed(1);line+=(j?`H${c.k}V${yy}`:`M${c.k} ${yy}`);});
  svg.append(S('path',{d:line,fill:'none',stroke:`var(${d.color})`,'stroke-width':1.5,'stroke-linejoin':'round'}));
  const dot=S('circle',{r:3,fill:`var(${d.color})`,cx:x(now),cy:yv(curV)});svg.append(dot);
  const yr=(xb-xa)>200*86400,lab=(t,a,xx)=>{const e=S('text',{x:xx,y:H-6,'text-anchor':a});e.textContent=dt(t,yr);svg.append(e);};
  lab(xa,'start',Pd.l);lab(xb,'end',W-Pd.r);
  const guide=S('line',{y1:Pd.t,y2:H-Pd.b,class:'k1-grid',visibility:'hidden'});svg.append(guide);
  const hit=S('rect',{x:Pd.l,y:Pd.t,width:Math.max(1,W-Pd.l-Pd.r),height:Math.max(1,H-Pd.t-Pd.b),fill:'transparent'});
  const show=ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;let c=cols[0];for(const cc of cols){if(cc.k<=px)c=cc;else break;}
   const p=c.p;guide.setAttribute('x1',c.k);guide.setAttribute('x2',c.k);guide.setAttribute('visibility','visible');dot.setAttribute('cx',c.k);dot.setAttribute('cy',yv(c.c));
   const range=d.band&&(c.h!==c.c||c.l!==c.c)?`\n<span>Hour range ${tf(c.l)}–${tf(c.h)} TAO</span>`:'';
   tp.innerHTML=`<strong>${tf(c.c)} TAO</strong><span>${p.end?'Now · '+dtt(now):p.carry?'In force at '+dtt(p.t):(d.band?'Hour from ':'')+dtt(p.t)}</span>${range}`;
   tp.hidden=false;tp.style.left=Math.max(0,Math.min(c.k-90,W-220))+'px';tp.style.top='4px';};
  hit.addEventListener('pointermove',show);hit.addEventListener('pointerdown',show);
  hit.addEventListener('pointerleave',()=>{tp.hidden=true;guide.setAttribute('visibility','hidden');dot.setAttribute('cx',x(now));dot.setAttribute('cy',yv(curV));});
  svg.append(hit);
 }
 const drawAll=()=>{if(!place())return;DEF.forEach((d,k)=>draw(d,tiles[k]));};
 async function load(){try{const r=await (window.AFX&&AFX.take('/reg-cost.json')||fetch('/reg-cost.json?t='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(20000)}));/* pt1070u */if(r.ok)doc=await r.json();}catch{}drawAll();}
 load();setInterval(load,15*60000);
 new MutationObserver(()=>{if(tiles.some(t=>!t.isConnected)&&place())drawAll();}).observe(main,{childList:true});
 let rt;const re=()=>{clearTimeout(rt);rt=setTimeout(drawAll,150);};window.addEventListener('resize',re);window.addEventListener('affine:chart-resize',re);
})();
