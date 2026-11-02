/* Epoch duration breakdown (2026-10-29, Jacob t780u). One square tile, last 20 finalized epochs, total minutes stacked by phase.
   Phases come only from each epoch's recorded timestamps (/site-metrics.json epochs + /gate.json post-probe end), in order:
   open → seal → train start → train end → candidate probe end → finalized. A phase that took no time draws nothing, and a phase
   with no time in any shown epoch is left out of the legend, so the chart follows the lifecycle (e.g. immediate seal) on its own.
   The head probe is not stacked: it is either reused or runs in parallel; the tooltip reports it. Outliers above the axis cap
   (e.g. the E214 stall) are clipped with a break mark and labelled with their real total. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const NS='http://www.w3.org/2000/svg',fin=n=>typeof n==='number'&&Number.isFinite(n);
 const S=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const hm=t=>fin(t)?new Date(t*1000).toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',timeZone:'America/Costa_Rica',hour12:false}):'—';
 const mfmt=v=>v<10?v.toFixed(1):String(Math.round(v));
 const PH=[['Open → seal','--incentive','epoch opened (previous epoch finalized) until the turn batch was sealed; under immediate seal this is the wait for the first ready K=1 batch'],['Prepare','--unchecked','seal until training started (an operator stall shows here)'],
  ['Train','--performance','training run'],['Wait for probe','--muted','training finished until the gate started probing the candidate (probe pod busy, e.g. the new head\'s re-probe)'],['Candidate probe','--accepted','gate probe of the candidate'],['Finalize','--axis','decision and publication']];
 let sm=null,gate=null;
 /* pt1070u: adopt the tile shell already in index.html (same markup, legend prefilled so the plot keeps its size) */
 const tile=document.getElementById('panel-timing')||(()=>{const s=document.createElement('section');s.id='panel-timing';s.className='site-panel k1-tile';s.setAttribute('aria-labelledby','panel-timing-t');
  s.innerHTML='<div class="section-heading"><h2 id="panel-timing-t" title="Wall-clock minutes per epoch, start to finalized, stacked by the phases recorded in site-metrics.json and gate.json. Only phases that took time are drawn. Head probe (reused or run in parallel with training) is in the tooltip.">Epoch duration</h2><div class="measurement-reading"><span class="reading">—</span></div></div><div class="chart-wrap k1-chart"></div><div class="legend chart-legend"></div>';return s;})();
 function place(){if(tile.isConnected)return true;const ref=document.getElementById('panel-k1-batch');if(!ref)return false;ref.after(tile);return true;}
 function rows(){
  const g={};for(const e of (gate?.entries||[]))if(fin(e.epoch))g[e.epoch]=e;
  let lastFin=0;const adj=(sm?.epochs||[]).slice().sort((x,y)=>x.index-y.index).map(e=>{const o=Object.assign({},e);if(fin(o.start)&&lastFin>o.start)o.start=lastFin;if(fin(e.finalized_at))lastFin=Math.max(lastFin,e.finalized_at);return o;});/* v829v pre-open: a turn starts when the previous one finalized */
  return adj.filter(e=>fin(e.index)&&fin(e.start)&&fin(e.finalized_at)&&e.finalized_at>e.start).slice(-20).map(e=>{
   const ge=g[e.index],post=ge?.timestamps?.post_probe,pre=ge?.timestamps?.pre_probe;
   const marks=[fin(e.sealed_at)?e.sealed_at:e.deadline,e.training_started_at,e.training_completed_at,post?.start,post?.end,e.finalized_at];
   let prev=e.start;const seg=marks.map(m=>{const v=fin(m)?Math.min(e.finalized_at,Math.max(prev,m)):prev;const d=(v-prev)/60;prev=v;return d;});
   let head=null;if(pre){head=pre.reused?'Head probe: reused ('+String(pre.checkpoint||ge.head||'').slice(0,8)+')':fin(pre.started_at)&&fin(pre.finished_at)?'Head probe: '+mfmt((pre.finished_at-pre.started_at)/60)+' min, '+hm(pre.started_at)+'–'+hm(pre.finished_at)+' CR (parallel)':null;}
   const early=fin(e.sealed_at)&&fin(e.deadline)&&e.deadline-e.sealed_at>30;
   return {i:e.index,early,seg,total:(e.finalized_at-e.start)/60,start:e.start,end:e.finalized_at,verdict:ge?.verdict,head};});
 }
 function draw(){
  if(!place())return;const wrap=tile.querySelector('.chart-wrap'),R=rows();wrap.querySelector('svg')?.remove();
  let tp=wrap.querySelector('.chart-tip');if(!tp){tp=document.createElement('output');tp.className='chart-tip';tp.hidden=true;wrap.append(tp);}
  if(!R.length){tile.querySelector('.reading').textContent='—';return;}
  const last=R[R.length-1],l5=R.slice(-5).map(r=>r.total).sort((a,b)=>a-b),med5=l5[Math.floor(l5.length/2)];
  const rd=tile.querySelector('.reading');rd.textContent='E'+last.i+' '+Math.round(last.total)+'m · med5 '+Math.round(med5)+'m';rd.title='Latest finalized epoch E'+last.i+': '+Math.round(last.total)+' min; median of the last '+l5.length+' finalized epochs: '+Math.round(med5)+' min';
  const W=wrap.clientWidth||400,H=wrap.clientHeight||300,P={l:40,r:10,t:14,b:22};
  const svg=S('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Epoch duration by phase'});wrap.prepend(svg);
  const tot=R.map(r=>r.total).sort((a,b)=>a-b),med=tot[Math.floor(tot.length/2)];
  const normal=tot.filter(v=>v<=Math.max(3*med,med+30));let cap=Math.max(...normal)*1.15;
  const step=cap>240?60:cap>120?30:cap>60?15:10;cap=Math.ceil(cap/step)*step;
  const y=v=>P.t+(H-P.t-P.b)*(1-Math.min(v,cap)/cap);
  for(let v=0;v<=cap+1e-9;v+=step*(cap/step>6?2:1)){svg.append(S('line',{x1:P.l,x2:W-P.r,y1:y(v),y2:y(v),class:v===0?'k1-axis':'k1-grid'}));const t=S('text',{x:P.l-6,y:y(v)+3,'text-anchor':'end'});t.textContent=v+'m';svg.append(t);}
  const n=R.length,bw=(W-P.l-P.r)/n,used=PH.map((_,k)=>R.some(r=>r.seg[k]>=.05));
  R.forEach((r,i)=>{const x=P.l+i*bw+bw*.15,w=Math.max(2,bw*.7);let base=0;
   r.seg.forEach((d,k)=>{if(d<.05||base>=cap){base+=d;return;}const y0=y(base),y1=y(base+d);svg.append(S('rect',{x,y:y1,width:w,height:Math.max(.5,y0-y1),fill:`var(${PH[k][1]})`}));base+=d;});
   if(r.total>cap){const yb=y(cap)+6;svg.append(S('path',{d:`M${x-1} ${yb+3} L${x+w/2} ${yb-1} L${x+w+1} ${yb+3}`,stroke:'var(--bg)','stroke-width':2.5,fill:'none'}));
    const t=S('text',{x:Math.min(Math.max(x+w/2,P.l+14),W-P.r-14),y:P.t-3,'text-anchor':'middle'});t.textContent=Math.round(r.total)+'m';t.style.fill='var(--ink)';svg.append(t);}
   const hit=S('rect',{x:P.l+i*bw,y:P.t,width:bw,height:H-P.t-P.b,fill:'transparent'});
   hit.addEventListener('mouseenter',()=>{tp.innerHTML=`<strong>E${r.i} · ${Math.round(r.total)} min${r.verdict?' · '+(r.verdict==='promoted'?'GREEN':'RED'):''}</strong>`+
     PH.map((p,k)=>r.seg[k]>=.05?`\n<span>${esc(p[0])}</span> ${mfmt(r.seg[k])} min${k===0&&r.early?' (immediate seal at first ready batch)':''}`:'').join('')+(r.head?`\n${esc(r.head)}`:'')+`\n<span>${hm(r.start)}–${hm(r.end)} CR</span>`;
    tp.hidden=false;tp.style.left=Math.max(0,Math.min(P.l+i*bw-60,W-230))+'px';tp.style.top='4px';});
   hit.addEventListener('mouseleave',()=>{tp.hidden=true;});svg.append(hit);});
  const lab=(i,a,x)=>{const t=S('text',{x,y:H-6,'text-anchor':a});t.textContent='E'+R[i].i;svg.append(t);};
  lab(0,'start',P.l);if(n>1)lab(n-1,'end',P.l+n*bw);
  const lg=tile.querySelector('.legend');lg.innerHTML=PH.map((p,k)=>used[k]?`<span title="${esc(p[2])}"><i class="swatch" style="background:var(${p[1]})"></i>${esc(p[0])}</span>`:'').join('');
 }
 async function j(u){const r=await (window.AFX&&AFX.take(u)||fetch(u+'?t='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(15000)}));/* pt1070u */if(!r.ok)throw Error(u);return r.json();}
 /* pt1076u: the tile only shows the last 20 finalized epochs; the -recent slices (site-tools/build_site_slices.py) carry exactly what it reads, ~11 KB gzipped instead of ~205 KB. Full files are the fallback. */
 const jr=(a,b)=>j(a).catch(()=>j(b));
 async function load(){try{[sm,gate]=await Promise.all([jr('/site-metrics-recent.json','/site-metrics.json'),jr('/gate-recent.json','/gate.json')]);}catch{}draw();}
 load();setInterval(load,60000);
 new MutationObserver(()=>{if(!tile.isConnected&&place())draw();}).observe(main,{childList:true});
 let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(draw,250);});
})();
