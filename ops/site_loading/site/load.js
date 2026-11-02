/* Load sequence (Jacob pt1070u 2026-11-02). Shapes first, data second:
   - the tiles are already in the HTML at their final size (title, legend, placeholder gridlines + shimmer: class afx afx-wait);
   - when a chart's first svg arrives the placeholder fades out and the data elements draw in once: lines draw left to right,
     bars grow from their axis, points and bands fade (~450 ms). Tables and lists fade in.
   - a redraw inside the first ~900 ms (tile-details/resize settle) continues the same animation instead of restarting it;
     any later redraw (refresh timers, resize, theme) is instant, as before.
   No chart script logic or data is changed; this only watches the DOM. Respects prefers-reduced-motion (CSS). */
(()=>{
 const WIN=900,first=new Map(),reduce=matchMedia('(prefers-reduced-motion:reduce)');
 const PLOT='.chart-wrap',LIST='.flip-table-wrap,.k1-lineage,.bench-scroll';
 const key=el=>{const host=el.closest('[id]');const i=el.closest('[data-i]');const sel=el.matches(PLOT)?'p':'l';return (host?host.id:'?')+':'+(i?i.dataset.i:'')+':'+sel+':'+(el.id||'');};
 // returns the ms elapsed since this element's first data, or -1 when the reveal window is over
 const phase=el=>{const k=key(el),now=performance.now();if(!first.has(k)){first.set(k,now);return 0;}const d=now-first.get(k);return d<WIN?d:-1;};
 const settle=(wrap,ms)=>setTimeout(()=>{wrap.classList.remove('afx');},ms);
 const done=el=>el.addEventListener('animationend',()=>{el.classList.remove('afx-in','afx-dot','afx-draw','afx-grow','afx-down');if(el.dataset.afxPl){el.removeAttribute('pathLength');delete el.dataset.afxPl;}},{once:true});
 function animateSvg(svg,t){
  const lag=-t;// negative delay keeps an in-flight reveal continuous across a redraw
  const axis=svg.querySelector('line.k1-axis');const ay=axis?+axis.getAttribute('y1'):null;
  let bars=0;
  for(const n of svg.querySelectorAll('polyline,path,rect,circle,line,polygon')){
   if(n.matches('line.k1-grid,line.k1-axis'))continue;
   const fill=n.getAttribute('fill');
   if(n.tagName==='rect'&&(fill==='transparent'||fill==='none'&&!n.getAttribute('stroke')))continue;// hover targets
   if(n.getAttribute('visibility')==='hidden'||n.getAttribute('pointer-events')==='none')continue;// hover markers
   let cls='afx-in',delay=0;
   if((n.tagName==='polyline'||n.tagName==='path')&&fill==='none'&&!n.getAttribute('stroke-dasharray')){n.setAttribute('pathLength','1');n.dataset.afxPl='1';cls='afx-draw';}
   else if(n.tagName==='rect'){cls='afx-grow';delay=Math.min(bars++*3,180);if(ay!=null&&+n.getAttribute('y')>=ay-0.5)n.classList.add('afx-down');}
   else if(n.tagName==='circle'){cls='afx-dot';delay=160;}
   n.classList.add(cls);n.style.animationDelay=(delay+lag)+'ms';done(n);
  }
 }
 function onSvg(svg){
  const wrap=svg.parentElement;if(!wrap||!wrap.matches(PLOT))return;
  const t=phase(wrap);
  if(wrap.classList.contains('afx-wait')){wrap.classList.remove('afx-wait');settle(wrap,400);}
  if(t<0||reduce.matches)return;animateSvg(svg,t);
 }
 function onList(el){
  const t=phase(el);if(t<0||reduce.matches)return;
  el.classList.add('afx-in');el.style.animationDelay=(-t)+'ms';done(el);
 }
 // a chart box created by a script before its data (timing.js, regcost.js, bench sparks) gets the placeholder too
 function onWrap(w){if(!w.querySelector(':scope>svg')&&!first.has(key(w))){w.classList.add('afx','afx-wait');}}
 function scan(root){
  if(root.nodeType!==1)return;
  if(root.matches('svg')){onSvg(root);return;}
  if(root.matches(PLOT))onWrap(root);
  if(root.matches(LIST)&&!root.classList.contains('afx-rows'))onList(root);
  for(const w of root.querySelectorAll(PLOT))onWrap(w);
  for(const s of root.querySelectorAll(PLOT.split(',').map(p=>p+'>svg').join(',')))onSvg(s);
  for(const l of root.querySelectorAll(LIST))if(!l.classList.contains('afx-rows'))onList(l);
 }
 new MutationObserver(recs=>{for(const r of recs)for(const n of r.addedNodes)scan(n);}).observe(document.body,{childList:true,subtree:true});
 // never shimmer forever: if a source is down the chart script shows its own empty/unavailable text
 setTimeout(()=>{for(const w of document.querySelectorAll('.afx-wait')){w.classList.remove('afx-wait');settle(w,400);}},15000);
})();
