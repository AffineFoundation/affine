/* Load sequence (Jacob pt1070u, tightened pt1076u 2026-11-02). Shapes first, data second, no waiting:
   - the tiles are already in the HTML at their final size (title, legend, static placeholder gridlines: class afx afx-wait);
   - the moment a chart's own first svg is drawn, its placeholder goes and the svg fades in over 200 ms (no delay, no
     stagger, nothing waits for other charts); tables and lists fade the same way;
   - a redraw within the first 200 ms continues that fade; any later redraw (refresh, resize, theme) is instant.
   Only watches the DOM; no chart logic or data changes. prefers-reduced-motion: no fade. */
(()=>{
 const WIN=200,first=new Map();
 const PLOT='.chart-wrap',LIST='.flip-table-wrap,.k1-lineage,.bench-scroll';
 const key=el=>{const host=el.closest('[id]'),i=el.closest('[data-i]');return (host?host.id:'?')+':'+(i?i.dataset.i:'')+':'+(el.matches(PLOT)?'p':'l')+':'+(el.id||'');};
 const phase=el=>{const k=key(el),now=performance.now();if(!first.has(k)){first.set(k,now);return 0;}const d=now-first.get(k);return d<WIN?d:-1;};
 const fade=(el,t)=>{if(t<0)return;el.classList.add('afx-in');el.style.animationDelay=(-t)+'ms';el.addEventListener('animationend',()=>{el.classList.remove('afx-in');el.style.animationDelay='';},{once:true});};
 function onSvg(svg){const w=svg.parentElement;if(!w||!w.matches(PLOT))return;const t=phase(w);if(w.classList.contains('afx-wait'))w.classList.remove('afx-wait','afx');fade(svg,t);}
 function onWrap(w){if(!w.querySelector(':scope>svg')&&!first.has(key(w)))w.classList.add('afx','afx-wait');}
 function scan(n){
  if(n.nodeType!==1)return;
  if(n.matches('svg')){onSvg(n);return;}
  if(n.matches(PLOT))onWrap(n);
  if(n.matches(LIST)&&!n.classList.contains('afx-rows'))fade(n,phase(n));
  for(const w of n.querySelectorAll(PLOT))onWrap(w);
  for(const s of n.querySelectorAll('.chart-wrap>svg'))onSvg(s);
  for(const l of n.querySelectorAll(LIST))if(!l.classList.contains('afx-rows'))fade(l,phase(l));
 }
 new MutationObserver(r=>{for(const x of r)for(const n of x.addedNodes)scan(n);}).observe(document.body,{childList:true,subtree:true});
 setTimeout(()=>{for(const w of document.querySelectorAll('.afx-wait'))w.classList.remove('afx-wait','afx');},15000);
})();
