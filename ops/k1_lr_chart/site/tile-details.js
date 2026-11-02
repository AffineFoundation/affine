/* Tile details (Jacob 2026-10-31, RA/20261031-perf-tile-simplify): every chart tile keeps only its heading and chart.
   Everything else in the tile (legend, stat blocks, notes) is moved into one .tile-details overlay, shown on hover
   or tap. Nodes are moved, not copied, so the chart scripts keep updating them through their own references. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const LEG=new Set(['panel-timing','panel-performance','panel-k1-pass','panel-k1-z','panel-k1-lr']);/* t994u: simple legend stays visible on these tiles */
 const KEEP=el=>el.matches('.section-heading,.chart-wrap,.heatmap-wrap,.sp-plot,.cx-btn,.tile-details,.perf-body')||(el.matches('.legend')&&LEG.has(el.parentNode&&el.parentNode.id));
 const touch=matchMedia('(hover:none)');
 let rt;const redraw=()=>{clearTimeout(rt);rt=setTimeout(()=>{window.dispatchEvent(new Event('resize'));window.dispatchEvent(new Event('affine:chart-resize'));},60);};
 const ro=new ResizeObserver(redraw);
 const isChart=s=>!!s.querySelector(':scope>.chart-wrap,:scope>.perf-body>.chart-wrap');
 function tidy(s){
  if(!isChart(s))return;
  let d=s.querySelector(':scope>.tile-details');
  const extra=[...s.children].filter(c=>!KEEP(c));
  const pl=s.querySelector(':scope>.perf-body>.perf-latest');if(pl)extra.unshift(pl);
  if(!extra.length&&d)return;
  if(!d){d=document.createElement('div');d.className='tile-details';s.classList.add('has-details');
   const cx=s.querySelector(':scope>.cx-btn');cx?s.insertBefore(d,cx):s.append(d);
   const cw=s.querySelector('.chart-wrap');if(cw)ro.observe(cw);
   if(!s.dataset.tdObs){s.dataset.tdObs='1';new MutationObserver(()=>tidy(s)).observe(s,{childList:true});}}
  if(!extra.length)return;
  for(const el of extra)d.append(el);

  redraw();
 }
 const all=()=>{for(const s of main.querySelectorAll(':scope>section'))tidy(s);};
 new MutationObserver(all).observe(main,{childList:true});all();
 /* pt932u: tap/click panel removed */

})();
