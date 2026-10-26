/* Chart expand, 2026-10-26: every chart tile gets a corner button that opens the same live chart large.
   The tile itself is lifted into a fixed overlay (a placeholder keeps the grid in place), and the chart
   scripts re-render at the new size on 'affine:chart-resize', so it stays vector-sharp and keeps updating. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const PLOT='.chart-wrap,.heatmap-wrap,.sp-plot';
 const icon='<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M7 1.5h3.5V5M10.5 1.5 7 5M5 10.5H1.5V7M1.5 10.5 5 7"/></svg>';
 const xicon='<svg viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M2.5 2.5l7 7M9.5 2.5l-7 7"/></svg>';
 const backdrop=document.createElement('div');backdrop.className='cx-backdrop';backdrop.hidden=true;document.body.append(backdrop);
 let open=null,holder=null,opener=null,lastFocus=null;
 const title=s=>(s.querySelector('h1,h2')?.textContent||'chart').trim();
 const fire=()=>window.dispatchEvent(new Event('affine:chart-resize'));
 function isChart(s){if(s.hidden||s.id==='flip-share')return false;const p=s.querySelector(':scope>'+PLOT.split(',').join(',:scope>'));return !!p&&!p.matches('#panel-mining .sp-plot,#panel-weights .sp-plot');}
 function wire(){
  for(const s of main.querySelectorAll(':scope>section')){
   if(s.querySelector(':scope>.cx-btn')||!isChart(s))continue;
   const b=document.createElement('button');b.type='button';b.className='cx-btn';b.innerHTML=icon;
   b.setAttribute('aria-label','Expand '+title(s));b.title='Expand';
   b.addEventListener('click',e=>{e.stopPropagation();open===s?close():expand(s,b);});
   s.append(b);
  }
 }
 function expand(s,b){
  if(open)close(true);
  lastFocus=document.activeElement;opener=b;open=s;
  const r=s.getBoundingClientRect();
  holder=document.createElement('div');holder.className='cx-holder';holder.style.height=r.height+'px';holder.setAttribute('aria-hidden','true');
  s.before(holder);
  s.classList.add('cx-open');s.setAttribute('role','dialog');s.setAttribute('aria-modal','true');
  b.innerHTML=xicon;b.setAttribute('aria-label','Close '+title(s));b.title='Close (Esc)';
  backdrop.hidden=false;document.documentElement.classList.add('cx-lock');
  requestAnimationFrame(()=>{fire();b.focus({preventScroll:true});});
 }
 function close(quiet){
  if(!open)return;const s=open,b=opener;open=null;
  s.classList.remove('cx-open');s.removeAttribute('role');s.removeAttribute('aria-modal');
  holder?.remove();holder=null;backdrop.hidden=true;document.documentElement.classList.remove('cx-lock');
  b.innerHTML=icon;b.setAttribute('aria-label','Expand '+title(s));b.title='Expand';
  if(!quiet){requestAnimationFrame(fire);(lastFocus&&lastFocus.focus?lastFocus:b).focus({preventScroll:true});}
 }
 backdrop.addEventListener('click',()=>close());
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&open){e.preventDefault();close();}});
 let t;window.addEventListener('resize',()=>{if(!open)return;clearTimeout(t);t=setTimeout(fire,120);});
 new MutationObserver(wire).observe(main,{childList:true});wire();
})();
