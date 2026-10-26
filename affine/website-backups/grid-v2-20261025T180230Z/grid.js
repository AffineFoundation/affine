/* Uniform chart grid, 2026-10-25: give the rollout intake chart its own tile. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 function split(){
  const ro=document.getElementById('panel-rollouts');if(!ro)return;
  const plot=ro.querySelector(':scope>.sp-plot-intake');if(!plot)return;
  const legends=ro.querySelectorAll(':scope>.sp-legend'),legend2=legends.length>1?legends[legends.length-1]:null;
  let tile=document.getElementById('panel-intake');
  if(!tile){
   tile=document.createElement('section');tile.className='site-panel';tile.id='panel-intake';
   tile.setAttribute('aria-labelledby','sp-intake');tile.style.setProperty('--sp-color','var(--performance)');
   tile.innerHTML='<div class="section-heading"><h2 id="sp-intake">Rollout intake</h2></div>';
   ro.after(tile);
  }
  tile.append(plot);if(legend2)tile.append(legend2);
  ro.querySelector(':scope>.sp-subhead')?.remove();
 }
 new MutationObserver(split).observe(main,{childList:true,subtree:true});split();
})();
