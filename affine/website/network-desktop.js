/* Layout only: every graph is rendered by the shared phone chart code. */
(()=>{
 const media=matchMedia('(min-width:1000px)'),market=document.querySelector('.market-prices');
 const home=document.createComment('market prices home');market.before(home);
 const readings=['tao-price','alpha-price','batches-total'].map(id=>document.getElementById(id));
 function fit(){
  for(const reading of readings){
   reading.parentElement.style.removeProperty('--desktop-reading-size');
   if(!media.matches)continue;
   const available=reading.parentElement.clientWidth;
   for(let i=0;i<4;i++){
    const width=reading.getBoundingClientRect().width,size=parseFloat(getComputedStyle(reading).fontSize);
    if(!width||width<=available-2)break;
    reading.parentElement.style.setProperty('--desktop-reading-size',Math.max(12,size*(available-4)/width)+'px');
   }
  }
 }
 function adapt(){
  if(media.matches)document.querySelector('header').insertBefore(market,document.querySelector('header nav'));
  else home.after(market);
  fit();
 }
 const observer=new MutationObserver(fit);
 readings.forEach(reading=>observer.observe(reading,{childList:true,characterData:true,subtree:true}));
 media.addEventListener('change',adapt);window.addEventListener('resize',fit);document.fonts.ready.then(fit);adapt();
})();

/* Appearance uses the same controls and chart palette on every viewport. */
(()=>{
 const root=document.documentElement,button=document.getElementById('theme-toggle');
 function sync(){const light=root.dataset.theme==='light';button.setAttribute('aria-pressed',String(light));button.setAttribute('aria-label',light?'Switch to dark mode':'Switch to light mode');button.title=button.getAttribute('aria-label');document.querySelector('meta[name="theme-color"]').content=light?'#ffffff':'#000000';}
 function apply(theme){if(theme==='light')root.dataset.theme='light';else delete root.dataset.theme;sync();window.dispatchEvent(new Event('affine:theme'));}
 button.addEventListener('click',()=>{const theme=root.dataset.theme==='light'?'dark':'light';try{localStorage.setItem('affine:theme',theme);}catch{}apply(theme);});
 window.addEventListener('storage',event=>{if(event.key==='affine:theme')apply(event.newValue);});sync();
})();

/* The desktop detail row and task concentration map were retired on 2026-10-21:
   they read audit/batch-grid fields that the current lifecycle no longer produces.
   Current-run panels live in site-metrics.js. */
