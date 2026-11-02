/* Header + footer for the trimmed K=1 page (2026-10-29, Jacob t774u). Replaces network.js / site-metrics.js on this page:
   TAO and Alpha prices from /incentive-live.json (live chain snapshot), Groups total from /site-metrics.json, footer freshness, Copy view. */
(()=>{
 const $=id=>document.getElementById(id),fin=n=>typeof n==='number'&&Number.isFinite(n);
 const utc=t=>new Date(t*1000).toLocaleString('en-GB',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false});
 const main=$('main');if(main){main.dataset.state='ready';main.setAttribute('aria-busy','false');}
 async function j(u){const r=await (window.AFX&&AFX.take(u)||fetch(u,{cache:'no-store',signal:AbortSignal.timeout(15000)}));/* pt1070u */if(!r.ok)throw Error(u+' '+r.status);return r.json();}
 async function prices(){
  const c=$('connection');
  try{const d=await j('/incentive-live.json'),p=d.incentive_prices||{};
   if(fin(p.tao_usd))$('tao-price').textContent='$'+p.tao_usd.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
   if(fin(p.alpha_tao)){if(fin(p.tao_usd))$('alpha-price').textContent='$'+(p.alpha_tao*p.tao_usd).toLocaleString('en-US',{minimumFractionDigits:4,maximumFractionDigits:4});$('alpha-price-tao').textContent=p.alpha_tao.toFixed(5)+' TAO';}
   const t=d.generated_at;if(c&&fin(t)){const stale=Date.now()/1000-t>900;c.dataset.state=stale?'stale':'current';c.textContent='Chain data · '+(stale?'Stale · ':'')+utc(t)+' UTC';}
  }catch{if(c){c.dataset.state='unavailable';c.textContent='Chain data · unavailable · retrying';}}
 }
 async function groups(){
  try{const d=await j('/site-metrics-recent.json').catch(()=>j('/site-metrics.json')),/* pt1076u: slice has every epoch's groups_total */rows=(d.epochs||[]).filter(e=>fin(e.index)&&fin(e.groups_total));if(!rows.length)return;
   const total=rows.reduce((a,e)=>a+e.groups_total,0),n=$('batches-total');n.textContent=total.toLocaleString('en-GB');
   const t='Rollout groups submitted across '+rows.length+' epochs (E'+rows[0].index+'–E'+rows[rows.length-1].index+'); K=1 trains one UID\'s batch per epoch';n.title=t;n.setAttribute('aria-label',t);
  }catch{}
 }
 $('copy-view')?.addEventListener('click',async()=>{const b=$('copy-view'),s=$('copy-view-status');
  try{await navigator.clipboard.writeText(location.href);b.textContent='Copied';if(s)s.textContent='View link copied';}catch{b.textContent='Copy unavailable';}
  setTimeout(()=>{b.textContent='Copy view';if(s)s.textContent='';},1800);});
 prices();groups();setInterval(prices,60000);setInterval(groups,120000);
})();
