(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const svgNS = 'http://www.w3.org/2000/svg';
  let data = {epochs:[],evaluations:[],summary:{}}, scope = 'test', chosen = null, metric='reward', environment='', evaluation='';
  const seriesKey = e => JSON.stringify([e.dataset_id,e.harness,e.environment_version,e.model,e.model_runtime_revision]);
  function svg(tag, attrs, text) {const n=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
  function completed() {return data.epochs.filter(e=>e.mode===scope && e.finalized).sort((a,b)=>a.start-b.start);}
  function drawChart(rows) {
    const s=$('chart');s.replaceChildren();
    // Draw in screen-sized coordinates so axis text stays readable on phones.
    const width=Math.max(240,s.clientWidth||1200),height=Math.max(240,s.clientHeight||540);
    const x0=44,y0=16,w=width-x0-14,h=height-y0-42;
    s.setAttribute('viewBox',`0 0 ${width} ${height}`);
    s.setAttribute('aria-label',metric==='reward'?`Held-out evaluation reward for ${environment}`:'Batches submitted per completed epoch');
    const values=rows.map(e=>e.value||0),max=Math.max(1,...values),floor=metric==='reward'?Math.min(0,...values):0;
    const ceiling=metric==='reward'?Math.ceil(max*4)/4:max<=4?4:Math.ceil(max/4)*4;
    for(let i=0;i<=4;i++){const y=y0+h-h*i/4;s.append(svg('line',{x1:x0,y1:y,x2:x0+w,y2:y,stroke:i===0?'#999':'#ddd','stroke-dasharray':i===0?'0':'4 5'}));s.append(svg('text',{x:x0-12,y:y+4,'text-anchor':'end'},String(Number((floor+(ceiling-floor)*i/4).toFixed(3)))));}
    s.append(svg('line',{x1:x0,y1:y0,x2:x0,y2:y0+h,stroke:'#999'}));
    $('chart-empty').hidden=rows.length>0;$('chart-empty').textContent=metric==='reward'?'No completed held-out evaluations for this environment yet.':'No completed epochs yet.';
    const x = i => rows.length===1?x0+w/2:x0+i*w/(rows.length-1), y = count => y0+h-h*(count-floor)/(ceiling-floor);
    if(rows.length){s.append(svg('polyline',{points:rows.map((e,i)=>`${x(i)},${y(e.value||0)}`).join(' '),fill:'none',stroke:'#111','stroke-width':1.7,'stroke-linejoin':'round'}));}
    rows.forEach((e,i)=>{
      const px=x(i),py=y(e.value||0);s.append(svg('circle',{cx:px,cy:py,r:rows.length===1?4:2,class:'chart-point'}));
      const hit=svg('rect',{x:px-18,y:y0,width:36,height:h,class:'chart-hit',tabindex:0,role:'button','aria-label':metric==='reward'?`${e.env_id}, evaluation reward ${e.value.toFixed(3)}`:`${e.id}, ${e.batches||0} batches`});
      const show=()=>{const t=$('chart-tip');t.hidden=false;t.textContent=metric==='reward'?`${e.env_id} · reward ${e.value.toFixed(3)} · ${e.successes}/${e.count} successes · ${(e.checkpoint||'').slice(0,10)}`:`${e.id} · ${e.batches||0} batches · ${e.accepted||0} accepted`;};
      hit.addEventListener('mouseenter',show);hit.addEventListener('focus',show);hit.addEventListener('mouseleave',()=>{$('chart-tip').hidden=true;});hit.addEventListener('blur',()=>{$('chart-tip').hidden=true;});
      const select=()=>{const epoch=data.epochs.find(x=>x.id===(e.epoch_id||e.id));if(epoch){chosen=epoch.id;$('epoch').value=chosen;drawGrid(epoch);}};
      hit.addEventListener('click',select);hit.addEventListener('keydown',ev=>{if(ev.key==='Enter')select();});s.append(hit);
      if(i===0||i===rows.length-1||rows.length<Math.max(3,Math.floor(w/75)))s.append(svg('text',{x:px,y:y0+h+24,'text-anchor':i===0&&rows.length>1?'start':i===rows.length-1&&rows.length>1?'end':'middle'},new Date(e.start*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false})));
    });
    $('period').textContent=`${rows.length} completed ${metric==='reward'?'evaluation':'epoch'}${rows.length===1?'':'s'} · UTC`;
  }
  function drawGrid(epoch) {
    const counts=epoch?.grid||Array(256).fill(0), max=Math.max(1,...counts);$('grid').replaceChildren();
    for(let uid=0;uid<256;uid++){const count=counts[uid]||0,shade=count?Math.round(235-220*count/max):242;
      const b=document.createElement('button');b.className='cell';b.textContent=count;b.style.backgroundColor=`rgb(${shade},${shade},${shade})`;b.style.color=shade<125?'#fff':'#111';b.title=`UID ${uid} · ${count} batches`;b.setAttribute('aria-label',`UID ${uid}: ${count} submitted batches`);b.setAttribute('aria-pressed','false');
      b.addEventListener('click',()=>{document.querySelectorAll('.cell[aria-pressed=true]').forEach(n=>n.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');$('selection').textContent=`UID ${uid} · ${count} batch${count===1?'':'es'} · ${epoch?.id||'No completed epoch'}`;});$('grid').append(b);
    }
    $('scale').textContent=max;$('grid-note').textContent=epoch?`UID 0–255 · ${epoch.mode==='test'?'nonpayable pilot · ':''}${epoch.batches||0} submitted batches`:'UID 0–255 · no completed epoch';
    $('selection').textContent=epoch?.unassigned_batches?`${epoch.unassigned_batches} batches have no mapped UID in this view. Select a square to inspect.`:'Select a square to inspect its UID.';
  }
  function render(){
    const rows=completed(), evals=(data.evaluations||[]).filter(e=>{
      const mode=data.epochs.find(epoch=>epoch.id===e.epoch_id)?.mode||(/^(nonpayable-|test-|mock-)/.test(e.epoch_id||'')?'test':'live');
      return e.status==='complete'&&mode===scope;
    });
    const envs=[...new Set(evals.map(e=>e.env_id))].sort(), menu=$('environment');menu.replaceChildren();
    for(const id of envs.length?envs:['']){const o=document.createElement('option');o.value=id;o.textContent=id||'No evaluations yet';menu.append(o);}
    if(!envs.includes(environment))environment=envs[0]||'';menu.value=environment;
    let chartRows=rows.map(e=>({...e,value:e.batches||0}));
    const candidates=evals.filter(e=>e.env_id===environment).sort((a,b)=>a.timestamp-b.timestamp);
    const groups=new Map();candidates.forEach(e=>groups.set(seriesKey(e),e));
    const seriesMenu=$('evaluation');seriesMenu.replaceChildren();
    for(const [key,e] of [...groups].reverse()){
      const o=document.createElement('option');o.value=key;
      o.textContent=`${(e.model||'Model').split('/').at(-1)} · ${e.model_runtime_revision?.startsWith('cuda')?'GPU':'CPU'} · ${e.harness} · ${e.dataset_id.slice(0,8)}`;
      seriesMenu.append(o);
    }
    if(!groups.has(evaluation))evaluation=candidates.length?seriesKey(candidates.at(-1)):'';
    seriesMenu.value=evaluation;seriesMenu.disabled=metric==='batches';
    if(metric==='reward'){
      const latest=groups.get(evaluation);
      chartRows=candidates.filter(e=>seriesKey(e)===evaluation).map(e=>({...e,id:e.run_id,start:e.timestamp,value:e.mean_reward}));
      const policy=latest?.policy_kind||(latest?.harness?.includes('candidates')?'curated-control':'autoregressive');
      $('metric-note').textContent=latest?`${policy==='curated-control'?'Curated policy':'Model sampling'} · fixed held-out set · ${latest.count} samples`:'Waiting for held-out evaluations';
      $('metric-note').title=latest?.harness||'';
    }else $('metric-note').textContent='Frozen submitted batches · includes rejected batches';
    $('chart-title').textContent=metric==='reward'?`${environment||'Environment'} / evaluation reward`:'Batches / epoch';
    menu.disabled=metric==='batches';drawChart(chartRows);
    const select=$('epoch');select.replaceChildren();rows.slice().reverse().forEach(e=>{const o=document.createElement('option');o.value=e.id;o.title=e.id;o.textContent=new Date(e.start*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false,timeZone:'UTC'})+' UTC';select.append(o);});if(!rows.some(e=>e.id===chosen))chosen=rows.at(-1)?.id;select.value=chosen||'';drawGrid(rows.find(e=>e.id===chosen));$('connection').textContent='SN120 / PILOT';$('update').textContent=`Updated ${new Date(data.summary.updated_at*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false})} UTC`;
  }
  document.querySelectorAll('[data-scope]').forEach(b=>b.addEventListener('click',()=>{scope=b.dataset.scope;chosen=null;document.querySelectorAll('[data-scope]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render();}));
  $('epoch').addEventListener('change',()=>{chosen=$('epoch').value;drawGrid(data.epochs.find(e=>e.id===chosen));});
  $('metric').addEventListener('change',()=>{metric=$('metric').value;render();});
  $('environment').addEventListener('change',()=>{environment=$('environment').value;render();});
  $('evaluation').addEventListener('change',()=>{evaluation=$('evaluation').value;render();});
  let resizeTimer;
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,150);});
  async function refresh(){try{const r=await fetch('/network-data.json',{cache:'no-store'});if(!r.ok)throw Error('Data unavailable');data=await r.json();render();}catch(e){$('connection').textContent='DATA UNAVAILABLE';$('chart-empty').hidden=false;$('chart-empty').textContent='Network records are temporarily unavailable.';if(!$('grid').children.length)drawGrid(null);}}
  refresh();setInterval(refresh,15000);
})();
