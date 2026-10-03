(() => {
  'use strict';
  const $ = id => document.getElementById(id), ns = 'http://www.w3.org/2000/svg';
  let data = {epochs:[], evaluations:[], summary:{}};
  // Weights may change within a cohort; every task/runtime/sampling identity stays fixed.
  const cohortKey = e => JSON.stringify([e.dataset_id,e.taskset_hash,e.fixed_task_ids,e.seed,e.count,e.requested_count,e.harness,e.environment_version,e.model,e.model_runtime_revision,e.output_token_budget,e.policy_kind]);
  function svg(tag, attrs, text){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
  function draw(kind, rows){
    const s=$(kind+'-chart');s.replaceChildren();$(kind+'-tip').hidden=true;
    const width=Math.max(240,s.clientWidth),height=s.clientHeight,x0=44,y0=24,w=width-x0-10,h=height-y0-46;
    s.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const values=rows.map(e=>e.value),low=Math.min(0,...values),high=kind==='evaluation'?Math.max(1,...values):Math.max(4,Math.ceil(Math.max(0,...values)/4)*4);
    for(let i=0;i<=4;i++){const y=y0+h-h*i/4;s.append(svg('line',{x1:x0,y1:y,x2:x0+w,y2:y,stroke:i===0?'#999':'#ddd','stroke-dasharray':i===0?'0':'4 5'}));s.append(svg('text',{x:x0-10,y:y+4,'text-anchor':'end'},String(Number((low+(high-low)*i/4).toFixed(3)))));}
    s.append(svg('line',{x1:x0,y1:y0,x2:x0,y2:y0+h,stroke:'#999'}));
    $(kind+'-empty').hidden=rows.length>0;
    $(kind+'-empty').textContent=kind==='evaluation'?'No completed held-out evaluations for this run.':'No finalized epochs for this run.';
    const first=rows[0]?.time,last=rows.at(-1)?.time;
    const x=e=>last===first?x0+w/2:x0+w*(e.time-first)/(last-first),y=e=>y0+h-h*(e.value-low)/(high-low);
    if(rows.length)s.append(svg('polyline',{points:rows.map(e=>`${x(e)},${y(e)}`).join(' '),fill:'none',stroke:'#111','stroke-width':1.7}));
    rows.forEach((e,i)=>{
      const px=x(e),py=y(e);s.append(svg('circle',{cx:px,cy:py,r:3,class:'chart-point','data-value':e.value,'data-record':e.run_id||e.id}));
      const text=kind==='evaluation'?`${e.env_id} · reward ${e.value.toFixed(3)} · ${e.successes}/${e.count} successes · checkpoint ${(e.checkpoint||'').slice(0,12)}${e.recovered_count?` · ${e.recovered_count} explicit recoveries`:''}`:`${e.id} · ${e.batches} submitted · ${e.accepted} accepted · ${e.rejected} rejected · ${e.unchecked||0} unchecked`;
      const hit=svg('rect',{x:px-8,y:py-12,width:16,height:24,class:'chart-hit',tabindex:0,role:'button','aria-label':text});
      const show=()=>{$(kind+'-tip').hidden=false;$(kind+'-tip').textContent=text;};
      hit.addEventListener('mouseenter',show);hit.addEventListener('focus',show);hit.addEventListener('click',show);
      for(const event of ['mouseleave','blur'])hit.addEventListener(event,()=>{$(kind+'-tip').hidden=true;});s.append(hit);
      if(i===0||i===rows.length-1)s.append(svg('text',{x:px,y:y0+h+28,'text-anchor':rows.length===1?'middle':i===0?'start':'end'},new Date(e.time*1000).toLocaleString('en-GB',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false})));
    });
    $(kind+'-period').textContent=`${rows.length} ${kind==='evaluation'?'completed evaluations':'finalized epochs'} · UTC`;
  }
  function render(){
    const epochs=data.epochs.filter(e=>e.source==='live-reward-math').sort((a,b)=>a.start-b.start);
    const ids=new Set(epochs.map(e=>e.id));
    const evals=data.evaluations.filter(e=>ids.has(e.epoch_id)&&e.env_id==='affine_math'&&e.status==='complete'&&Number.isFinite(e.mean_reward)).sort((a,b)=>a.timestamp-b.timestamp);
    const latest=evals.at(-1);
    const comparable=latest?evals.filter(e=>cohortKey(e)===cohortKey(latest)):[];
    $('evaluation-title').textContent='Math performance / time';
    $('evaluation-note').textContent=latest?`${latest.successes}/${latest.count} solved · fixed held-out tasks`:'Actual held-out measurements only';
    $('evaluation-note').title='Only the current run. Fixed task, runtime and sampling settings are compared; small cohorts do not establish broad improvement.';
    draw('evaluation',comparable.map(e=>({...e,time:e.timestamp,value:e.mean_reward})));
    draw('batch',epochs.filter(e=>e.finalized).map(e=>({...e,time:e.start,value:e.batches})));
    $('connection').textContent=`Current run · updated ${new Date(data.summary.updated_at*1000).toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',timeZone:'UTC'})} UTC`;
  }
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,100);});
  async function refresh(){try{const r=await fetch('/network-data.json',{cache:'no-store'});if(!r.ok)throw Error('Data unavailable');const next=await r.json();if(!Array.isArray(next.epochs)||!Array.isArray(next.evaluations))throw Error('Invalid data');data=next;render();}catch{
    $('connection').textContent='Data unavailable';
    for(const kind of ['evaluation','batch']){$(kind+'-chart').replaceChildren();$(kind+'-tip').hidden=true;$(kind+'-empty').hidden=false;$(kind+'-empty').textContent='Network records are temporarily unavailable.';$(kind+'-period').textContent='Waiting for records';}
  }}
  refresh();setInterval(refresh,15000);
})();
