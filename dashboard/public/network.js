(() => {
  'use strict';
  const $ = id => document.getElementById(id), ns = 'http://www.w3.org/2000/svg';
  let data = {epochs:[], evaluations:[], summary:{}}, scope='test', environment='affine_math', cohort='', source='all', sourceChosen=false;
  // Weights may change within a cohort; every task/runtime/sampling identity stays fixed.
  const cohortKey = e => JSON.stringify([e.dataset_id,e.taskset_hash,e.fixed_task_ids,e.seed,e.count,e.requested_count,e.harness,e.environment_version,e.model,e.model_runtime_revision,e.output_token_budget,e.policy_kind]);
  const mode = e => data.epochs.find(x=>x.id===e.epoch_id)?.mode || (/^(nonpayable-|test-|mock-)/.test(e.epoch_id||'')?'test':'live');
  const names = {'native-math-common':'Original MATH · SmolLM2-1.7B','separated-hopper-math':'Original MATH · Qwen2.5-Math-7B','separated-hopper-math-recovery':'Original MATH · Qwen2.5-Math-7B recovery','gpu-wide':'Wide pilot','gpu-continuous':'GPU pilot','native-sql-common':'SQL pilot','native-agent-common':'Agent pilot','native-eog-common':'Calendar pilot'};
  function svg(tag, attrs, text){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
  function menu(id, options, chosen){const n=$(id);n.replaceChildren();for(const [value,label] of options){const o=document.createElement('option');o.value=value;o.textContent=label;o.title=label;n.append(o);}n.value=chosen;n.disabled=!options.length;}
  function draw(kind, rows){
    const s=$(kind+'-chart');s.replaceChildren();$(kind+'-tip').hidden=true;
    const width=Math.max(240,s.clientWidth),height=s.clientHeight,x0=44,y0=24,w=width-x0-10,h=height-y0-46;
    s.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const values=rows.map(e=>e.value),low=Math.min(0,...values),high=kind==='evaluation'?Math.max(1,...values):Math.max(4,Math.ceil(Math.max(0,...values)/4)*4);
    for(let i=0;i<=4;i++){const y=y0+h-h*i/4;s.append(svg('line',{x1:x0,y1:y,x2:x0+w,y2:y,stroke:i===0?'#999':'#ddd','stroke-dasharray':i===0?'0':'4 5'}));s.append(svg('text',{x:x0-10,y:y+4,'text-anchor':'end'},String(Number((low+(high-low)*i/4).toFixed(3)))));}
    s.append(svg('line',{x1:x0,y1:y0,x2:x0,y2:y0+h,stroke:'#999'}));
    $(kind+'-empty').hidden=rows.length>0;
    $(kind+'-empty').textContent=kind==='evaluation'?'No completed held-out evaluations for this scope.':'No finalized epochs for this scope.';
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
    const evals=data.evaluations.filter(e=>e.status==='complete'&&mode(e)===scope&&Number.isFinite(e.mean_reward)).sort((a,b)=>a.timestamp-b.timestamp);
    const envs=[...new Set(evals.map(e=>e.env_id))].sort();
    if(!envs.includes(environment))environment=envs.includes('affine_math')?'affine_math':envs[0]||'';
    menu('environment',envs.map(id=>[id,id==='affine_math'?'Original MATH':id==='affine_numina'?'Numina (historical Lean)':id]),environment);
    const candidates=evals.filter(e=>e.env_id===environment),groups=new Map();candidates.forEach(e=>groups.set(cohortKey(e),e));
    if(!groups.has(cohort))cohort=candidates.length?cohortKey(candidates.at(-1)):'';
    menu('evaluation',[...groups].reverse().map(([key,e])=>[key,`${(e.model||'Model').split('/').at(-1)} · ${e.count} tasks · ${e.output_token_budget||'?'} tokens · ${e.harness} · ${e.dataset_id.slice(0,8)} · ${e.model_runtime_revision||'runtime unrecorded'}`]),cohort);
    const latest=groups.get(cohort),selected=candidates.filter(e=>cohortKey(e)===cohort);
    $('evaluation-title').textContent=`${environment==='affine_math'?'Original MATH':environment||'Evaluation'} / performance over time`;
    $('evaluation-note').textContent=latest?`${latest.policy_kind==='curated-control'||latest.harness?.includes('candidates')?'Curated control':'Model sampling'} · ${latest.count} fixed held-out tasks · mean reward`:'Actual held-out measurements only';
    $('evaluation-note').title=latest?`Cohort ${latest.dataset_id}. Repeated checkpoint evaluations remain separate measurements. Small cohorts do not establish broad improvement.`:'';
    draw('evaluation',selected.map(e=>({...e,time:e.timestamp,value:e.mean_reward})));
    const epochs=data.epochs.filter(e=>e.mode===scope&&e.finalized).sort((a,b)=>a.start-b.start),sources=[...new Set(epochs.map(e=>e.source))].sort();
    const latestMath=epochs.filter(e=>['native-math-common','separated-hopper-math','separated-hopper-math-recovery'].includes(e.source)).at(-1);
    if(!sourceChosen||(source!=='all'&&!sources.includes(source))){source=latestMath?.source||'all';sourceChosen=false;}
    menu('batch-source',[['all','All epoch series'],...sources.map(s=>[s,names[s]||s])],source);
    draw('batch',epochs.filter(e=>source==='all'||e.source===source).map(e=>({...e,time:e.start,value:e.batches})));
    $('connection').textContent=`${scope==='test'?'Nonpayable pilot':'Network'} · updated ${new Date(data.summary.updated_at*1000).toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',timeZone:'UTC'})} UTC`;
  }
  document.querySelectorAll('[data-scope]').forEach(b=>b.addEventListener('click',()=>{scope=b.dataset.scope;document.querySelectorAll('[data-scope]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));render();}));
  $('environment').addEventListener('change',()=>{environment=$('environment').value;cohort='';render();});
  $('evaluation').addEventListener('change',()=>{cohort=$('evaluation').value;render();});
  $('batch-source').addEventListener('change',()=>{source=$('batch-source').value;sourceChosen=true;render();});
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,100);});
  async function refresh(){try{const r=await fetch('/network-data.json',{cache:'no-store'});if(!r.ok)throw Error('Data unavailable');const next=await r.json();if(!Array.isArray(next.epochs)||!Array.isArray(next.evaluations))throw Error('Invalid data');data=next;render();}catch{
    $('connection').textContent='Data unavailable';
    for(const kind of ['evaluation','batch']){$(kind+'-chart').replaceChildren();$(kind+'-tip').hidden=true;$(kind+'-empty').hidden=false;$(kind+'-empty').textContent='Network records are temporarily unavailable.';$(kind+'-period').textContent='Waiting for records';}
  }}
  refresh();setInterval(refresh,15000);
})();
