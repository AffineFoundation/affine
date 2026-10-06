(() => {
  'use strict';
  const $ = id => document.getElementById(id), ns = 'http://www.w3.org/2000/svg';
  let data = {epochs:[], evaluations:[], summary:{}}, hasRecords = false, recordsUnavailable = false;
  // A changing checkpoint is expected. A changing task/runtime/sampling cohort is not comparable.
  const cohortKey = e => JSON.stringify([e.dataset_id,e.taskset_hash,e.fixed_task_ids,e.seed,e.count,e.requested_count,e.harness,e.environment_version,e.model,e.model_runtime_revision,e.output_token_budget,e.policy_kind,e.sampling_policy,e.experiment_id]);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const count = value => finite(value) && Number.isInteger(value) && value >= 0;
  const number = value => value.toLocaleString('en-GB');
  const percent = value => `${Number((value * 100).toFixed(1))}%`;
  const utc = time => new Date(time * 1000).toLocaleString('en-GB',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false});
  const epochName = row => {const match = String(row.id).match(/-(\d+)$/);return match ? `Epoch ${match[1]}` : 'Epoch';};
  function svg(tag, attrs, text){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
  function ceiling(value){const magnitude=10 ** Math.floor(Math.log10(Math.max(1,value)));return Math.ceil(value/magnitude)*magnitude;}
  function detail(kind, row){
    if(kind==='evaluation')return `${utc(row.time)} UTC · ${percent(row.value)} · ${row.successes}/${row.count} solved · checkpoint ${String(row.checkpoint||'').slice(0,12)}${row.recovered_count?` · ${row.recovered_count} explicit recoveries`:''}`;
    const audit = row.audit_breakdown_available===false?'Independent audit breakdown unavailable here':['accepted','rejected','unchecked'].map(key=>count(row[key])?`${number(row[key])} ${key==='accepted'?'fully audited and accepted':key==='rejected'?'rejected':'unchecked'}`:`${key} count unavailable`).join(' · ');
    return `${epochName(row)} · ${utc(row.time)} UTC · ${number(row.value)} submitted${row.learner_input_assurance==='unaudited'?` · ${number(row.learner_eligible)} learner-eligible (unaudited) · ${number(row.learner_excluded)} excluded`:''} · ${audit}`;
  }
  function draw(kind, rows){
    const s=$(kind+'-chart'),tip=$(kind+'-tip'),activeRecord=s.contains(document.activeElement)?document.activeElement.getAttribute('data-record'):null;s.replaceChildren();tip.hidden=true;
    const width=Math.max(240,s.clientWidth),height=Math.max(200,s.clientHeight),x0=width<500?42:46,y0=28,w=width-x0-12,h=height-y0-42;
    s.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const high=kind==='evaluation'?1:ceiling(Math.max(4,...rows.map(e=>e.value)));
    for(let i=0;i<=4;i++){
      const value=high*i/4,y=y0+h-h*i/4;
      s.append(svg('line',{x1:x0,y1:y,x2:x0+w,y2:y,stroke:i===0?'#b8b8b8':'#e8e8e8','stroke-dasharray':i===0?'0':'3 5'}));
      s.append(svg('text',{x:x0-12,y:y+4,'text-anchor':'end'},kind==='evaluation'?percent(value):number(Number(value.toFixed(2)))));
    }
    $(kind+'-empty').hidden=rows.length>0;
    $(kind+'-empty').textContent=kind==='evaluation'?'Awaiting a completed held-out evaluation.\nOnly measured results from this run appear here.':'Awaiting a committed epoch.\nCommitted submission counts will appear here.';
    const first=rows[0]?.time,last=rows.at(-1)?.time;
    const x=(e,i)=>kind==='batch'?(rows.length===1?x0+w/2:x0+w*i/(rows.length-1)):(last===first?x0+w/2:x0+w*(e.time-first)/(last-first));
    const y=e=>y0+h-h*e.value/high;
    if(rows.length){
      s.append(svg('polyline',{points:rows.map((e,i)=>`${x(e,i)},${y(e)}`).join(' '),fill:'none',stroke:'#111','stroke-width':1.75,'stroke-linejoin':'round'}));
      s.setAttribute('aria-label',`${kind==='evaluation'?'Held-out math performance':'Committed batches per epoch'}, ${rows.length} measurements. Use left and right arrow keys to inspect points.`);
    }else s.setAttribute('aria-label',kind==='evaluation'?'No completed held-out evaluations for the current run':'No committed epochs for the current run');
    const selection=svg('g',{class:'chart-selection'}),guide=svg('line',{y1:y0,y2:y0+h,stroke:'#aaa','stroke-dasharray':'3 4'}),marker=svg('circle',{r:5,fill:'#111',stroke:'#fff','stroke-width':2});
    selection.append(guide,marker);
    const hits=[];
    const hide=()=>{tip.hidden=true;selection.setAttribute('visibility','hidden');};
    const show=index=>{const e=rows[index];tip.hidden=false;tip.textContent=detail(kind,e);selection.setAttribute('visibility','visible');guide.setAttribute('x1',x(e,index));guide.setAttribute('x2',x(e,index));marker.setAttribute('cx',x(e,index));marker.setAttribute('cy',y(e));};
    rows.forEach((e,i)=>{
      const px=x(e,i),py=y(e);s.append(svg('circle',{cx:px,cy:py,r:rows.length===1?4:2.5,class:'chart-point','data-value':e.value,'data-record':e.run_id||e.id}));
      const previous=i===0?x0:(x(rows[i-1],i-1)+px)/2,next=i===rows.length-1?x0+w:(px+x(rows[i+1],i+1))/2;
      const hit=svg('rect',{x:previous,y:y0,width:Math.max(1,next-previous),height:h,class:'chart-hit',tabindex:i===rows.length-1?0:-1,role:'button','aria-label':detail(kind,e),'data-record':e.run_id||e.id});
      hit.addEventListener('pointerenter',event=>{if(event.pointerType!=='touch')show(i);});
      hit.addEventListener('pointerleave',event=>{if(event.pointerType!=='touch'&&document.activeElement!==hit)hide();});
      hit.addEventListener('focus',()=>show(i));hit.addEventListener('blur',hide);hit.addEventListener('click',()=>show(i));
      hit.addEventListener('keydown',event=>{
        let index=i;
        if(event.key==='ArrowLeft'||event.key==='ArrowDown')index=Math.max(0,i-1);
        else if(event.key==='ArrowRight'||event.key==='ArrowUp')index=Math.min(rows.length-1,i+1);
        else if(event.key==='Home')index=0;
        else if(event.key==='End')index=rows.length-1;
        else if(event.key==='Escape'){event.preventDefault();hide();return;}
        else if(event.key==='Enter'||event.key===' '){event.preventDefault();show(i);return;}
        else return;
        event.preventDefault();hits.forEach((n,j)=>n.setAttribute('tabindex',j===index?0:-1));hits[index].focus();
      });
      hits.push(hit);s.append(hit);
    });
    selection.setAttribute('visibility','hidden');s.append(selection);
    const labels = rows.length<=3?rows.map((_,i)=>i):width<500?[0,rows.length-1]:[0,Math.floor((rows.length-1)/2),rows.length-1];
    for(const i of new Set(labels)){
      const e=rows[i],anchor=rows.length===1?'middle':i===0?'start':i===rows.length-1?'end':'middle';
      s.append(svg('text',{x:x(e,i),y:y0+h+28,'text-anchor':anchor},kind==='batch'?epochName(e):utc(e.time)));
    }
    const restored=hits.find(hit=>hit.getAttribute('data-record')===activeRecord);
    if(restored){hits.forEach(hit=>hit.setAttribute('tabindex',hit===restored?0:-1));restored.focus({preventScroll:true});}
    $(kind+'-period').textContent=`${rows.length} ${kind==='evaluation'?'completed evaluations · UTC':'committed epochs · current run'}`;
  }
  function render(){
    const epochs=data.epochs.filter(e=>e.source==='live-reward-math'&&finite(e.start)).sort((a,b)=>a.start-b.start),ids=new Set(epochs.map(e=>e.id));
    const evals=data.evaluations.filter(e=>ids.has(e.epoch_id)&&e.env_id==='affine_math'&&e.status==='complete'&&finite(e.timestamp)&&finite(e.mean_reward)&&e.mean_reward>=0&&e.mean_reward<=1&&count(e.count)&&e.count>0&&count(e.successes)&&e.successes<=e.count).sort((a,b)=>a.timestamp-b.timestamp);
    const latest=evals.at(-1),comparable=latest?evals.filter(e=>cohortKey(e)===cohortKey(latest)):[],finalized=epochs.filter(e=>e.batches_available!==false&&count(e.batches)),lastEpoch=finalized.at(-1);
    $('evaluation-value').textContent=latest?percent(latest.mean_reward):'—';
    $('evaluation-reading').textContent=latest?`${latest.successes} / ${latest.count} problems solved`:'Awaiting a completed evaluation';
    $('evaluation-note').textContent=latest?`${latest.count} fixed held-out problems · same evaluation settings`:'Actual held-out measurements only';
    $('evaluation-note').title='Only the current run and latest comparable task/runtime/sampling cohort. Small diagnostic cohorts do not establish broad improvement.';
    $('batch-value').textContent=lastEpoch?number(lastEpoch.batches):'—';
    $('batch-reading').textContent=lastEpoch?`${epochName(lastEpoch)} · committed submissions`:'Awaiting a committed epoch';
    $('batch-note').textContent=lastEpoch?.learner_input_assurance==='unaudited'?`${number(lastEpoch.learner_eligible)} learner-eligible · unaudited inputs · independent audits`:lastEpoch&&count(lastEpoch.accepted)&&count(lastEpoch.unchecked)?`${number(lastEpoch.accepted)} fully audited and accepted · ${number(lastEpoch.unchecked)} unchecked`:'Frozen submissions · inspect a point for audit results';
    draw('evaluation',comparable.map(e=>({...e,time:e.timestamp,value:e.mean_reward})));
    draw('batch',finalized.map(e=>({...e,time:e.start,value:e.batches})));
    const updated=data.summary?.updated_at,stale=finite(updated)&&Date.now()/1000-updated>120;
    $('connection').dataset.state=recordsUnavailable?'unavailable':stale?'stale':'current';
    $('connection').textContent=recordsUnavailable?`Records unavailable · showing snapshot${finite(updated)?` from ${utc(updated)} UTC`:''}`:finite(updated)?`${stale?'Snapshot may be stale':'Records updated'} · ${utc(updated)} UTC`:'Snapshot update time unavailable';
  }
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,100);});
  document.addEventListener('pointerdown',event=>{if(!event.target.closest('.chart-wrap'))for(const kind of ['evaluation','batch']){$(kind+'-tip').hidden=true;$(kind+'-chart').querySelector('.chart-selection')?.setAttribute('visibility','hidden');}});
  async function refresh(){try{
    const r=await fetch('/network-data.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('Data unavailable');const next=await r.json();
    if(!Array.isArray(next.epochs)||!Array.isArray(next.evaluations)||next.epochs.some(e=>!e||typeof e!=='object')||next.evaluations.some(e=>!e||typeof e!=='object'))throw Error('Invalid data');
    data=next;hasRecords=true;recordsUnavailable=false;render();
  }catch{
    recordsUnavailable=true;
    $('connection').dataset.state='unavailable';
    $('connection').textContent=hasRecords?`Records unavailable · showing snapshot${finite(data.summary?.updated_at)?` from ${utc(data.summary.updated_at)} UTC`:''}`:'Network records temporarily unavailable';
    if(!hasRecords)for(const kind of ['evaluation','batch']){$(kind+'-chart').replaceChildren();$(kind+'-tip').hidden=true;$(kind+'-empty').hidden=false;$(kind+'-empty').textContent='Network records are temporarily unavailable.\nWaiting for a valid snapshot.';$(kind+'-period').textContent='Waiting for network records';}
  }}
  refresh();setInterval(refresh,15000);
})();
