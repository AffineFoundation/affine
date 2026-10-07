(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ns = 'http://www.w3.org/2000/svg';
  const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue('--'+name).trim();
  const colors=Object.fromEntries(['accepted','performance','incentive','batch','unchecked','rejected'].map(name=>[name,null]));
  for(const name of Object.keys(colors))Object.defineProperty(colors,name,{get:()=>themeColor(name)});
  const colorCache=new Map(),colorCanvas=document.createElement('canvas');colorCanvas.width=colorCanvas.height=1;
  const colorContext=colorCanvas.getContext('2d',{willReadFrequently:true});
  const colorChannels=(color,fallback)=>{
    const value=CSS.supports('color',color)?color:fallback;
    if(!colorCache.has(value)){
      colorContext.clearRect(0,0,1,1);colorContext.fillStyle=value;colorContext.fillRect(0,0,1,1);
      colorCache.set(value,Array.from(colorContext.getImageData(0,0,1,1).data).slice(0,3));
    }
    return colorCache.get(value);
  };
  const mixColor=(low,high,weight)=>{
    const a=colorChannels(low,'#386da9'),b=colorChannels(high,'#86baff');
    return `rgb(${a.map((value,index)=>Math.round(value+(b[index]-value)*weight)).join(',')})`;
  };
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const count = value => finite(value) && Number.isInteger(value) && value >= 0;
  const number = value => value.toLocaleString('en-GB');
  const percent = value => `${Number((100 * value).toFixed(1))}%`;
  const day = time => new Date(time * 1000).toLocaleDateString('en-GB',{day:'numeric',month:'short',timeZone:'UTC'});
  const utc = time => new Date(time * 1000).toLocaleString('en-GB',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false});
  const epochNumber = row => String(row.id).match(/-(\d+)$/)?.[1] || String(row.id);
  const epochName = row => `Epoch ${epochNumber(row)}`;
  // Checkpoints change, but tasks, runtime and sampling must stay comparable.
  const cohortKey = e => JSON.stringify([e.dataset_id,e.taskset_hash,e.fixed_task_ids,e.seed,e.count,e.requested_count,e.harness,e.environment_version,e.model,e.model_runtime_revision,e.output_token_budget,e.policy_kind,e.sampling_policy,e.experiment_id]);
  let data = null, fingerprint = '', unavailable = false;
  let selectedMinerEpoch = null, minerEpochs = [];
  let minerInitialized=false;
  const requestedView=new URLSearchParams(location.search);
  let incentiveUnit=['tao','usd'].includes(requestedView.get('incentive_unit'))?requestedView.get('incentive_unit'):'alpha';
  let incentiveView=requestedView.get('incentive_view')==='bars'?'bars':'grid';
  let evaluationView=requestedView.get('performance_view')==='results'?'results':'trend';
  let batchView=requestedView.get('batch_view')==='trend'?'trend':'bars';
  const controllers = new Map(), entered = new Set();
  const chartStates = new Map();
  const recordKey = row => String(row.run_id||row.id);
  function syncViewLink() {
    if(matchMedia('(min-width:1000px)').matches){window.dispatchEvent(new Event('affine:copy-desktop-view'));return;}
    const url=new URL(location.href);
    const miner=minerEpochs.find(row=>row.id===selectedMinerEpoch);
    const performance=chartStates.get('evaluation'),batches=chartStates.get('batch');
    const evaluation=performance?.rows.find(row=>recordKey(row)===performance.selected);
    const batch=batches?.rows.find(row=>recordKey(row)===batches.selected);
    for(const [key,value] of [['epoch',miner?epochNumber(miner):null],['performance',evaluation?String(evaluation.timestamp):null],['batches',batch?epochNumber(batch):null]]){
      if(value===null)url.searchParams.delete(key);else url.searchParams.set(key,value);
    }
    const emission=chartStates.get('incentive');const emissionRow=emission?.rows.find(row=>recordKey(row)===emission.selected);
    if(emissionRow)url.searchParams.set('incentive_epoch',String(emissionRow.block));else url.searchParams.delete('incentive_epoch');
    if(incentiveUnit!=='alpha')url.searchParams.set('incentive_unit',incentiveUnit);else url.searchParams.delete('incentive_unit');
    if(incentiveView==='bars')url.searchParams.set('incentive_view','bars');else url.searchParams.delete('incentive_view');
    if(evaluationView==='results')url.searchParams.set('performance_view','results');else url.searchParams.delete('performance_view');
    if(batchView==='trend')url.searchParams.set('batch_view','trend');else url.searchParams.delete('batch_view');
    if(minerView==='bars')url.searchParams.set('miners','bars');else url.searchParams.delete('miners');
    if(url.href!==location.href)history.replaceState(null,'',url);
  }
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const entranceTimers=new WeakMap();
  function replayEntrance(node){
    clearTimeout(entranceTimers.get(node));
    node.classList.remove('animate-in');
    if(reducedMotion.matches)return;
    // Restart the entrance even when the previous view is still animating.
    void node.getBoundingClientRect();
    node.classList.add('animate-in');
    entranceTimers.set(node,setTimeout(()=>node.classList.remove('animate-in'),650));
  }
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      if(entry.target.id==='incentive-wrap')startIncentiveIntro();
      else replayEntrance(entry.target);
      observer.unobserve(entry.target);
    }
  },{threshold:.25}) : null;

  function svg(tag, attrs = {}, text) {
    const node = document.createElementNS(ns,tag);
    for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function positionTip(tip, wrap, x, y, pointer = null) {
    const width = wrap.clientWidth, height = wrap.clientHeight;
    tip.style.left = `${Math.max(4,Math.min(width-tip.offsetWidth-4,x-tip.offsetWidth/2))}px`;
    const anchor=pointer===null?y:pointer;
    const gap=pointer===null?16:40;
    const above=anchor-tip.offsetHeight-gap,below=anchor+gap;
    const top=above>=4?above:below+tip.offsetHeight<=height-4?below:4;
    tip.style.top = `${Math.max(4,Math.min(height-tip.offsetHeight-4,top))}px`;
  }
  function hideTip(tip){
    if(!tip.hidden)tip.dataset.hiddenAt=String(performance.now());
    tip.hidden=true;
  }
  function tooltipContent(tip, description, dismiss) {
    const hiddenAt=Number(tip.dataset.hiddenAt);
    const entering=tip.hidden&&(!finite(hiddenAt)||performance.now()-hiddenAt>150);
    tip.getAnimations().forEach(animation=>animation.cancel());
    tip.replaceChildren();
    tip.classList.toggle('has-actions',Boolean(dismiss));
    const lines=description.split('\n');
    const heading=document.createElement('strong'),header=lines[0].split(' · ');heading.textContent=header.shift();
    if(header.length){const status=document.createElement('span');status.className='tip-status';status.textContent=header.join(' · ');heading.append(document.createTextNode(' '),status);}
    tip.append(heading);
    let timestamp=null;
    let primary=false;
    const outcomePattern=/^(?:[\d,]+ (?:accepted|verified|unchecked|rejected|not solved|solved))(?: · [\d,]+ (?:accepted|verified|unchecked|rejected|not solved|solved))*$/;
    for(let index=1;index<lines.length;index++){
      const line=lines[index];
      if(/UTC$/.test(line)){
        timestamp=document.createElement('span');timestamp.className='tip-time';timestamp.textContent=line.replace(' at ',' · ');timestamp.title=line;continue;
      }
      const stats=line.split(' · ').map(part=>part.match(/^([\d,]+) (eligible \(unaudited\)|excluded|miners|training steps?|steps?)$/));
      if(stats.length&&stats.every(Boolean)){
        const group=document.createElement('div');group.className='tip-stats';
        for(const match of stats){
          const item=document.createElement('span');item.className='tip-stat';
          const value=document.createElement('b');value.textContent=match[1];
          const label=document.createElement('span');label.textContent=match[2].replace('training ','');
          if(label.textContent==='eligible (unaudited)'){label.textContent='eligible';const qualifier=document.createElement('small');qualifier.textContent='Unaudited';label.append(document.createTextNode(' '),qualifier);}
          item.append(value,label);group.append(item);
        }
        tip.append(group);continue;
      }
      if(line.startsWith('Checkpoint ')){
        const checkpoint=document.createElement('span');checkpoint.className='tip-checkpoint';checkpoint.title=line;
        checkpoint.textContent='CP '+line.slice(11).split(' · ')[0].slice(0,8);
        if(line.includes(' · updated')){const badge=document.createElement('span');badge.className='tip-status tip-updated';badge.textContent='✓';badge.title='Checkpoint updated';badge.setAttribute('aria-label','Checkpoint updated');checkpoint.append(badge);}
        tip.append(checkpoint);continue;
      }
      if(outcomePattern.test(line)){

        const group=document.createElement('div');group.className='tip-outcomes';
        do{
          for(const match of lines[index].matchAll(/([\d,]+) (accepted|verified|unchecked|rejected|not solved|solved)/g)){
            const row=document.createElement('span');row.className='tip-outcome';row.dataset.outcome=match[2];
            const value=document.createElement('b');value.textContent=match[1];
            const label=document.createElement('span');label.className='tip-outcome-label';label.textContent=match[2];
            row.append(value,document.createTextNode(' '),label);group.append(row);
          }
          index++;
        }while(index<lines.length&&outcomePattern.test(lines[index]));
        index--;tip.append(group);continue;
      }
      const metric=!primary&&line.match(/^([\d,.]+%?)( (?:submitted )?batches?| · .*)?$/);
      const part=document.createElement('span');
      if(metric){
        part.className='tip-primary';primary=true;
        const value=document.createElement('b');value.textContent=metric[1];
        const detail=document.createElement('span');detail.textContent=(metric[2]||'').trim();
        part.append(value);if(detail.textContent)part.append(document.createTextNode(' '),detail);
      }else{part.className=line==='Audit breakdown unavailable'?'tip-detail tip-note':'tip-detail';part.textContent=line==='Audit breakdown unavailable'?'Audit unavailable':line;}
      tip.append(part);
    }
    const statGroups=[...tip.querySelectorAll('.tip-stats')];
    for(const group of statGroups.slice(1)){statGroups[0].append(...group.children);group.remove();}
    const checkpoint=tip.querySelector('.tip-checkpoint');
    if(timestamp||checkpoint){const footer=document.createElement('div');footer.className='tip-footer';if(checkpoint)footer.append(checkpoint);if(timestamp)footer.append(timestamp);tip.append(footer);}
    if(typeof dismiss==='function'){
      const actions=document.createElement('div');actions.className='tip-actions';
      const close=document.createElement('button');close.type='button';close.textContent='×';close.setAttribute('aria-label','Close chart details');close.addEventListener('click',dismiss);
      actions.append(close);tip.append(actions);
    }
    if(entering&&!reducedMotion.matches&&tip.animate){
      const animation=tip.animate([{opacity:0,transform:'translateY(3px)'},{opacity:1,transform:'translateY(0)'}],{duration:130,easing:'ease-out'});
      animation.id='tip-enter';
    }
  }
  function setReading(node, value) {
    const previous=node.textContent;
    if(previous===value)return;
    if(value.endsWith('%')){
      const unit=document.createElement('span');unit.className='reading-unit';unit.textContent='%';
      node.replaceChildren(document.createTextNode(value.slice(0,-1)),unit);
    }else node.textContent=value;
    if(previous!=='—'&&value!=='—'&&!reducedMotion.matches&&node.animate){
      node.getAnimations().forEach(animation=>animation.cancel());
      node.animate([{opacity:.6,transform:'translateY(2px)'},{opacity:1,transform:'translateY(0)'}],{duration:280,easing:'ease-out'});
    }
  }
  function setChange(kind, delta, unit, reference) {
    const node=$(kind+'-change');
    if(!finite(delta)){
      node.textContent='';delete node.dataset.direction;node.removeAttribute('aria-label');node.removeAttribute('title');return;
    }
    const rounded=Math.round(delta*10)/10,sign=rounded>0?'+':rounded<0?'−':'';
    node.textContent=`${sign}${number(Math.abs(rounded))}${unit==='percentage points'?' pp':''} vs prior`;
    node.dataset.direction=rounded>0?'up':rounded<0?'down':'flat';
    const description=rounded===0?`Unchanged from ${reference}`:`${number(Math.abs(rounded))} ${unit} ${rounded>0?'higher':'lower'} than ${reference}`;
    node.setAttribute('aria-label',description);node.title=description;
  }
  function animateOnce(kind, node, populated) {
    if (!populated || entered.has(kind)) return;
    entered.add(kind);
    if (observer) observer.observe(node);
  }
  function niceCeiling(value) {
    const target = Math.max(4,value), magnitude = 10 ** Math.floor(Math.log10(target));
    return Math.ceil(target/magnitude)*magnitude;
  }
  function axis(node, plot, high, performance) {
    for (const fraction of [0,.5,1]) {
      const y = plot.top + plot.height * (1-fraction);
      node.append(svg('line',{x1:plot.left,y1:y,x2:plot.left+plot.width,y2:y,stroke:fraction===0?themeColor('axis'):themeColor('line'),'stroke-width':1,...(fraction===0?{}:{'stroke-dasharray':'2 6'})}));
      node.append(svg('text',{x:plot.left-10,y,'text-anchor':'end','dominant-baseline':'middle'},performance?percent(fraction):number(high*fraction)));
    }
  }
  function evaluationValue(row){return evaluationView==='results'?row.successes/row.count:row.mean_reward;}
  function evaluationPhase(row){
    if(row.original_epoch_id)return 'Checkpoint baseline';
    const phase=String(row.run_id||'').match(/-eval-(before|after)-/)?.[1];
    return phase==='before'?'Before training':phase==='after'?'After training':'Evaluation';
  }
  function evaluationDetails(row) {
    const phase=evaluationPhase(row).replace('Checkpoint baseline','Baseline');
    return `${epochName({id:row.epoch_id})}\n${percent(evaluationValue(row))} · ${number(row.count)} tasks\n${number(row.successes)} solved · ${number(row.count-row.successes)} not solved\n${phase}${row.setup_changed?' · Setup changed':''}\n${utc(row.timestamp)} UTC`;
  }
  function validOutcomes(row) {
    return ['accepted','unchecked','rejected'].every(key=>count(row[key])) && row.accepted+row.unchecked+row.rejected===row.batches;
  }
  const batchAvailable=row=>row?.batches_available!==false;
  const batchPhase=row=>row.finalized?'Finalized':!batchAvailable(row)&&row.phase!=='collecting'?'Awaiting audit':row.phase||'Unfinalized';
  function batchDetails(row) {
    const lines = [`${epochName(row)} · ${batchPhase(row)}`,`${utc(row.start)} UTC`,batchAvailable(row)?`${number(row.batches)} batches`:'Batch counts pending audit'];
    if (!batchAvailable(row)) return lines.join('\n');
    if (row.learner_input_assurance==='unaudited') lines.push(`${number(row.learner_eligible)} eligible (unaudited) · ${number(row.learner_excluded)} excluded`);
    if (row.audit_breakdown_available!==false&&validOutcomes(row)) lines.push(`${number(row.accepted)} verified · ${number(row.unchecked)} unchecked · ${number(row.rejected)} rejected`);
    else lines.push('Audit breakdown unavailable');
    if (Array.isArray(row.grid) && row.grid.length===256 && row.grid.every(count)) {
      const steps=row.training?`${number(count(row.training.steps)?row.training.steps:0)} training ${row.training.steps===1?'step':'steps'}`:'';
      lines.push(`${number(row.grid.filter(value=>value>0).length)} miners${steps?' · '+steps:''}`);
    }
    if (row.training && !validGrid(row)) lines.push(`${number(count(row.training.steps)?row.training.steps:0)} training ${row.training.steps===1?'step':'steps'}`);
    if (row.checkpoint) lines.push(`Checkpoint ${String(row.checkpoint)}${row.training?.weights_changed?' · updated':''}`);
    return lines.join('\n');
  }
  function defaultRecord(kind, rows) {
    const latestAvailable=kind==='batch'?rows.findLastIndex(row=>batchAvailable(row)&&count(row.batches)):-1;
    return latestAvailable>=0?latestAvailable:rows.length-1;
  }
  const validEligibility=row=>count(row.learner_eligible)&&count(row.learner_excluded)&&row.learner_eligible+row.learner_excluded===row.batches;
  function syncBatchLegend(row){
    const legend=$('batch-legend');legend.replaceChildren();legend.hidden=!row;
    for(const view of ['bars','trend']){const button=$('batch-view-'+view);button.disabled=!row;button.setAttribute('aria-pressed',String(batchView===view));}
    if(!row)return;
    const add=(key,label,countValue)=>{
      const item=document.createElement('span'),swatch=document.createElement('i'),value=document.createElement('b');
      item.dataset.outcome=key;swatch.className='swatch '+key;value.className='legend-value';value.dataset.zero=String(countValue===0);value.textContent=number(countValue);
      item.append(swatch,label+' ',value);legend.append(item);
    };
    if(!batchAvailable(row)){legend.textContent='Counts pending';return;}
    const eligibility=validEligibility(row),audit=row.audit_breakdown_available!==false&&validOutcomes(row);
    add('batch','Submitted',row.batches);
    if(eligibility){
      add('eligible','Included',row.learner_eligible);add('excluded','Excluded',row.learner_excluded);
      for(const item of legend.querySelectorAll('[data-outcome="eligible"],[data-outcome="excluded"]'))item.setAttribute('title','Training admission'+(row.learner_input_assurance==='unaudited'?' · unaudited':''));
    }else if(batchView==='bars'&&audit){
      for(const [key,label] of [['accepted','Passed'],['rejected','Failed'],['unchecked','Pending']])add(key,label,row[key]);
    }
  }
  for(const view of ['bars','trend'])$('batch-view-'+view).addEventListener('click',()=>{
    if(batchView===view)return;batchView=view;
    const state=chartStates.get('batch');if(state){drawChart('batch',state.rows);controllers.get('batch')?.hide();}
    syncViewLink();const chart=$('batch-chart');chart.getAnimations().forEach(animation=>animation.cancel());observer?.unobserve(chart);replayEntrance(chart);
  });
  function syncEvaluationLegend(rows,row){
    const legend=$('evaluation-legend');legend.replaceChildren();
    for(const view of ['trend','results']){
      const button=$('evaluation-view-'+view);button.disabled=!rows.length;button.setAttribute('aria-pressed',String(evaluationView===view));
    }
    legend.hidden=!rows.length;
    if(!rows.length)return;
    if(evaluationView==='results'&&row){
      for(const [label,value,className] of [['Solved',row.successes,'accepted'],['Not solved',row.count-row.successes,'unchecked']]){
        const item=document.createElement('span'),swatch=document.createElement('i');swatch.className=`swatch ${className}`;
        const countLabel=document.createElement('b');countLabel.className='legend-value';countLabel.dataset.zero=String(value===0);countLabel.textContent=number(value);
        item.dataset.outcome=className;item.append(swatch,`${label} `,countLabel);legend.append(item);
      }
    }else{
      const changes=rows.filter(row=>row.setup_changed).length;
      legend.hidden=!changes;
      if(changes){const item=document.createElement('span');item.className='setup-legend';item.textContent='Setup changed';item.setAttribute('aria-label','Evaluation setup changed');legend.append(item);}
    }
  }
  for(const view of ['trend','results'])$('evaluation-view-'+view).addEventListener('click',()=>{
    if(evaluationView===view)return;
    evaluationView=view;
    const state=chartStates.get('evaluation');
    if(state){drawChart('evaluation',state.rows);controllers.get('evaluation')?.hide();}
    syncViewLink();
    const chart=$('evaluation-chart');chart.getAnimations().forEach(animation=>animation.cancel());
    observer?.unobserve(chart);replayEntrance(chart);
  });
  function syncNavigation(prefix, index, length) {
    $(prefix+'-previous').disabled=length<2||index<=0;
    $(prefix+'-next').disabled=length<2||index>=length-1;
  }
  function syncChartControl(kind, index) {
    if(kind==='incentive'){syncIncentiveControl(index);return;}
    const rows=chartStates.get(kind)?.rows||[],row=rows[index],slider=$(kind+'-slider');
    slider.max=Math.max(1,rows.length-1);slider.value=Math.max(0,index);slider.disabled=rows.length<2;
    syncNavigation(kind,index,rows.length);
    slider.style.setProperty('--progress',`${rows.length>1?100*index/(rows.length-1):0}%`);
    const label=row?kind==='evaluation'?`${epochName({id:row.epoch_id})} · ${evaluationPhase(row).toLowerCase()}`:`${epochName(row)} · ${batchPhase(row).toLowerCase()}`:'No records available';
    $(kind+'-slider-label').textContent=row?epochName(kind==='evaluation'?{id:row.epoch_id}:row):label;
    slider.setAttribute('aria-valuetext',row?kind==='evaluation'?`${label}, ${percent(evaluationValue(row))}, ${number(row.successes)} of ${number(row.count)} solved`:`${label}, ${batchAvailable(row)?number(row.batches)+' submitted batches':'batch counts pending audit'}`:label);
    setReading($(kind+'-value'),row?kind==='evaluation'?percent(evaluationValue(row)):batchAvailable(row)?number(row.batches):'—':'—');
    if(kind==='evaluation'){
      const chart=$(kind+'-chart');
      chart.querySelectorAll('.plot-point').forEach(point=>point.classList.toggle('is-current',row&&point.dataset.record===recordKey(row)));
      chart.querySelectorAll('.epoch-tick').forEach(tick=>tick.classList.toggle('is-current',Boolean(row&&tick.dataset.epoch===epochNumber({id:row.epoch_id}))));
    }
    if(kind==='batch')$(kind+'-chart').querySelectorAll('.epoch-tick').forEach(tick=>tick.classList.toggle('is-current',Boolean(row&&tick.dataset.record===recordKey(row))));
    $(kind+'-value').setAttribute('aria-label',row?kind==='evaluation'?`${label}, performance ${percent(evaluationValue(row))}`:`${label}, ${batchAvailable(row)?number(row.batches)+' submitted batches':'batch counts pending audit'}`:'Measurement unavailable');
    const previous=kind==='evaluation'?(rows[index-1]&&cohortKey(rows[index-1])===cohortKey(row)?rows[index-1]:null):row?.finalized?rows.slice(0,index).findLast(item=>item.finalized):null;
    const delta=row&&previous&&(kind!=='batch'||batchAvailable(row))?kind==='evaluation'?100*(evaluationValue(row)-evaluationValue(previous)):row.batches-previous.batches:null;
    setChange(kind,delta,kind==='evaluation'?'percentage points':'batches',previous?kind==='evaluation'?`the previous comparable evaluation (${utc(previous.timestamp)} UTC)`:epochName(previous):'');
    if(kind==='evaluation')syncEvaluationLegend(rows,row);else syncBatchLegend(row);
  }
  function bindInspection(kind, rows, coordinates, plot, descriptions) {
    const node = $(kind+'-chart'), wrap = $(kind+'-wrap'), tip = $(kind+'-tip');
    const old = controllers.get(kind), selection = old?.selected;
    const active = node.dataset.focused;
    const overlay = svg('g',{class:'chart-selection',visibility:'hidden'});
    const guide = svg('line',{y1:plot.top,y2:plot.top+plot.height,stroke:themeColor('guide'),'stroke-dasharray':'3 4'});
    const horizontal=kind==='evaluation'?svg('line',{x1:plot.left,x2:plot.left+plot.width,stroke:themeColor('line'),'stroke-dasharray':'3 5'}):null;
    const marker = svg('circle',{r:5,fill:themeColor('bg'),stroke:kind==='evaluation'?colors.performance:colors.batch,'stroke-width':2});
    if(horizontal)overlay.append(horizontal);
    overlay.append(guide,marker);
    const hits = [];
    let hoverPaused=false;
    const controller = {
      selected:null,
      dismiss(){hoverPaused=true;this.hide();$(kind+'-slider').focus({preventScroll:true});},
      hide(){
        const pinned=rows.findIndex(row=>recordKey(row)===chartStates.get(kind)?.selected);
        if(pinned>=0){this.show(pinned,null,false);return;}
        hideTip(tip);overlay.setAttribute('visibility','hidden');node.classList.remove('inspecting');node.querySelectorAll('.bar-stack').forEach(item=>item.classList.remove('is-selected'));this.selected=null;
        syncChartControl(kind,defaultRecord(kind,rows));
      },
      select(index,showTip=false,pointer=null){
        if(!rows[index])return;
        hoverPaused=false;
        chartStates.get(kind).selected=recordKey(rows[index]);this.show(index,pointer,showTip);
        syncViewLink();
      },
      show(index,pointer=null,showTip=true){
        const point = coordinates[index];
        if (!point) return;
        const key = String(rows[index].run_id||rows[index].id);
        if (this.selected!==key || tip.hidden) tooltipContent(tip,descriptions[index],()=>controller.dismiss());
        this.selected=key;showTip?tip.hidden=false:hideTip(tip);syncChartControl(kind,index);
        tip.dataset.interactive=chartStates.get(kind)?.selected===key?'true':'false';
        guide.setAttribute('x1',point.x);guide.setAttribute('x2',point.x);
        marker.setAttribute('visibility',point.unavailable?'hidden':'visible');
        marker.setAttribute('cx',point.x);marker.setAttribute('cy',point.y);
        if(horizontal){horizontal.setAttribute('y1',point.y);horizontal.setAttribute('y2',point.y);}
        node.classList.add('inspecting');
        node.querySelectorAll('.bar-stack').forEach(item=>item.classList.toggle('is-selected',item.dataset.epoch===key));
        overlay.setAttribute('visibility','visible');positionTip(tip,wrap,point.x,point.y,pointer);
      }
    };
    rows.forEach((row,index) => {
      const point = coordinates[index];
      const left = index===0?plot.left:(coordinates[index-1].x+point.x)/2;
      const right = index===rows.length-1?plot.left+plot.width:(point.x+coordinates[index+1].x)/2;
      const key = String(row.run_id||row.id);
      const hit = svg('rect',{x:left,y:plot.top,width:Math.max(1,right-left),height:plot.height,class:'chart-hit',role:'button',tabindex:key===active||(!active&&index===rows.length-1)?0:-1,'aria-label':descriptions[index],'data-record':key});
      hit.addEventListener('pointerenter',event=>{if(event.pointerType!=='touch'&&!hoverPaused)controller.show(index);});
      hit.addEventListener('pointerleave',event=>{
        if(event.pointerType!=='touch'&&document.activeElement!==hit&&!tip.contains(event.relatedTarget))controller.hide();
      });
      hit.addEventListener('click',event=>{if(event.pointerType==='touch')inspectAt(event);else controller.select(index,true);});
      hit.addEventListener('focus',()=>{node.dataset.focused=key;hits.forEach(item=>item.setAttribute('tabindex',item===hit?0:-1));controller.select(index,true);});
      hit.addEventListener('blur',event=>{delete node.dataset.focused;if(!tip.contains(event.relatedTarget))controller.hide();});
      hit.addEventListener('keydown',event=>{
        let next=index;
        if(event.key==='ArrowLeft'||event.key==='ArrowDown') next=Math.max(0,index-1);
        else if(event.key==='ArrowRight'||event.key==='ArrowUp') next=Math.min(rows.length-1,index+1);
        else if(event.key==='Home') next=0;
        else if(event.key==='End') next=rows.length-1;
        else if(event.key==='Escape'){event.preventDefault();controller.hide();return;}
        else if(event.key==='Enter'||event.key===' '){event.preventDefault();controller.show(index);return;}
        else return;
        event.preventDefault();hits.forEach((item,i)=>item.setAttribute('tabindex',i===next?0:-1));hits[next].focus();
      });
      hits.push(hit);node.append(hit);
    });
    node.append(overlay);controllers.set(kind,controller);
    wrap.onpointerleave=event=>{if(event.pointerType!=='touch'&&!wrap.contains(document.activeElement))controller.hide();};
    let touching=false;
    const inspectAt=event=>{
      const box=node.getBoundingClientRect(),x=event.clientX-box.left;
      let nearest=0;
      for(let i=1;i<coordinates.length;i++) if(Math.abs(coordinates[i].x-x)<Math.abs(coordinates[nearest].x-x)) nearest=i;
      controller.select(nearest,true,event.clientY-box.top);
    };
    // pan-y keeps normal page scrolling; horizontal scrubbing selects measurements.
    node.onpointerdown=event=>{if(event.pointerType==='touch'&&rows.length){touching=true;tip.dataset.touching='true';inspectAt(event);}};
    node.onpointermove=event=>{
      if(touching&&rows.length)inspectAt(event);
      else if(hoverPaused&&event.pointerType!=='touch'&&(event.movementX||event.movementY))hoverPaused=false;
    };
    node.onpointerup=node.onpointercancel=()=>{touching=false;setTimeout(()=>delete tip.dataset.touching,250);};
    const pinned=rows.findIndex(row=>recordKey(row)===chartStates.get(kind)?.selected);
    if(pinned>=0)controller.show(pinned,null,!tip.hidden);
    else if (selection) {const index=rows.findIndex(row=>recordKey(row)===selection);if(index>=0)controller.show(index);}
    if (active) {const hit=hits.find(item=>item.dataset.record===active);if(hit)hit.focus({preventScroll:true});}
  }
  function drawChart(kind, rows) {
    const node=$(kind+'-chart'),tip=$(kind+'-tip');
    const state=chartStates.get(kind)||{selected:null};
    const selectedEpoch=kind==='evaluation'?state.rows?.find(row=>recordKey(row)===state.selected)?.epoch_id:null;
    state.rows=rows;
    if(!state.initialized&&rows.length){
      const requested=requestedView.get(kind==='evaluation'?'performance':'batches');
      let row=rows.find(item=>(kind==='evaluation'?String(item.timestamp):epochNumber(item))===requested);
      if(!row&&kind==='evaluation'){const original=data?.evaluations.find(item=>String(item.timestamp)===requested);row=rows.find(item=>item.epoch_id===original?.epoch_id);}
      if(row)state.selected=recordKey(row);state.initialized=true;
    }
    if(state.selected&&!rows.some(row=>recordKey(row)===state.selected)){const replacement=selectedEpoch&&rows.find(row=>row.epoch_id===selectedEpoch);state.selected=replacement?recordKey(replacement):null;}
    chartStates.set(kind,state);
    const chosen=rows.findIndex(row=>recordKey(row)===state.selected);
    syncChartControl(kind,chosen>=0?chosen:defaultRecord(kind,rows));
    const width=node.clientWidth,height=node.clientHeight;
    if (!width || !height) return;
    const drawKey=JSON.stringify([width,height,rows,kind==='evaluation'?evaluationView:batchView]);
    if(node.dataset.drawKey===drawKey)return;
    node.dataset.drawKey=drawKey;
    const focused=node.dataset.focused,replaying=node.classList.contains('animate-in');
    node.classList.remove('animate-in');
    node.replaceChildren();hideTip(tip);
    if(focused)node.dataset.focused=focused;
    const plot={left:width<500?38:46,top:12,width:width-(width<500?38:46)-10,height:height-64};
    node.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const performance=kind==='evaluation';
    const high=performance?1:niceCeiling(Math.max(0,...rows.map(row=>row.batches)));
    axis(node,plot,high,performance);
    $(kind+'-empty').hidden=rows.length>0;
    $(kind+'-empty').textContent=performance?'No completed evaluations yet.':'No epoch records yet.';
    node.setAttribute('aria-label',`${performance?(evaluationView==='results'?'Solved and not solved results by epoch':'Performance by epoch, with markers when the evaluation setup changes'):batchView==='trend'?'Submitted, training-included and excluded batches by epoch':'Batch outcomes by epoch'}, ${rows.length} records. Use left and right arrow keys to inspect.`);
    const coordinates=[];
    if(performance) {
      const groups=[];
      rows.forEach((row,index)=>{
        const epoch=epochNumber({id:row.epoch_id});
        if(groups.at(-1)?.epoch===epoch)groups.at(-1).indices.push(index);
        else groups.push({epoch,indices:[index]});
      });
      const epochWidth=plot.width/Math.max(1,groups.length);
      groups.forEach((group,epochIndex)=>{
        group.x=plot.left+epochWidth*(epochIndex+.5);
        group.indices.forEach((index,position)=>{
          const offset=group.indices.length>1?(position/(group.indices.length-1)-.5)*epochWidth*.4:0;
          coordinates[index]={x:group.x+offset,y:plot.top+plot.height*(1-evaluationValue(rows[index]))};
        });
      });
      if(rows.length) {
        const base=plot.top+plot.height;
        if(evaluationView==='trend'){
          const defs=svg('defs'),gradient=svg('linearGradient',{id:'performance-fill',x1:0,y1:0,x2:0,y2:1});
          gradient.append(svg('stop',{offset:'0%','stop-color':colors.performance,'stop-opacity':.2}),svg('stop',{offset:'100%','stop-color':colors.performance,'stop-opacity':.025}));
          defs.append(gradient);node.append(defs);
          if(coordinates.length>1){
            node.append(svg('polygon',{points:[`${coordinates[0].x},${base}`,...coordinates.map(point=>`${point.x},${point.y}`),`${coordinates.at(-1).x},${base}`].join(' '),fill:'url(#performance-fill)',class:'plot-area'}));
            node.append(svg('polyline',{points:coordinates.map(point=>`${point.x},${point.y}`).join(' '),class:'plot-line',pathLength:1}));
          }
          rows.forEach((row,index)=>{
            if(row.setup_changed){const x=(coordinates[index-1].x+coordinates[index].x)/2;node.append(svg('line',{x1:x,x2:x,y1:plot.top,y2:base,stroke:themeColor('guide'),'stroke-dasharray':'3 5',class:'setup-boundary'}));}
          });
          coordinates.forEach((point,index)=>node.append(svg('circle',{cx:point.x,cy:point.y,r:3,class:'plot-point','data-record':recordKey(rows[index])})));
        }else{
          rows.forEach((row,index)=>{
            const point=coordinates[index];
            const barWidth=Math.max(.5,epochWidth-Math.min(2,epochWidth*.08));
            const x=Math.max(plot.left,Math.min(plot.left+plot.width-barWidth,point.x-barWidth/2));
            const solved=plot.height*row.successes/row.count;
            const stack=svg('g',{class:'bar-stack evaluation-stack','data-epoch':recordKey(row)});
            stack.append(svg('rect',{x,y:base-solved,width:barWidth,height:solved,fill:colors.performance,'data-outcome':'solved','data-count':row.successes}));
            stack.append(svg('rect',{x,y:plot.top,width:barWidth,height:plot.height-solved,fill:colors.unchecked,'data-outcome':'not-solved','data-count':row.count-row.successes}));
            node.append(stack);
          });
        }
        const tickStride=Math.max(1,Math.ceil(groups.length/Math.max(1,Math.floor(plot.width/40))));
        groups.forEach((group,index)=>{
          if(index%tickStride===0||index===groups.length-1)node.append(svg('text',{x:group.x,y:base+25,'text-anchor':'middle',class:'epoch-tick','data-epoch':group.epoch},group.epoch));
        });
        node.append(svg('text',{x:plot.left+plot.width/2,y:base+42,'text-anchor':'middle',class:'time-tick'},'Epoch'));
      }
    } else {
      const step=plot.width/Math.max(1,rows.length),barWidth=Math.max(.5,step-Math.min(2,step*.08));
      rows.forEach((row,index)=>{
        const x=plot.left+step*(index+.5),base=plot.top+plot.height;
        const y=base-plot.height*row.batches/high;
        const available=batchAvailable(row);
        coordinates.push({x,y:available?y:plot.top+plot.height/2,unavailable:!available});
        if (batchView==='bars'&&available&&row.batches>0) {
          const stack=svg('g',{class:'bar-stack','data-epoch':row.id});
          const segments=validEligibility(row)?[{key:'eligible',value:row.learner_eligible,color:colors.performance},{key:'excluded',value:row.learner_excluded,color:colors.rejected}]:row.audit_breakdown_available!==false&&validOutcomes(row)?['accepted','unchecked','rejected'].map(key=>({key,value:row[key],color:key==='accepted'?colors.performance:colors[key]})):[{key:'unclassified',value:row.batches,color:colors.batch}];
          let cumulative=0;
          for(const segment of segments) {
            const height=plot.height*segment.value/high;
            stack.append(svg('rect',{x:x-barWidth/2,y:base-plot.height*(cumulative+segment.value)/high,width:barWidth,height,fill:segment.color,'data-outcome':segment.key,'data-count':segment.value}));
            cumulative+=segment.value;
          }
          stack.append(svg('line',{x1:x-barWidth/2,x2:x+barWidth/2,y1:y,y2:y,stroke:colors.batch,'stroke-width':1.5,class:'submitted-cap','data-count':row.batches}));
          node.append(stack);
        } else if(batchView==='bars'&&available&&row.batches===0&&!row.finalized) {
          node.append(svg('circle',{cx:x,cy:base,r:3,fill:themeColor('bg'),stroke:themeColor('muted'),'stroke-width':1.5}));
        }
        if(!available) {
          const marker=svg('circle',{cx:x,cy:base,r:3,fill:themeColor('bg'),stroke:themeColor('muted'),'stroke-width':1.5,class:'pending-marker'});
          marker.append(svg('title',{},`Epoch ${epochNumber(row)}: batch count pending`));node.append(marker);
        }
        if(rows.length<=8||width>650||index%2===0||index===rows.length-1) node.append(svg('text',{x,y:base+28,'text-anchor':'middle',class:'epoch-tick','data-record':recordKey(row)},epochNumber(row)));
      });
    }
    if(!performance&&rows.length)node.append(svg('text',{x:plot.left+plot.width/2,y:plot.top+plot.height+44,'text-anchor':'middle',class:'time-tick'},'Epoch'));
    if(!performance&&batchView==='trend'){
      const base=plot.top+plot.height,defs=svg('defs'),gradient=svg('linearGradient',{id:'batch-fill',x1:0,y1:0,x2:0,y2:1});
      gradient.append(svg('stop',{offset:'0%','stop-color':colors.batch,'stop-opacity':.2}),svg('stop',{offset:'100%','stop-color':colors.batch,'stop-opacity':.025}));defs.append(gradient);node.append(defs);
      const series=[];rows.forEach((row,index)=>{if(batchAvailable(row)){if(!series.length||index!==series.at(-1).at(-1)+1)series.push([]);series.at(-1).push(index);}});
      for(const indices of series){const points=indices.map(index=>coordinates[index]);if(points.length>1){
        node.append(svg('polygon',{points:[`${points[0].x},${base}`,...points.map(p=>`${p.x},${p.y}`),`${points.at(-1).x},${base}`].join(' '),fill:'url(#batch-fill)',class:'plot-area'}));
        node.append(svg('polyline',{points:points.map(p=>`${p.x},${p.y}`).join(' '),class:'plot-line',pathLength:1}));
      }}
      for(const key of ['learner_eligible','learner_excluded']){
        const series=[];rows.forEach((row,index)=>{if(batchAvailable(row)&&validEligibility(row)){if(!series.length||index!==series.at(-1).at(-1)+1)series.push([]);series.at(-1).push(index);}});
        for(const indices of series)if(indices.length>1)node.append(svg('polyline',{points:indices.map(index=>`${coordinates[index].x},${base-plot.height*rows[index][key]/high}`).join(' '),class:'plot-line '+(key==='learner_eligible'?'eligible-line':'excluded-line'),pathLength:1}));
      }
      coordinates.forEach((p,index)=>{if(!p.unavailable)node.append(svg('circle',{cx:p.x,cy:p.y,r:3,class:'plot-point','data-record':recordKey(rows[index])}));});
    }
    bindInspection(kind,rows,coordinates,plot,rows.map(performance?evaluationDetails:batchDetails));
    const highlighted=rows.findIndex(row=>recordKey(row)===state.selected);
    syncChartControl(kind,highlighted>=0?highlighted:defaultRecord(kind,rows));
    if(replaying)replayEntrance(node);else animateOnce(kind,node,rows.length>0);
  }
  function validGrid(row) {
    return Array.isArray(row.grid)&&row.grid.length===256&&row.grid.every(count);
  }
  let minerView=requestedView.get('miners')==='bars'?'bars':'grid';
  let minerLayoutKey='';
  let minerOrder=[];
  function minerOutcomes(row){
    const values=row?.grid_outcomes;
    if(!values||!['verified','rejected','unchecked'].every(key=>Array.isArray(values[key])&&values[key].length===256&&values[key].every(count)))return null;
    return row.grid.every((total,uid)=>values.verified[uid]+values.rejected[uid]+values.unchecked[uid]===total)?values:null;
  }
  function minerDetails(row,uid){
    const values=minerOutcomes(row),total=row.grid[uid];
    return `UID ${String(uid).padStart(3,'0')}\n${number(total)} submitted ${total===1?'batch':'batches'}${values?`\n${number(values.verified[uid])} verified · ${number(values.rejected[uid])} rejected\n${number(values.unchecked[uid])} unchecked`:''}\n${epochName(row)} · ${row.finalized?'finalized':row.phase||'unfinalized'}`;
  }
  function layoutMiners(row,animate=false) {
    const grid=$('miner-grid'),axes=$('miner-bar-axes');
    const cells=Array.from(grid.children);
    const wrap=grid.parentElement,previousHeight=wrap.getBoundingClientRect().height;
    const before=animate&&!reducedMotion.matches?cells.map(cell=>cell.getBoundingClientRect()):null;
    grid.dataset.view=minerView;
    const width=grid.clientWidth,height=grid.clientHeight;
    const key=JSON.stringify([minerView,width,height,row?.id,row?.grid,row?.grid_outcomes]);
    if(key===minerLayoutKey)return;
    minerLayoutKey=key;
    wrap.getAnimations().filter(animation=>animation.id==='miner-resize').forEach(animation=>animation.cancel());

    cells.forEach(cell=>cell.getAnimations().filter(animation=>animation.id==='miner-morph').forEach(animation=>animation.cancel()));
    grid.dataset.view=minerView;
    axes.toggleAttribute('hidden',minerView!=='bars'||!row);
    $('miner-grid-legend').hidden=minerView==='bars';
    $('miner-bar-legend').hidden=minerView!=='bars';
    if(!row)$('miner-bar-legend').replaceChildren();
    for(const view of ['grid','bars'])$("miner-view-"+view).setAttribute('aria-pressed',String(minerView===view));
    minerOrder=row?row.grid.map((batches,uid)=>({batches,uid})).filter(item=>item.batches>0).sort((a,b)=>b.batches-a.batches||a.uid-b.uid).map(item=>item.uid):[];
    cells.forEach((cell,uid)=>cell.hidden=minerView==='bars'&&row?.grid[uid]===0);
    const remembered=cells.find(cell=>cell.tabIndex===0);
    if(minerView==='bars'&&remembered?.hidden&&minerOrder.length)cells.forEach((cell,uid)=>cell.tabIndex=uid===minerOrder[0]?0:-1);
    if(row){$('miner-empty').hidden=minerView!=='bars'||minerOrder.length>0;$('miner-empty').textContent='No batches submitted in this epoch.';}
    if(minerView==='bars'&&row){
      const left=38,top=18,base=height-42,plotHeight=base-top,step=(width-left-6)/Math.max(1,minerOrder.length);
      const maximum=Math.max(1,...row.grid);
      const high=maximum<=4?maximum:niceCeiling(maximum);
      axes.replaceChildren();axes.style.height=`${height}px`;axes.setAttribute('viewBox',`0 0 ${width} ${height}`);
      for(const fraction of high<=4?Array.from({length:high+1},(_,index)=>index/high):[0,.5,1]){
        const y=base-plotHeight*fraction;
        axes.append(svg('line',{x1:left,x2:width-6,y1:y,y2:y,stroke:fraction?themeColor('line'):themeColor('axis'),...(fraction?{'stroke-dasharray':'2 6'}:{})}));
        axes.append(svg('text',{x:left-8,y,'text-anchor':'end','dominant-baseline':'middle'},number(high*fraction)));
      }
      const ranks=[...new Set([0,Math.round((minerOrder.length-1)/3),Math.round(2*(minerOrder.length-1)/3),minerOrder.length-1])].filter(rank=>rank>=0&&minerOrder.length);
      for(const rank of ranks)axes.append(svg('text',{x:left+step*(rank+.5),y:base+25,'text-anchor':rank===0?'start':rank===minerOrder.length-1?'end':'middle'},String(rank+1)));
      axes.append(svg('text',{x:left+(width-left-6)/2,y:base+40,'text-anchor':'middle',class:'time-tick'},'Rank'));
      const outcomes=minerOutcomes(row),legend=$('miner-bar-legend');legend.replaceChildren();
      if(outcomes){
        for(const key of ['verified','rejected','unchecked']){
          const item=document.createElement('span'),swatch=document.createElement('i');swatch.className=`swatch ${key==='verified'?'accepted':key}`;
          item.dataset.outcome=key;const total=outcomes[key].reduce((sum,value)=>sum+value,0),countLabel=document.createElement('b');countLabel.className='legend-value';countLabel.dataset.zero=String(total===0);countLabel.textContent=number(total);
          item.append(swatch,`${key[0].toUpperCase()+key.slice(1)} `,countLabel);legend.append(item);
        }
      }else legend.textContent='Outcome breakdown unavailable';
      minerOrder.forEach((uid,rank)=>{
        const cell=cells[uid],total=row.grid[uid],barHeight=plotHeight*total/high;
        Object.assign(cell.style,{left:`${left+step*rank}px`,top:`${base-barHeight}px`,width:`${Math.max(.5,step*.82)}px`,height:`${barHeight}px`});
        cell.dataset.rank=rank;
        if(outcomes){
          const verified=100*outcomes.verified[uid]/total,unchecked=verified+100*outcomes.unchecked[uid]/total;
          const verifiedColor=themeColor(['zero','one','two','accepted'][Math.min(3,outcomes.verified[uid])]);
          cell.style.background=`linear-gradient(to top,${verifiedColor} 0%,${verifiedColor} ${verified}%,var(--unchecked) ${verified}%,var(--unchecked) ${unchecked}%,var(--rejected) ${unchecked}%,var(--rejected) 100%)`;
        }else cell.style.background=themeColor(['zero','one','two','accepted'][Math.min(3,total)]);
      });
    }else cells.forEach(cell=>{for(const property of ['left','top','width','height','background'])cell.style.removeProperty(property);delete cell.dataset.rank;});
    if(before)cells.forEach((cell,index)=>{
      const first=before[index],last=cell.getBoundingClientRect();
      if(!last.width||!last.height)return;
      const frames=first.width&&first.height?[{transform:`translate(${first.x-last.x}px,${first.y-last.y}px) scale(${first.width/last.width},${first.height/last.height})`},{transform:'none'}]:[{opacity:0},{opacity:1}];
      const animation=cell.animate(frames,{duration:440,easing:'cubic-bezier(.22,1,.36,1)'});
      animation.id='miner-morph';
    });
    if(before&&Math.abs(previousHeight-height)>1){
      wrap.style.overflow='hidden';
      const animation=wrap.animate([{height:`${previousHeight}px`},{height:`${height}px`}],{duration:440,easing:'cubic-bezier(.22,1,.36,1)'});
      animation.id='miner-resize';
      const cleanup=()=>{if(!wrap.getAnimations().some(item=>item.id==='miner-resize'&&item!==animation))wrap.style.removeProperty('overflow');};
      animation.onfinish=animation.oncancel=cleanup;
    }
    if(row)grid.setAttribute('aria-label',`Submitted batches by UID, ${epochName(row)}. ${minerView==='bars'?'Bar chart sorted by submitted batches, highest first. Left and right arrows navigate ranks.':'Arrow keys navigate the 16 by 16 grid.'}`);
  }
  for(const view of ['grid','bars'])$('miner-view-'+view).addEventListener('click',()=>{
    if(minerView===view)return;
    minerView=view;
    const row=minerEpochs.find(row=>row.id===$('miner-grid').dataset.epoch);
    const controller=controllers.get('miner');
    if(controller){controller.hoverPaused=true;controller.hide();}
    const grid=$('miner-grid');observer?.unobserve(grid);
    layoutMiners(row);replayEntrance(grid);syncViewLink();
  });

  function drawMiners(epochs) {
    // Prefer the current epoch once it has recorded work; otherwise label the
    // latest epoch with submissions. Never substitute illustrative miners.
    const rows=epochs.filter(validGrid);
    minerEpochs=epochs;
    if(!minerInitialized&&rows.length){
      const row=rows.find(item=>epochNumber(item)===requestedView.get('epoch'));
      if(row)selectedMinerEpoch=row.id;minerInitialized=true;
    }
    const row=rows.find(item=>item.id===selectedMinerEpoch)||rows.filter(item=>item.grid.some(value=>value>0)).at(-1)||rows.at(-1);
    if(selectedMinerEpoch&&!rows.some(item=>item.id===selectedMinerEpoch))selectedMinerEpoch=null;
    const slider=$('miner-epoch-slider'),index=row?rows.indexOf(row):0;
    slider.max=Math.max(1,rows.length-1);slider.value=index;slider.disabled=rows.length<2;
    syncNavigation('miner-epoch',index,rows.length);
    slider.style.setProperty('--progress',`${rows.length>1?100*index/(rows.length-1):0}%`);
    $('miner-epoch-label').textContent=row?epochName(row):'Epoch —';
    slider.setAttribute('aria-valuetext',row?`${epochName(row)}, ${row.finalized?'finalized':row.phase||'unfinalized'}, ${number(row.grid.filter(value=>value>0).length)} submitting miners`:'No epochs available');
    const grid=$('miner-grid'),tip=$('miner-tip'),wrap=grid.parentElement;
    const active=grid.contains(document.activeElement)?Number(document.activeElement.dataset.uid):null;
    const old=controllers.get('miner'),selected=old&&row&&old.epoch===row.id?old.selected:null;
    const previous=row?.finalized?rows.slice(0,index).findLast(item=>item.finalized):null;
    setChange('miner',previous?row.grid.filter(value=>value>0).length-previous.grid.filter(value=>value>0).length:null,'submitting miners',previous?epochName(previous):'');
    for(const view of ['grid','bars'])$('miner-view-'+view).disabled=!row;
    if(!row) {
      grid.replaceChildren();layoutMiners(null);delete grid.dataset.epoch;setReading($('miner-value'),'—');$('miner-epoch').textContent='No miner records yet';$('miner-empty').textContent='No miner contributions recorded.';$('miner-empty').hidden=false;hideTip(tip);controllers.delete('miner');return;
    }
    $('miner-empty').hidden=true;
    setReading($('miner-value'),number(row.grid.filter(value=>value>0).length));
    $('miner-epoch').textContent=row.finalized?'Finalized':String(row.phase||'Unfinalized').replace(/^./,letter=>letter.toUpperCase());
    const extra=$('miner-extra');extra.hidden=!count(row.unassigned_batches)||row.unassigned_batches===0;
    extra.textContent=extra.hidden?'':`${number(row.unassigned_batches)} batches without UID`;
    const cells=Array.from(grid.children);
    if(cells.length!==256) {
      grid.replaceChildren();
      for(let uid=0;uid<256;uid++) {
        const cell=document.createElement('button');cell.type='button';cell.className='miner-cell';cell.dataset.uid=uid;cell.tabIndex=uid===0?0:-1;
        cell.style.setProperty('--reveal-delay',`${(Math.floor(uid/16)+uid%16)*7}ms`);
        grid.append(cell);
      }
    }
    const buttons=Array.from(grid.children);
    const controller={epoch:row.id,selected:null,hoverPaused:old?.hoverPaused||false,
      hide(){hideTip(tip);buttons.forEach(cell=>cell.classList.remove('selected'));this.selected=null;},
      show(uid){
        const cell=buttons[uid],value=row.grid[uid];
        if(minerView==='bars'&&value===0){this.hide();return;}
        if(this.selected!==uid||tip.hidden)tooltipContent(tip,minerDetails(row,uid));
        this.selected=uid;tip.hidden=false;buttons.forEach(item=>item.classList.toggle('selected',item===cell));
        positionTip(tip,wrap,cell.offsetLeft+cell.offsetWidth/2,cell.offsetTop);
      }
    };
    buttons.forEach((cell,uid)=>{
      const value=row.grid[uid],changed=grid.dataset.epoch===row.id&&cell.dataset.batches!==undefined&&Number(cell.dataset.batches)!==value;
      cell.dataset.level=Math.min(3,value);cell.dataset.batches=value;cell.setAttribute('aria-label',minerDetails(row,uid).replaceAll('\n',', '));
      if(changed&&!reducedMotion.matches&&cell.animate){
        cell.getAnimations().filter(animation=>animation.id==='cell-update').forEach(animation=>animation.cancel());
        const update=cell.animate([{opacity:1},{opacity:.55},{opacity:1}],{duration:420,easing:'ease-in-out'});
        update.id='cell-update';
      }
      cell.onpointerenter=event=>{if(event.pointerType!=='touch'&&!controller.hoverPaused)controller.show(uid);};
      cell.onpointerleave=event=>{if(event.pointerType!=='touch'&&document.activeElement!==cell)controller.hide();};
      cell.onclick=event=>{
        controller.hoverPaused=false;
        if(minerView==='bars'&&event.detail>0){
          inspectBar(event);
          if(controller.selected!==null)buttons[controller.selected].focus({preventScroll:true});
        }else controller.show(uid);
      };cell.onfocus=()=>{buttons.forEach((item,index)=>item.tabIndex=index===uid?0:-1);controller.show(uid);};cell.onblur=()=>controller.hide();
      cell.onkeydown=event=>{
        let next=uid;
        if(minerView==='bars'&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(event.key)){
          const rank=minerOrder.indexOf(uid),step=['ArrowLeft','ArrowUp'].includes(event.key)?-1:1;
          const target=event.key==='Home'?0:event.key==='End'?minerOrder.length-1:Math.max(0,Math.min(minerOrder.length-1,rank+step));
          next=minerOrder[target];event.preventDefault();buttons.forEach((item,index)=>item.tabIndex=index===next?0:-1);buttons[next]?.focus();return;
        }
        if(event.key==='ArrowLeft')next=Math.max(0,uid-1);
        else if(event.key==='ArrowRight')next=Math.min(255,uid+1);
        else if(event.key==='ArrowUp')next=Math.max(0,uid-(minerView==='bars'?1:16));
        else if(event.key==='ArrowDown')next=Math.min(255,uid+(minerView==='bars'?1:16));
        else if(event.key==='Home')next=0;
        else if(event.key==='End')next=255;
        else if(event.key==='Escape'){event.preventDefault();controller.hoverPaused=true;controller.hide();return;}
        else return;
        event.preventDefault();buttons.forEach((item,index)=>item.tabIndex=index===next?0:-1);buttons[next].focus();
      };
    });
    grid.dataset.epoch=row.id;
    layoutMiners(row);
    let scrubbing=false;
    const inspectBar=event=>{
      if(minerView!=='bars')return;
      const box=grid.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;
      if(x<38||x>box.width-6||y<18||y>box.height-42)return;
      const rank=Math.max(0,Math.min(minerOrder.length-1,Math.floor((x-38)/(box.width-44)*minerOrder.length)));
      if(minerOrder.length)controller.show(minerOrder[rank]);
    };
    grid.onpointerdown=event=>{controller.hoverPaused=false;if(minerView==='bars'){scrubbing=true;inspectBar(event);}};
    grid.onpointermove=event=>{if(event.movementX||event.movementY)controller.hoverPaused=false;if(!controller.hoverPaused&&(event.pointerType!=='touch'||scrubbing))inspectBar(event);};
    grid.onpointerup=grid.onpointercancel=()=>{scrubbing=false;};
    grid.onpointerleave=()=>{scrubbing=false;if(minerView==='bars'&&!grid.contains(document.activeElement))controller.hide();};
    controllers.set('miner',controller);
    if(active!==null)buttons[active].focus({preventScroll:true});
    if(selected!==null&&selected!==undefined)controller.show(selected);else controller.hide();
    if(minerView==='grid')animateOnce('miner',grid,true);
    syncViewLink();
  }
  let livePrices=null,marketStream=null,marketFetchPending=false;
  const currentPrices=()=>livePrices||data?.incentive_prices;
  function acceptMarketPrices(quote){
    if(!quote||quote.source!=='taomarketcap')return;
    const next={...livePrices,source:'taomarketcap'};
    for(const key of ['alpha_tao','tao_usd']){
      const at=key+'_at';
      if(finite(quote[key])&&quote[key]>0&&finite(quote[at])&&quote[at]>0&&(!finite(next[at])||quote[at]>=next[at])){
        next[key]=quote[key];next[at]=quote[at];
        if(key==='tao_usd')next.tao_usd_source=quote.tao_usd_source==='kraken'?'kraken':'taomarketcap';
      }
    }
    if(!['alpha_tao','tao_usd','alpha_tao_at','tao_usd_at'].every(key=>finite(next[key])&&next[key]>0))return;
    livePrices=next;
    window.dispatchEvent(new CustomEvent('affine:prices',{detail:next}));
    renderMarketPrices();if(data)drawIncentive();
  }
  async function refreshMarketPrices(){
    if(document.hidden||marketFetchPending)return;
    marketFetchPending=true;
    const abort=new AbortController(),timeout=setTimeout(()=>abort.abort(),6000);
    try{const response=await fetch('/api/v1/market/prices',{cache:'no-store',signal:abort.signal});if(response.ok)acceptMarketPrices(await response.json());}catch{}finally{clearTimeout(timeout);marketFetchPending=false;marketInitialFetchComplete=true;renderMarketPrices();}
  }
  function connectMarketPrices(){
    if(document.hidden||marketStream||!window.EventSource)return;
    marketStream=new EventSource('/api/v1/market/prices/stream');
    marketStream.addEventListener('prices',event=>{try{acceptMarketPrices(JSON.parse(event.data));}catch{}});
  }
  const lastPriceKey='affine:last-seen-prices:v1';
  const validSeenPrices=prices=>prices&&['tao','alphaUsd','alphaTao'].every(key=>finite(prices[key])&&prices[key]>0&&prices[key]<1e11);
  let lastDisplayedPrices=null,priceReplay=null,priceArrivalPending=false,marketInitialFetchComplete=false;
  try{const saved=JSON.parse(localStorage.getItem(lastPriceKey));if(validSeenPrices(saved)){lastDisplayedPrices=saved;priceArrivalPending=true;}}catch{}
  const marketDollars=value=>value.toLocaleString('en-US',{style:'currency',currency:'USD',minimumFractionDigits:value<1?4:2,maximumFractionDigits:value<1?4:2});
  const alphaDollars=value=>value.toLocaleString('en-US',{style:'currency',currency:'USD',minimumFractionDigits:4,maximumFractionDigits:4});
  function saveSeenPrices(){if(validSeenPrices(lastDisplayedPrices))try{localStorage.setItem(lastPriceKey,JSON.stringify(lastDisplayedPrices));}catch{}}
  function paintMarketPrices(values){
    marketReading($('tao-price'),values.tao===null?'—':marketDollars(values.tao),values.tao);
    marketReading($('alpha-price'),values.alphaUsd===null?'—':alphaDollars(values.alphaUsd),values.alphaUsd);
    marketReading($('alpha-price-tao'),values.alphaTao===null?'':`${values.alphaTao.toLocaleString('en-GB',{minimumFractionDigits:8,maximumFractionDigits:8})} TAO`,values.alphaTao);
    if(validSeenPrices(values))lastDisplayedPrices={...values};
  }
  function pausePriceReplay(){
    if(priceReplay){cancelAnimationFrame(priceReplay.frame);priceReplay=null;}
    saveSeenPrices();
  }
  function showMarketPrices(values){
    if(priceArrivalPending){
      if(!validSeenPrices(values)||(!livePrices&&!marketInitialFetchComplete))return;
      priceArrivalPending=false;
      if(!reducedMotion.matches&&validSeenPrices(lastDisplayedPrices)){
        const from={...lastDisplayedPrices},started=performance.now();
        if(['tao','alphaUsd','alphaTao'].some(key=>from[key]!==values[key])){
          priceReplay={target:values,frame:null};
          let painted=-Infinity;
          const advance=now=>{
            if(!priceReplay)return;
            if(document.hidden){pausePriceReplay();return;}
            const progress=Math.max(0,Math.min(1,(now-started)/1400)),target=priceReplay.target;
            if(progress>=1||reducedMotion.matches){priceReplay=null;paintMarketPrices(target);saveSeenPrices();return;}
            if(now-painted>=40){const amount=1-(1-progress)**3;paintMarketPrices(Object.fromEntries(['tao','alphaUsd','alphaTao'].map(key=>[key,from[key]+(target[key]-from[key])*amount])));painted=now;}
            priceReplay.frame=requestAnimationFrame(advance);
          };
          priceReplay.frame=requestAnimationFrame(advance);return;
        }
      }
    }
    if(priceReplay){if(validSeenPrices(values))priceReplay.target=values;return;}
    paintMarketPrices(values);saveSeenPrices();
  }
  const marketValues=new WeakMap();
  function marketReading(node,text,value){
    const previous=node.textContent,previousValue=marketValues.get(node);
    marketValues.set(node,value);
    if(previous===text)return;
    const direction=finite(value)&&finite(previousValue)?value>previousValue?'up':value<previousValue?'down':null:null;
    const tone=direction==='up'?'#69d6a4':'#f0808b',restingColor=getComputedStyle(node).color;
    const characters=Array.from(text),old=Array.from(previous),offset=old.length-characters.length;
    node.replaceChildren(...characters.map((character,index)=>{
      if(!direction||!/[0-9]/.test(character)||character===old[index+offset])return document.createTextNode(character);
      const digit=document.createElement('span');digit.className='price-digit';digit.dataset.direction=direction;digit.textContent=character;digit.style.color=tone;
      if(reducedMotion.matches)setTimeout(()=>digit.style.removeProperty('color'),1100);
      else{const animation=digit.animate([{color:tone,offset:0},{color:tone,offset:.35},{color:restingColor,offset:1}],{duration:1100,easing:'ease-out'});animation.onfinish=()=>digit.style.removeProperty('color');}
      return digit;
    }));
  }
  function renderMarketPrices(){
    const prices=currentPrices(),now=Date.now()/1000;
    const tao=prices&&finite(prices.tao_usd)&&prices.tao_usd>0&&finite(prices.tao_usd_at)?prices.tao_usd:null;
    const alphaTao=prices&&finite(prices.alpha_tao)&&prices.alpha_tao>0&&finite(prices.alpha_tao_at)?prices.alpha_tao:null;
    const alphaUsd=tao!==null&&alphaTao!==null?tao*alphaTao:null;
    const dollars=marketDollars;
    const taoNode=$('tao-price'),alphaNode=$('alpha-price');
    showMarketPrices({tao,alphaUsd,alphaTao});
    const provider=prices?.source==='taomarketcap'?'TaoMarketCap':'CoinGecko',usdProvider=prices?.tao_usd_source==='kraken'?'Kraken':provider,maxAge=prices?.source==='taomarketcap'?180:1800;
    taoNode.setAttribute('aria-label',tao===null?'TAO price unavailable':`TAO price ${dollars(tao)} USD`);
    alphaNode.setAttribute('aria-label',alphaUsd===null?'Subnet 120 alpha USD price unavailable':`Subnet 120 alpha price ${alphaDollars(alphaUsd)} USD`);
    taoNode.title=tao===null?'Awaiting market quote':`TAO / USD · ${usdProvider} · ${utc(prices.tao_usd_at)} UTC`;
    alphaNode.title=alphaUsd===null?'Awaiting market quote':`Subnet 120 · ${provider}${usdProvider!==provider?' + '+usdProvider:''} · ${utc(Math.min(prices.alpha_tao_at,prices.tao_usd_at))} UTC`;

    $('alpha-price-tao').title=alphaTao===null?'':`SN120 / TAO · ${prices.source==='taomarketcap'?'TaoMarketCap':'Pool reserves'} · ${utc(prices.alpha_tao_at)} UTC`;
    const taoStale=tao!==null&&Math.abs(now-prices.tao_usd_at)>maxAge;
    const alphaStale=(alphaUsd!==null||alphaTao!==null)&&Math.abs(now-(alphaUsd===null?prices.alpha_tao_at:Math.min(prices.alpha_tao_at,prices.tao_usd_at)))>maxAge;
    $('tao-price-age').hidden=true;$('alpha-price-age').hidden=true;
    if(taoStale)taoNode.title+=' · Last available quote';
    if(alphaStale)alphaNode.title+=' · Last available quote';

  }
  function status() {
    renderMarketPrices();
    const connection=$('connection'),updated=data?.summary?.updated_at;
    const stale=finite(updated)&&Date.now()/1000-updated>120;
    connection.dataset.state=unavailable?'unavailable':stale?'stale':'current';
    const desktop=matchMedia('(min-width:1000px)').matches;
    const fullMessage=unavailable?finite(updated)?`Update delayed · showing ${utc(updated)} UTC`:'Connection unavailable · retrying':finite(updated)?`${stale?'Snapshot may be stale · ':''}Updated ${utc(updated)} UTC`:'Update time unavailable';
    const message=desktop?(finite(updated)?`Network data · ${unavailable?'Delayed · ':stale?'Stale · ':''}${utc(updated)} UTC`:unavailable?'Network data · Unavailable · retrying':'Network data · time unavailable'):fullMessage;
    if(desktop){const description=fullMessage+'. Network chart data; market prices update separately.';connection.title=description;connection.setAttribute('aria-label',description);}else{connection.removeAttribute('title');connection.removeAttribute('aria-label');}
    if(connection.textContent!==message){
      if(finite(updated)){
        const timestamp=`${utc(updated)} UTC`,stamp=document.createElement('span');stamp.className='footer-time';stamp.textContent=timestamp;
        const prefix=desktop?document.createElement('span'):document.createTextNode('');if(desktop)prefix.className='footer-state';prefix.textContent=message.slice(0,-timestamp.length);connection.replaceChildren(prefix,stamp);
      }else if(desktop){const prefix=document.createElement('span');prefix.className='footer-state';prefix.textContent=message;connection.replaceChildren(prefix);}else connection.textContent=message;
    }
  }
  const alpha=value=>Number((value/1e9).toFixed(4)).toLocaleString('en-GB',{maximumFractionDigits:4});
  function emissionRate(unit=incentiveUnit){
    if(unit==='alpha')return 1;
    const prices=currentPrices(),now=Date.now()/1000;
    if(!prices||!finite(prices.alpha_tao)||prices.alpha_tao<=0||!finite(prices.alpha_tao_at)||Math.abs(now-prices.alpha_tao_at)>1800)return null;
    if(unit==='tao')return prices.alpha_tao;
    if(!finite(prices.tao_usd)||prices.tao_usd<=0||!finite(prices.tao_usd_at)||Math.abs(now-prices.tao_usd_at)>1800)return null;
    return prices.alpha_tao*prices.tao_usd;
  }
  const emissionUnitName=()=>({alpha:'Alpha',tao:'TAO',usd:'USD'})[incentiveUnit];
  const emissionValue=row=>row.total/1e9*emissionRate();
  const emissionNumber=value=>value.toLocaleString('en-GB',{maximumFractionDigits:incentiveUnit==='tao'?4:2});
  const emissionTick=value=>(incentiveUnit==='usd'?'$':'')+value.toLocaleString('en-GB',{maximumSignificantDigits:4,notation:value>=10000?'compact':'standard'});
  function incentiveDetails(row,snapshot){
    const unit=emissionUnitName(),value=row.amount/1e9*emissionRate(),formatted=incentiveUnit==='alpha'?alpha(row.amount):emissionNumber(value);
    const lines=[`UID ${row.uid}`,`${formatted} · ${unit==='Alpha'?'α':unit}`,`${percent(row.share)} share`,`Block ${number(snapshot.block)}`,`${utc(snapshot.timestamp)} UTC`];
    if(incentiveUnit!=='alpha'){
      const prices=currentPrices();lines.push(`${alpha(row.amount)} α · estimated`,`α/TAO ${Number(prices.alpha_tao.toPrecision(5))}`);
      if(incentiveUnit==='usd')lines.push(`TAO/USD $${prices.tao_usd.toFixed(2)}`);
    }
    return lines.join('\n');
  }
  function syncIncentiveControl(index){
    const state=chartStates.get('incentive'),rows=state?.rows||[],row=rows[index],slider=$('incentive-slider');
    slider.max=Math.max(1,rows.length-1);slider.value=Math.max(0,index);slider.disabled=rows.length<2;
    slider.style.setProperty('--progress',`${rows.length>1?100*index/(rows.length-1):0}%`);
    syncNavigation('incentive',index,rows.length);
    const label=row?`${utc(row.timestamp)} UTC`:'No chain epochs';
    const epochLabel=$('incentive-slider-label');epochLabel.replaceChildren();
    if(row){const date=document.createElement('span'),time=document.createElement('span');date.textContent=day(row.timestamp);time.textContent=new Date(row.timestamp*1000).toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'UTC'});epochLabel.append(date,time);}else epochLabel.textContent=label;
    slider.setAttribute('aria-valuetext',row?`${label}, block ${number(row.block)}, ${emissionNumber(emissionValue(row))} ${emissionUnitName()} emission`:label);
    $('incentive-slider-label').title=label;
    setReading($('incentive-value'),row?emissionNumber(emissionValue(row)):'—');
    $('incentive-value').setAttribute('aria-label',row?`${emissionNumber(emissionValue(row))} ${emissionUnitName()} miner emission, ${label}`:'Emission unavailable');
    $('incentive-value-unit').textContent=({alpha:'α',tao:'τ',usd:'USD'})[incentiveUnit];
    for(const unit of ['alpha','tao','usd']){const button=$('incentive-unit-'+unit);button.disabled=!rows.length||emissionRate(unit)===null;button.setAttribute('aria-pressed',String(unit===incentiveUnit));button.title=button.disabled&&unit!=='alpha'?'Current price unavailable':unit==='alpha'?'Recorded alpha emission':'Estimated value at current prices';}
    for(const view of ['grid','bars']){const button=$('incentive-view-'+view);button.disabled=!rows.length;button.setAttribute('aria-pressed',String(incentiveView===view));}
    const scale=$('incentive-color-legend'),maximum=row?Math.max(...row.emission_rao):0;
    const rate=emissionRate(),unit=({alpha:' α / epoch',tao:' τ / epoch',usd:' / epoch'})[incentiveUnit];
    scale.replaceChildren();scale.hidden=!row;
    scale.setAttribute('aria-label',`Color scale: ${emissionUnitName()} emission per miner per epoch`);
    for(const weight of maximum>0?[0,.5,1]:[0]){
      const item=document.createElement('span'),swatch=document.createElement('i');
      const value=maximum/1e9*rate*weight;
      swatch.className='swatch';swatch.setAttribute('aria-hidden','true');
      swatch.style.background=weight?mixColor(themeColor('incentive-low'),colors.incentive,weight):themeColor('zero');
      const label=(incentiveUnit==='usd'?'$':'')+emissionNumber(value);
      item.append(swatch,document.createTextNode(label+(weight===1||maximum===0?unit:'')));
      item.title=`${emissionTick(value)} ${emissionUnitName()} per miner per epoch`;
      scale.append(item);
    }
    const delayed=data?.incentive&&Date.now()/1000-data.incentive.observed_at>900;
    $('incentive-status').textContent=row?`${row.emission_rao.filter(value=>value>0).length} miners · ${row.tempo}-block tempo${delayed?' · delayed':''}`:'Awaiting chain snapshot';
  }
  let incentiveSnapshot=null,incentiveRows=[],incentiveSelected=0,incentiveKey='',incentivePinned=false,incentivePaused=false;
  let incentiveIntroFrame=null,incentiveIntroIndex=null,incentiveIntroPlayed=false;
  function stopIncentiveIntro(keepFrame=false){
    incentiveIntroPlayed=true;delete $('incentive-grid').dataset.replaying;
    if(incentiveIntroFrame===null)return;
    cancelAnimationFrame(incentiveIntroFrame);incentiveIntroFrame=null;
    const state=chartStates.get('incentive');
    if(keepFrame&&state?.rows[incentiveIntroIndex])state.selected=state.rows[incentiveIntroIndex].id;
    incentiveIntroIndex=null;drawIncentive();
    if(keepFrame)syncViewLink();
  }
  function startIncentiveIntro(){
    if(incentiveIntroPlayed)return;
    incentiveIntroPlayed=true;
    const state=chartStates.get('incentive'),rows=state?.rows||[];
    if(reducedMotion.matches||rows.length<2||requestedView.has('incentive_epoch'))return;
    const target=rows.findIndex(row=>row.id===state.selected),last=target>=0?target:rows.length-1;
    if(last<1)return;
    const blocks=rows.slice(0,last+1).map(row=>row.id),started=performance.now();
    $('incentive-grid').dataset.replaying='true';
    const advance=now=>{
      const current=chartStates.get('incentive');
      if(document.hidden||reducedMotion.matches){stopIncentiveIntro();return;}
      const progress=Math.max(0,Math.min(1,(now-started)/1400));
      if(progress>=1){incentiveIntroFrame=null;incentiveIntroIndex=null;delete $('incentive-grid').dataset.replaying;drawIncentive();return;}
      const position=Math.min(last,Math.floor(progress*(last+1))),index=current?.rows.findIndex(row=>row.id===blocks[position]);
      if(index>=0&&index!==incentiveIntroIndex){incentiveIntroIndex=index;drawIncentive();}
      incentiveIntroFrame=requestAnimationFrame(advance);
    };
    incentiveIntroFrame=requestAnimationFrame(advance);
  }
  const incentiveSection=$('incentive-wrap').closest('section');
  for(const type of ['pointerdown','keydown','input','click','focusin'])incentiveSection.addEventListener(type,event=>{
    const requestedValue=event.type==='input'&&event.target===$('incentive-slider')?event.target.value:null;
    stopIncentiveIntro(true);
    if(requestedValue!==null)event.target.value=requestedValue;
  },{capture:true});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopIncentiveIntro();});
  function incentiveOrder(){return incentiveView==='bars'?incentiveRows.filter(row=>row.amount>0).sort((a,b)=>b.amount-a.amount||a.uid-b.uid):incentiveRows;}
  function hideIncentive(){hideTip($('incentive-tip'));incentivePinned=false;for(const cell of $('incentive-grid').children)cell.classList.remove('is-selected');}
  function inspectIncentive(uid,pinned=false){
    const row=incentiveRows[uid],cell=$('incentive-grid').children[uid],snapshot=incentiveSnapshot;
    if(!row||!cell||cell.hidden)return;
    incentiveSelected=uid;incentivePinned=pinned;
    for(const node of $('incentive-grid').children){node.tabIndex=node===cell?0:-1;node.classList.toggle('is-selected',node===cell);}
    const tip=$('incentive-tip'),wrap=$('incentive-wrap');
    tooltipContent(tip,incentiveDetails(row,snapshot),pinned?()=>{hideIncentive();incentivePaused=true;cell.focus({preventScroll:true});}:undefined);
    tip.hidden=false;
    const rect=cell.getBoundingClientRect(),base=wrap.getBoundingClientRect();
    positionTip(tip,wrap,rect.left-base.left+rect.width/2,rect.top-base.top);
  }
  controllers.set('incentive',{select(index){const state=chartStates.get('incentive'),row=state?.rows[index];if(row){state.selected=row.id;incentivePaused=true;drawIncentive();syncViewLink();}},hide:hideIncentive,dismiss:()=>{hideIncentive();incentivePaused=true;$('incentive-grid').children[incentiveSelected]?.focus({preventScroll:true});}});
  function drawIncentive(animateLayout=true){
    if(emissionRate()===null){incentiveUnit='alpha';syncViewLink();}
    const source=Array.isArray(data?.incentive_history)&&data.incentive_history.length?data.incentive_history:data?.incentive?[data.incentive]:[];
    const rows=source.filter(row=>row&&count(row.block)&&finite(row.timestamp)&&count(row.tempo)&&Array.isArray(row.emission_rao)&&row.emission_rao.length&&row.emission_rao.every(value=>count(value)&&Number.isSafeInteger(value))).map(row=>({...row,id:String(row.block),total:row.emission_rao.reduce((a,b)=>a+b,0)})).sort((a,b)=>a.block-b.block);
    const state=chartStates.get('incentive')||{selected:null};state.rows=rows;
    if(!state.initialized&&rows.length){const requested=requestedView.get('incentive_epoch');if(rows.some(row=>row.id===requested))state.selected=requested;state.initialized=true;}
    if(state.selected&&!rows.some(row=>row.id===state.selected))state.selected=null;
    chartStates.set('incentive',state);
    const selectedIndex=rows.findIndex(row=>row.id===state.selected),index=incentiveIntroIndex!==null?Math.min(incentiveIntroIndex,rows.length-1):selectedIndex>=0?selectedIndex:rows.length-1;
    const snapshot=incentiveSnapshot=rows[index],grid=$('incentive-grid'),empty=$('incentive-empty'),axes=$('incentive-axes');
    syncIncentiveControl(index);
    if(!snapshot){empty.hidden=false;empty.textContent='Chain emissions temporarily unavailable.';hideIncentive();grid.replaceChildren();axes.replaceChildren();axes.setAttribute('hidden','');incentiveRows=[];incentiveKey='';return;}
    grid.dataset.view=incentiveView;
    const amounts=snapshot.emission_rao,total=snapshot.total,max=Math.max(...amounts);
    incentiveRows=amounts.map((amount,uid)=>({uid,amount,share:total?amount/total:0}));
    const order=incentiveOrder();
    if(!order.some(row=>row.uid===incentiveSelected))incentiveSelected=order[0]?.uid??0;
    empty.hidden=!(incentiveView==='bars'&&!order.length);if(!empty.hidden)empty.textContent='No miner emissions this chain epoch.';
    const incentiveHigh=colors.incentive,incentiveLow=themeColor('incentive-low'),zeroColor=themeColor('zero');
    const width=grid.clientWidth,height=grid.clientHeight,key=JSON.stringify([incentiveView,incentiveUnit,emissionRate(),width,height,snapshot.block,amounts,incentiveHigh,incentiveLow,zeroColor]);
    if(key===incentiveKey)return;incentiveKey=key;
    const animate=animateLayout&&grid.children.length>0&&incentiveIntroFrame===null&&!grid.classList.contains('animate-in');
    const previous=animate?new Map([...grid.children].map(cell=>[Number(cell.dataset.uid),cell.hidden?null:cell.getBoundingClientRect()])):new Map();
    hideIncentive();
    while(grid.children.length>amounts.length)grid.lastChild.remove();
    while(grid.children.length<amounts.length){
      const uid=grid.children.length,cell=document.createElement('button');cell.type='button';cell.className='incentive-cell';cell.dataset.uid=uid;
      cell.addEventListener('pointermove',event=>{if(event.pointerType!=='touch'&&!incentivePinned){incentivePaused=false;inspectIncentive(uid);}});
      cell.addEventListener('focus',()=>{if(!incentivePaused)inspectIncentive(uid);});
      cell.addEventListener('click',event=>{incentivePaused=false;let target=uid;if(incentiveView==='bars'&&event.detail>0){const bounds=grid.getBoundingClientRect(),order=incentiveOrder(),left=bounds.width<500?38:46,index=Math.max(0,Math.min(order.length-1,Math.floor((event.clientX-bounds.left-left)/(bounds.width-left-8)*order.length)));target=order[index]?.uid??uid;}inspectIncentive(target,true);});
      cell.addEventListener('keydown',event=>{
        const ordered=incentiveOrder(),index=ordered.findIndex(row=>row.uid===uid),columns=Math.ceil(Math.sqrt(incentiveRows.length));
        const offsets={ArrowLeft:-1,ArrowRight:1,ArrowUp:incentiveView==='grid'?-columns:-1,ArrowDown:incentiveView==='grid'?columns:1};
        let target=event.key==='Home'?0:event.key==='End'?ordered.length-1:offsets[event.key]!==undefined?Math.max(0,Math.min(ordered.length-1,index+offsets[event.key])):null;
        if(target===null)return;event.preventDefault();incentivePaused=false;const next=ordered[target];inspectIncentive(next.uid,true);grid.children[next.uid].focus({preventScroll:true});
      });grid.append(cell);
    }
    axes.toggleAttribute('hidden',incentiveView!=='bars');axes.replaceChildren();axes.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const left=width<500?38:46,right=8,top=8,bottom=28,plotWidth=width-left-right,plotHeight=height-top-bottom;
    const maxValue=max/1e9*emissionRate(),magnitude=maxValue?10**Math.floor(Math.log10(maxValue)):1;
    const ceiling=maxValue?Math.ceil(maxValue/magnitude)*magnitude:1;
    if(incentiveView==='bars'){
      for(const fraction of [0,.5,1]){const y=top+plotHeight*(1-fraction);axes.append(svg('line',{x1:left,x2:width-right,y1:y,y2:y,stroke:themeColor(fraction?'line':'axis'),'stroke-dasharray':fraction?'2 6':'none'}),svg('text',{x:left-8,y:y+4,'text-anchor':'end'},emissionTick(ceiling*fraction)));}
      const ticks=[0,Math.floor((order.length-1)/2),order.length-1];for(const index of [...new Set(ticks)].filter(index=>index>=0)){const x=left+(index+.5)*plotWidth/Math.max(1,order.length);axes.append(svg('text',{x,y:height-6,'text-anchor':index===0?'start':index===order.length-1?'end':'middle'},String(index+1)));}

    }
    const columns=Math.ceil(Math.sqrt(amounts.length)),gap=width<400?3:4,size=(width-gap*(columns-1))/columns,rank=new Map(order.map((row,index)=>[row.uid,index]));
    for(const row of incentiveRows){
      const cell=grid.children[row.uid],index=rank.get(row.uid);
      cell.hidden=incentiveView==='bars'&&index===undefined;
      cell.tabIndex=row.uid===incentiveSelected?0:-1;
      cell.style.setProperty('--reveal-delay',`${(Math.floor(row.uid/columns)+row.uid%columns)*7}ms`);
      cell.dataset.emissionRao=row.amount;cell.dataset.emissionValue=row.amount/1e9*emissionRate();
      cell.setAttribute('aria-label',`Miner UID ${row.uid}: ${emissionNumber(row.amount/1e9*emissionRate())} ${emissionUnitName()}, ${percent(row.share)} of miner emissions`);
      if(cell.hidden)continue;
      const bars=incentiveView==='bars',barWidth=plotWidth/Math.max(1,order.length),barHeight=plotHeight*(row.amount/1e9*emissionRate())/ceiling;
      Object.assign(cell.style,{left:`${bars?left+index*barWidth:row.uid%columns*(size+gap)}px`,top:`${bars?top+plotHeight-barHeight:Math.floor(row.uid/columns)*(size+gap)}px`,width:`${bars?Math.max(1,barWidth-1):size}px`,height:`${bars?barHeight:size}px`,background:!row.amount?zeroColor:mixColor(incentiveLow,incentiveHigh,row.amount/max)});
      if(!reducedMotion.matches&&animate&&previous.get(row.uid)){
        cell.getAnimations().filter(animation=>animation.id==='incentive-morph').forEach(animation=>animation.cancel());const before=previous.get(row.uid),after=cell.getBoundingClientRect();
        if(after.width&&after.height){const animation=cell.animate([{transform:`translate(${before.left-after.left}px,${before.top-after.top}px) scale(${before.width/after.width},${before.height/after.height})`},{transform:'none'}],{duration:420,easing:'cubic-bezier(.2,.7,.3,1)'});animation.id='incentive-morph';}
      }
    }
    if(!animate&&observer&&!incentiveIntroPlayed)observer.observe($('incentive-wrap'));
  }
  $('incentive-grid').addEventListener('pointerleave',()=>{if(!incentivePinned)hideIncentive();});
  for(const view of ['grid','bars'])$('incentive-view-'+view).addEventListener('click',()=>{
    if(incentiveView===view)return;
    incentiveView=view;incentivePaused=true;
    const grid=$('incentive-grid');
    grid.classList.remove('animate-in');
    for(const cell of grid.children)cell.getAnimations().forEach(animation=>animation.cancel());
    drawIncentive(false);observer?.unobserve($('incentive-wrap'));replayEntrance(grid);syncViewLink();
  });

  for(const unit of ['alpha','tao','usd'])$('incentive-unit-'+unit).addEventListener('click',()=>{
    if(incentiveUnit===unit||emissionRate(unit)===null)return;incentiveUnit=unit;incentivePaused=true;drawIncentive();syncViewLink();
  });
  const lastBatchesKey='affine:last-seen-batches:v1';
  let displayedBatches=null,batchesReplay=null,batchesInitialized=false;
  try{const saved=JSON.parse(localStorage.getItem(lastBatchesKey));if(Number.isSafeInteger(saved)&&saved>=0)displayedBatches=saved;}catch{}
  function saveSeenBatches(){if(Number.isSafeInteger(displayedBatches))try{localStorage.setItem(lastBatchesKey,JSON.stringify(displayedBatches));}catch{}}
  function paintBatches(value){displayedBatches=Math.round(value);$('batches-total').textContent=number(displayedBatches);}
  function stopBatchReplay(){if(batchesReplay){cancelAnimationFrame(batchesReplay.frame);batchesReplay=null;}saveSeenBatches();}
  function renderBatchTotal(epochs){
    const rows=epochs.filter(batchAvailable),total=rows.reduce((sum,row)=>sum+row.batches,0),node=$('batches-total');
    node.title=`Submitted batches across ${number(rows.length)} recorded epochs`;node.setAttribute('aria-label',`${number(total)} submitted batches across ${number(rows.length)} recorded epochs`);
    if(batchesReplay?.target===total)return;
    const returning=!batchesInitialized;batchesInitialized=true;
    if(reducedMotion.matches||displayedBatches===null||displayedBatches===total){stopBatchReplay();paintBatches(total);saveSeenBatches();return;}
    stopBatchReplay();const from=displayedBatches,started=performance.now(),duration=returning?1400:650;
    paintBatches(from);batchesReplay={target:total,frame:null};
    const advance=now=>{
      if(!batchesReplay)return;
      if(document.hidden){stopBatchReplay();return;}
      const progress=Math.max(0,Math.min(1,(now-started)/duration));
      paintBatches(from+(total-from)*(1-(1-progress)**3));
      if(progress>=1){batchesReplay=null;paintBatches(total);saveSeenBatches();return;}
      batchesReplay.frame=requestAnimationFrame(advance);
    };
    batchesReplay.frame=requestAnimationFrame(advance);
  }
  window.addEventListener('pagehide',stopBatchReplay);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopBatchReplay();});
  function render() {
    if(!data)return;
    const epochs=data.epochs.filter(row=>row.source==='live-reward-math'&&finite(row.start)&&count(row.batches)).sort((a,b)=>a.start-b.start);
    renderBatchTotal(epochs);
    window.dispatchEvent(new CustomEvent('affine:snapshot',{detail:data}));
    if(window.matchMedia('(min-width:1000px)').matches){status();return;}
    const ids=new Set(epochs.map(row=>row.id)),epochOrder=new Map(epochs.map((row,index)=>[row.id,index]));
    const evaluations=data.evaluations.filter(row=>ids.has(row.epoch_id)&&row.env_id==='affine_math'&&row.status==='complete'&&finite(row.timestamp)&&finite(row.mean_reward)&&row.mean_reward>=0&&row.mean_reward<=1&&count(row.count)&&row.count>0&&count(row.successes)&&row.successes<=row.count).sort((a,b)=>epochOrder.get(a.epoch_id)-epochOrder.get(b.epoch_id)||a.timestamp-b.timestamp);
    const latestByEpoch=new Map();
    for(const row of evaluations)latestByEpoch.set(row.epoch_id,row);
    const epochResults=[...latestByEpoch.values()],setups=new Map();
    const history=epochResults.map((row,index)=>{
      const key=cohortKey(row);if(!setups.has(key))setups.set(key,setups.size+1);
      return {...row,display_setup:setups.get(key),setup_changed:index>0&&key!==cohortKey(epochResults[index-1])};
    });
    drawIncentive();drawChart('evaluation',history);drawChart('batch',epochs.slice(-12));drawMiners(epochs);status();
  }
  document.addEventListener('pointerdown',event=>{
    for(const [kind,controller] of controllers) {
      const container=kind==='miner'?$('miner-grid').parentElement:$(kind+'-wrap');
      if(!container.contains(event.target))controller.hide();
    }
  });
  document.addEventListener('keydown',event=>{if(event.key==='Escape')for(const [kind,controller] of controllers){
    const fromTip=kind!=='miner'&&$(kind+'-tip').contains(document.activeElement);
    if(kind==='miner')controller.hoverPaused=true;
    if(fromTip)controller.dismiss();else controller.hide();
  }});
  $('miner-epoch-slider').addEventListener('input',event=>{
    const row=minerEpochs.filter(validGrid)[Number(event.target.value)];
    if(row){selectedMinerEpoch=row.id;drawMiners(minerEpochs);}
  });
  for(const kind of ['evaluation','batch','incentive']){
    $(kind+'-slider').addEventListener('input',event=>controllers.get(kind)?.select(Number(event.target.value)));
  }
  for(const prefix of ['miner-epoch','evaluation','batch','incentive']){
    for(const [direction,step] of [['previous',-1],['next',1]]){
      $(prefix+'-'+direction).addEventListener('click',()=>{
        const slider=$(prefix+'-slider');
        slider.value=Math.max(Number(slider.min),Math.min(Number(slider.max),Number(slider.value)+step));
        slider.dispatchEvent(new Event('input',{bubbles:true}));
      });
    }
  }
  $('copy-view').addEventListener('click',async()=>{
    const button=$('copy-view'),status=$('copy-view-status');button.disabled=true;
    const miner=minerEpochs.filter(validGrid)[Number($('miner-epoch-slider').value)];
    if(miner)selectedMinerEpoch=miner.id;
    for(const kind of ['evaluation','batch']){
      const state=chartStates.get(kind),row=state?.rows[Number($(kind+'-slider').value)];
      if(row)state.selected=recordKey(row);
    }
    syncViewLink();
    try{await navigator.clipboard.writeText(location.href);button.dataset.copyState='success';button.textContent='Copied';status.textContent='View link copied';}
    catch{button.dataset.copyState='error';button.textContent='Copy unavailable';status.textContent='Clipboard access unavailable';}
    setTimeout(()=>{button.disabled=false;delete button.dataset.copyState;button.textContent='Copy view';status.textContent='';},1800);
  });
  let resizeTimer;
  const resizeObserver='ResizeObserver' in window?new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,100);}):null;
  if(resizeObserver)resizeObserver.observe($('main'));
  else window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,100);});
  document.fonts?.ready.then(()=>{if(data)render();});
  async function refresh() {
    try {
      const response=await fetch('/network-data.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});
      if(!response.ok)throw Error('Snapshot unavailable');
      const next=await response.json();
      if(!next||!Array.isArray(next.epochs)||!Array.isArray(next.evaluations)||[...next.epochs,...next.evaluations].some(row=>!row||typeof row!=='object'||Array.isArray(row)))throw Error('Invalid snapshot');
      const nextFingerprint=JSON.stringify([next.epochs,next.evaluations,next.incentive,next.incentive_history,next.incentive_prices]);
      data=next;unavailable=false;
      $('main').dataset.state='ready';$('main').setAttribute('aria-busy','false');
      if(nextFingerprint!==fingerprint){fingerprint=nextFingerprint;render();}else{drawIncentive();status();}
    } catch {
      unavailable=true;status();
      if(!data) {
        $('main').dataset.state='unavailable';$('main').setAttribute('aria-busy','false');
        for(const kind of ['evaluation','batch']){$(kind+'-empty').hidden=false;$(kind+'-empty').textContent='Network records temporarily unavailable.';}
        $('incentive-empty').textContent='Chain emissions temporarily unavailable.';$('incentive-status').textContent='Awaiting chain snapshot';
        $('miner-epoch').textContent='Network records temporarily unavailable.';
        $('miner-empty').hidden=false;$('miner-empty').textContent='Miner records temporarily unavailable.';
      }
    }
  }
  if(lastDisplayedPrices)paintMarketPrices(lastDisplayedPrices);
  refresh();setInterval(refresh,15000);
  refreshMarketPrices();connectMarketPrices();setInterval(refreshMarketPrices,1000);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){pausePriceReplay();marketStream?.close();marketStream=null;}else{priceArrivalPending=validSeenPrices(lastDisplayedPrices);marketInitialFetchComplete=false;refreshMarketPrices();connectMarketPrices();}});
  window.addEventListener('pagehide',()=>{pausePriceReplay();marketStream?.close();marketStream=null;});
  window.addEventListener('pageshow',event=>{if(data)render();if(event.persisted){priceArrivalPending=validSeenPrices(lastDisplayedPrices);marketInitialFetchComplete=false;}refreshMarketPrices();connectMarketPrices();});
})();
