/* Current-run panels fed by /site-metrics.json (aggregates and UIDs only). */
(()=>{
 'use strict';
 const ns='http://www.w3.org/2000/svg',main=document.getElementById('main');
 const fin=n=>typeof n==='number'&&Number.isFinite(n);
 const css=name=>getComputedStyle(document.documentElement).getPropertyValue('--'+name).trim();
 const fmt=(n,d=1)=>fin(n)?n.toLocaleString('en-GB',{maximumFractionDigits:d}):'—';
 const pct=(n,d=1)=>fin(n)?(100*n).toFixed(d)+'%':'—';
 const mins=s=>fin(s)?Math.round(s/60):null;
 const utc=t=>fin(t)?new Date(t*1000).toLocaleString('en-GB',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false})+' UTC':'—';
 const ago=t=>{if(!fin(t))return '—';const s=Math.max(0,Date.now()/1000-t);return s<90?Math.round(s)+'s ago':s<5400?Math.round(s/60)+' min ago':(s/3600).toFixed(1)+' h ago';};
 const el=(tag,attrs={},text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
 const h=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
 /* Chart box in viewBox units: 480x230 in a tile; sized to the plot area when the tile is expanded (chart-expand.js). */
 const box=(svg,W=480,H=230)=>{const host=svg.closest('.cx-open,.probe-trend-hero,.net-hero')&&svg.parentElement;const r=host&&host.getBoundingClientRect();if(!r||r.width<50||r.height<50)return{W,H};const k=Math.max(1,Math.min(1.6,r.width/640));return{W:Math.round(r.width/k),H:Math.round(r.height/k)};};

 function panel(id,title,color,views,legend){
  const s=document.createElement('section');s.className='site-panel';s.id='panel-'+id;s.setAttribute('aria-labelledby','sp-'+id);
  s.style.setProperty('--sp-color','var(--'+color+')');
  s.innerHTML='<div class="section-heading"><h2 id="sp-'+id+'">'+title+'</h2><div class="sp-reading"><span class="reading">—</span><span class="reading-change sp-sub"></span></div></div><div class="sp-controls"><span class="sp-context">Loading</span><div class="miner-view-control" role="group" aria-label="'+title+' view"></div></div><div class="sp-plot"><svg role="group" aria-label="'+title+'"></svg><p class="empty" hidden></p><output class="sp-tip" hidden></output></div><div class="legend sp-legend"></div>';
  const p={id,section:s,svg:s.querySelector('svg'),tip:s.querySelector('output'),view:0,views:views||[]};
  (views||[]).forEach((label,i)=>{const b=document.createElement('button');b.type='button';b.textContent=label;b.setAttribute('aria-pressed',String(i===0));b.addEventListener('click',()=>{p.view=i;render();});s.querySelector('.miner-view-control').append(b);});
  p.setLegend=items=>{const L=s.querySelector('.sp-legend');L.replaceChildren();for(const[label,c]of items){const span=h('span');const i=h('i','swatch');i.style.background=c.startsWith('--')?'var('+c+')':c;span.append(i,document.createTextNode(label));L.append(span);}};
  p.setLegend(legend||[]);
  return p;
 }
 function place(p,after){const ref=after&&document.querySelector(after);if(ref)ref.after(p.section);else main.append(p.section);}

 const P={
  weights:panel('weights','Weights · v9','accepted',['Bars','Table'],[]),
  probe:panel('probe','Net change per epoch','performance',[],[['Net ≥ 0','--performance'],['Net < 0','--rejected'],['Gate verdict dot','--axis']]),
  probetrend:panel('perf','Performance','performance',[],[['Held-out (32 fixed)','--performance'],['Env probe','--incentive']]),
  mining:panel('mining','Mining status','incentive',[],[]),
  groups:panel('groups','Groups per epoch','accepted',['All','Last 20'],[['Groups','--accepted']]),
  miners:panel('miners','Miners per epoch','incentive',['All','Last 20'],[['Miners sealed','--incentive']]),
  rollouts:panel('rollouts','Rollouts per epoch','performance',['All','Last 20'],[]),
  grad:panel('grad','Gradient norm','incentive',['All','Last 20'],[['Pre-clip grad norm / step','--incentive']]),
  loss:panel('loss','Training loss','rejected',['All','Last 20'],[['Loss / step','--rejected']]),
  timing:panel('timing','Epoch timings','unchecked',['Last 20','All'],[['Gather','--incentive'],['Train','--performance'],['Post-probe (gate)','--accepted'],['Other','--unchecked']]),
  gate:panel('gate','Promotion gate','accepted',['Chart','Table'],[['Promoted · cap-hit','--performance'],['Rejected · cap-hit','--rejected']]),
 };
 place(P.weights,'#main>.incentive-section');
 place(P.mining,'#main>section[aria-labelledby="evaluation-title"]');
 place(P.groups,'#main>section[aria-labelledby="batch-title"]');
 for(const k of ['miners','rollouts','grad','loss','timing','gate'])main.append(P[k].section);
 /* Net change per epoch (Jacob t693u): converted Env probe pass-rate chart, moved to the top of the page, full width.
    Bar = candidate pass rate minus parent pass rate. Gated epochs (E152+): C vs H on the private gate sample minus that epoch's batch.
    Pre-gate epochs: full env-probe pass rate of the checkpoint vs its parent (same optimizer step − 1), shown faded. */
 main.prepend(P.probe.section);
 P.probe.section.classList.add('net-hero');
 /* Env probe pass rate over time (restored t694u): directly below the net-change chart. */
 P.probe.section.after(P.probetrend.section);
 P.probetrend.section.classList.add('probe-trend-hero');
 (()=>{const note=h('p','sp-note');note.append(
   document.createTextNode('Net change = post-probe pass rate − pre-probe pass rate. Green ≥ 0, red < 0. Dot marks the gate verdict (pass unless clearly worse). '),
   Object.assign(document.createElement('a'),{href:'/gate.json',textContent:'gate.json'}),
   document.createTextNode(' · '),
   Object.assign(document.createElement('a'),{href:'/probe/series.json',textContent:'series.json'})
 );P.probe.section.append(note);})();

 let data=null,selectedIndex=null;
 /* Promotion gate (Jacob t690u): append-only verdict ledger from /gate.json; sample task ids are never published. */
 let gateData=null;
 const gateOf=i=>(gateData?.entries||[]).filter(g=>g.epoch===i).pop()||null;
 const gspan=t=>t&&fin(t.start)&&fin(t.end)&&t.end>t.start?(t.end-t.start)/60:null;
 const hm=t=>fin(t)?new Date(t*1000).toLocaleString('en-GB',{hour:'2-digit',minute:'2-digit',timeZone:'UTC',hour12:false}):'—';
 const gateLines=g=>{if(!g)return [];const T=g.timestamps||{},pre=T.pre_probe,post=T.post_probe;return [
   'Gate: '+(g.verdict==='promoted'?'GREEN promoted':'RED rejected')+' · z '+fmt(g.z,2)+' · cap-hit '+pct(g.candidate_cap_rate,1)+' · n '+fmt(g.n,0),
   g.thresholds?'Rule: pass unless clearly worse · z '+(g.thresholds.z_inclusive?'≥ ':'> ')+fmt(g.thresholds.z_min,1)+' · cap-hit < '+pct(g.thresholds.cap_max,0)+' · n ≥ '+fmt(g.thresholds.min_tasks,0):null,
   pre?'Pre-probe H '+String(g.head).slice(0,8)+(pre.reused?' (reused) ':' ')+hm(pre.started_at)+'–'+hm(pre.finished_at)+' UTC':null,
   post&&fin(post.start)?'Post-probe C '+String(g.candidate).slice(0,8)+' '+hm(post.start)+'–'+hm(post.end)+' UTC':null];};
 const epochsAll=()=>(data?.epochs||[]).filter(e=>fin(e.index));
 const windowed=(p,rows)=>p.view===1?rows.slice(-20):rows;

 function setReading(p,text,sub,context){
  p.section.querySelector('.reading').textContent=text;p.section.querySelector('.sp-sub').textContent=sub||'';
  p.section.querySelector('.sp-context').textContent=context||'';
  p.section.querySelectorAll('.miner-view-control button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===p.view)));
 }
 function empty(p,message){const e=p.section.querySelector('.empty');e.hidden=!message;e.textContent=message||'';}
 function tip(p,title,value,detail){p.tip.replaceChildren(h('span','',title),h('b','',value));if(detail)for(const line of [].concat(detail))p.tip.append(h('small','',line));p.tip.hidden=false;}

 /* Generic epoch chart: stacked bars and/or a line, x = epoch label. */
 function chart(p,rows,series,opts={}){
  const svg=p.svg,{W,H}=box(p.svg);svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+W+' '+H);p.tip.hidden=true;
  const vals=rows.map(r=>series.map(s=>s.get(r)));
  const usable=vals.some(v=>v.some(fin));empty(p,usable?null:(opts.emptyText||'No measurements yet.'));if(!usable)return;
  const stackTotals=vals.map(v=>opts.stacked?v.reduce((a,b)=>a+(fin(b)?b:0),0):Math.max(...v.filter(fin)));
  let lo=opts.min!==undefined?opts.min:Math.min(0,...vals.flat().filter(fin));
  let hi=opts.max!==undefined?opts.max:Math.max(...stackTotals.filter(fin));
  if(hi===lo)hi=lo+1;
  if(opts.max===undefined){const span=hi-lo,mag=10**Math.floor(Math.log10(span)),step=[1,2,2.5,5,10].map(n=>n*mag).find(n=>n*4>=span)||mag*10;hi=Math.ceil(hi/step)*step;if(lo<0)lo=Math.floor(lo/step)*step;}
  const L=46,R=W-10,T=12,B=H-40,w=R-L,n=rows.length,x=i=>L+w*(i+.5)/n,y=v=>B-(B-T)*(v-lo)/(hi-lo);
  for(const t of [lo,(lo+hi)/2,hi]){svg.append(el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'sp-guide'}),el('text',{x:L-8,y:y(t)+4,'text-anchor':'end'},opts.tick?opts.tick(t):fmt(t,3)));}
  if(lo<0)svg.append(el('line',{x1:L,x2:R,y1:y(0),y2:y(0),class:'sp-zero'}));
  const selected=rows.findIndex(r=>r.index===selectedIndex);
  const focus=selected>=0?selected:n-1;
  const bw=Math.max(1,w/n*.78);
  series.forEach((s,si)=>{
   if(s.type!=='bar')return;
   rows.forEach((r,i)=>{const v=vals[i][si];if(!fin(v)||v===0)return;const base=opts.stacked?vals[i].slice(0,si).reduce((a,b)=>a+(fin(b)?b:0),0):0;
    svg.append(el('rect',{x:x(i)-bw/2,y:y(base+v),width:bw,height:Math.max(0,y(base)-y(base+v)),fill:'var('+s.color+')',opacity:i===focus?1:.72,class:'sp-rise'}));});
  });
  series.forEach((s,si)=>{
   if(s.type!=='line')return;let d='',open=false,gap=0;
   rows.forEach((r,i)=>{const v=vals[i][si];if(!fin(v)){gap++;if(!opts.connect||gap>opts.connect)open=false;return;}gap=0;d+=(open?' L':' M')+x(i).toFixed(1)+' '+y(v).toFixed(1);open=true;});
   svg.append(el('path',{d,class:'sp-line',stroke:'var('+s.color+')'}));
  });
  const label=opts.label||(r=>'E'+r.index);
  const every=Math.max(1,Math.ceil(n/Math.max(4,Math.round(8*W/480))));
  rows.forEach((r,i)=>{if(i===0||i===n-1||i===focus||(i%every===0&&Math.abs(i-focus)>every/2&&n-1-i>every/2))svg.append(el('text',{x:x(i),y:B+18,'text-anchor':'middle'},label(r)));
   const hit=el('rect',{x:x(i)-w/n/2,y:T,width:w/n,height:B-T+22,fill:'transparent',tabindex:0,role:'img','aria-label':label(r)+': '+series.map((s,si)=>s.label+' '+fmt(vals[i][si],4)).join(', ')});
   const show=()=>{const lines=series.map((s,si)=>s.label+': '+(s.fmt?s.fmt(vals[i][si]):fmt(vals[i][si],4)));if(opts.extra)lines.push(...[].concat(opts.extra(r)).filter(Boolean));tip(p,opts.title?opts.title(r):'Epoch '+r.index,opts.headline?opts.headline(r):(series[0].fmt?series[0].fmt(vals[i][0]):fmt(vals[i][0],4)),lines);};
   hit.addEventListener('pointerenter',show);hit.addEventListener('pointermove',show);hit.addEventListener('focus',show);
   hit.addEventListener('pointerleave',()=>p.tip.hidden=true);hit.addEventListener('blur',()=>p.tip.hidden=true);
   svg.append(hit);});
  svg.append(el('text',{x:(L+R)/2,y:H-5,'text-anchor':'middle'},opts.axis||'Epoch'));
 }

 function focusRow(rows){return rows.find(r=>r.index===selectedIndex)||rows[rows.length-1];}

 function drawWeights(){
  const p=P.weights,w=data?.weights;
  if(!w||!Array.isArray(w.paid)){setReading(p,'—','', 'Weights unavailable');empty(p,'v9 weights unavailable.');p.svg.replaceChildren();return;}
  const a=w.assessment||{},chain=w.chain||{},perDay=fin(chain.miner_emission_rao_per_tempo)&&fin(chain.tempo)?chain.miner_emission_rao_per_tempo*(7200/chain.tempo)/1e9:null;
  setReading(p,String(w.paid.length),'paid UIDs',(w.chain_status==='submitted'?'Set on chain':'Status '+(w.chain_status||'—'))+' · block '+fmt(w.block,0)+' · '+ago(w.run_at));
  const sm=a.smoothing;
  const L=p.section.querySelector('.sp-legend');{const sp=h('span');const sw=h('i','swatch');sw.style.background='var(--accepted)';sp.append(sw,document.createTextNode('Weight per UID · top '+Math.min(24,w.paid.length)+' of '+w.paid.length));L.replaceChildren(sp);}
  const svg=p.svg;svg.replaceChildren();p.tip.hidden=true;
  const rows=w.paid.slice(0,24);empty(p,rows.length?null:'No UID is paid in this window.');
  const plot=p.section.querySelector('.sp-plot');
  plot.querySelector('.sp-table')?.remove();
  if(p.view===1){svg.style.display='none';
   const t=h('table','sp-table');t.innerHTML='<thead><tr><th>UID</th><th>Weight</th><th>Est. α / day</th><th>Last tempo α</th></tr></thead>';const tb=h('tbody');
   for(const r of w.paid){const tr=h('tr');[String(r.uid),pct(r.weight,2),perDay?fmt(r.weight*perDay,1):'—',fin(r.last_tempo_emission_rao)?fmt(r.last_tempo_emission_rao/1e9,2):'—'].forEach(v=>tr.append(h('td','',v)));tb.append(tr);}
   t.append(tb);plot.append(t);return;}
  svg.style.display='';
  const PER=12,height=Math.max(230,10+Math.min(rows.length,PER)*18);svg.setAttribute('viewBox','0 0 480 '+height);
  const max=Math.max(...rows.map(r=>r.weight),0.01);
  rows.forEach((r,i)=>{const col=Math.floor(i/PER),L0=52+col*244,R0=L0+142,yy=10+(i%PER)*18,wd=(R0-L0)*r.weight/max;
   svg.append(el('text',{x:L0-8,y:yy+10,'text-anchor':'end'},'UID '+r.uid),el('rect',{x:L0,y:yy,width:Math.max(1,wd),height:12,fill:'var(--accepted)',class:'sp-grow'}),el('text',{x:L0+wd+6,y:yy+10},pct(r.weight,1)));
   const hit=el('rect',{x:col*244,y:yy-2,width:236,height:18,fill:'transparent',tabindex:0,role:'img','aria-label':'UID '+r.uid+' weight '+pct(r.weight,2)});
   const show=()=>tip(p,'UID '+r.uid,pct(r.weight,2),[perDay?'≈ '+fmt(r.weight*perDay,1)+' α / day at current miner emission':null,fin(r.last_tempo_emission_rao)?'Last tempo on chain: '+fmt(r.last_tempo_emission_rao/1e9,2)+' α (block '+fmt(chain.block,0)+')':null,'Window ending '+utc(w.window_end)].filter(Boolean));
   hit.addEventListener('pointerenter',show);hit.addEventListener('focus',show);hit.addEventListener('pointerleave',()=>p.tip.hidden=true);hit.addEventListener('blur',()=>p.tip.hidden=true);svg.append(hit);});
 }

 function drawHeldout(){
  const p=P.heldout,all=(data?.heldout||[]).filter(r=>fin(r.score));
  const rows=windowed(p,all).map(r=>({...r,index:r.produced_by_epoch}));
  const last=all[all.length-1],prev=all[all.length-2],sel=rows.find(r=>r.index===selectedIndex&&selectedIndex!==null)||last;
  const delta=sel&&prev?sel.score-(all[all.indexOf(sel)-1]||prev).score:null;
  setReading(p,sel?pct(sel.score,1):'—',fin(delta)?(delta>=0?'+':'')+(100*delta).toFixed(1)+' pts vs previous':'',sel?'Checkpoint from '+sel.label+' · '+sel.successes+' / '+sel.count+' solved':'');
  chart(p,rows,[{key:'score',label:'Solved',type:'line',color:'--performance',get:r=>r.score,fmt:v=>pct(v,1)}],{min:0,max:1,tick:v=>Math.round(v*100)+'%',label:r=>r.label,title:r=>'Checkpoint from '+r.label,axis:'Checkpoint (training epoch that produced it)',extra:r=>[r.successes+' / '+r.count+' problems · '+(r.set_name||'held-out'),'Evaluated '+utc(r.evaluated_at)]});
  const lastEpoch=epochsAll().filter(e=>e.phase==='finalized').pop();
  p.setLegend([['Solve rate','--performance']]);
 }

 function drawMining(){
  const p=P.mining,m=data?.mining;p.svg.style.display='none';p.tip.hidden=true;
  const plot=p.section.querySelector('.sp-plot');plot.querySelector('.sp-status')?.remove();
  if(!m){setReading(p,'—','','Mining status unavailable');return;}
  const id=String(m.epoch||'').match(/-e(\d+)$/),epoch=id?'E'+Number(id[1]):'—',now=Date.now()/1000;
  const open=m.accepting_submissions===true,left=fin(m.deadline)?m.deadline-now:null;
  setReading(p,open?'Open':'Closed',epoch+(open&&left>0?' · '+Math.ceil(left/60)+' min left':''),'Status '+String(m.status||'—').replace(/_/g,' ')+' · updated '+ago(m.status_updated_at));
  const cur=epochsAll().slice(-1)[0],fin2=epochsAll().filter(e=>e.phase==='finalized').slice(-1)[0];
  const dl=h('dl','sp-status');
  const add=(k,v)=>{const d=h('div');d.append(h('dt','',k),h('dd','',v));dl.append(d);};
  add('Epoch',epoch+(cur&&cur.phase?' · '+cur.phase:''));
  add('Window',utc(m.start)+' → '+(fin(m.deadline)?new Date(m.deadline*1000).toISOString().slice(11,16)+' UTC':'—'));
  add('Payable',m.payable===true?'Yes':m.payable===false?'No':'—');
  add('Group',fmt(m.group_size||m.samples_per_batch,0)+' rollouts · max '+fmt(m.max_batches,0)+' groups / miner');
  add('Last finalized',fin2?'E'+fin2.index+' · '+utc(fin2.finalized_at):'—');
  if(fin2&&fin2.timing&&fin(fin2.timing.cycle_seconds||fin2.timing.total_seconds))add('Epoch length',mins(fin2.timing.cycle_seconds||fin2.timing.total_seconds)+' min');
  if(m.notice)add('Notice',m.notice);
  plot.append(dl);
  p.section.querySelector('.sp-legend').replaceChildren(h('span','','Data refreshed '+ago(data.generated_at)+(data.problems?.length?' · '+data.problems.join(', '):'')));
 }

 /* ---- Rollouts per epoch: intake funnel (groups) + outcome of every sampled rollout ----
    Gaps (missing bars / broken line) mean the epoch's artifacts do not record that field. */
 const RO={
  out:[
   {key:'passed',label:'Pass',color:'--performance',get:t=>t.correct},
   {key:'failed',label:'Fail',color:'--accepted',get:t=>t.wrong},
   {key:'unres',label:'Unres.',color:'--unchecked',get:t=>t.unresolved_uncapped},
   {key:'unk',label:'Unres. n/r',color:'--unchecked',opacity:.55,get:t=>fin(t.unresolved_uncapped)?null:t.unresolved},
   {key:'cap',label:'Hit output cap',color:'--rejected',get:t=>t.hit_cap},
  ],
  intake:[
   {key:'trained',label:'Trained',color:'--performance',get:e=>e.training?e.training.groups:null},
   {key:'not',label:'Not selected',color:'--unchecked',get:e=>{if(fin(e.groups_total)){const tr=e.training&&fin(e.training.groups)?e.training.groups:0;return Math.max(0,e.groups_total-tr);}const r=e.not_selected_reasons||{},a=[e.strip?e.strip.stripped:null,!e.training&&e.control&&fin(e.groups_sealed)?e.groups_sealed:null,r['cross-miner-task-collision'],r['training-task-budget']].filter(fin);return a.length?a.reduce((x,y)=>x+y,0):null;}},
  ],
  policy:{trained:'unresolved trained at reward 0',dropped:'unresolved dropped before training',masked:'unresolved masked out of the loss'},
 };
 const capOf=e=>fmt(e?.training?.output_token_cap||8192,0);
 const capPct=e=>{const t=e.training;return t&&fin(t.hit_cap)&&t.rollouts?t.hit_cap/t.rollouts:null;};
 function roIntakeText(e){
  const parts=[];
  if(fin(e.groups_total))parts.push(fmt(e.groups_total,0)+' submitted');
  if(fin(e.groups_sealed))parts.push(fmt(e.groups_sealed,0)+' sealed');
  parts.push(e.training&&fin(e.training.groups)?fmt(e.training.groups,0)+' trained':(e.control?'none trained ('+(e.control.action||'control')+(fin(e.control.rollback_epoch)?' to E'+e.control.rollback_epoch:'')+')':'— trained'));
  return parts.join(' → ')+' groups';
 }
 function roReasons(obj){return Object.entries(obj||{}).filter(([,v])=>v).map(([k,v])=>fmt(v,0)+' '+k).join(' · ');}
 function roLines(e){
  const t=e.training,lines=[];
  if(t){const n=t.rollouts,share=v=>fin(v)&&n?' ('+pct(v/n,1)+')':'';
   lines.push('Passed: '+fmt(t.correct,0)+share(t.correct),'Failed (wrong answer): '+fmt(t.wrong,0)+share(t.wrong));
   if(fin(t.unresolved_uncapped))lines.push('Unresolved, under cap: '+fmt(t.unresolved_uncapped,0)+share(t.unresolved_uncapped),'Hit '+capOf(e)+'-token cap: '+fmt(t.hit_cap,0)+share(t.hit_cap));
   else lines.push('Unresolved: '+fmt(t.unresolved,0)+share(t.unresolved)+' · cap hits not recorded');
   lines.push('Mean output: '+(fin(t.mean_output_tokens)?fmt(t.mean_output_tokens,0)+' tokens':'— (not recorded for every rollout)'));
   if(t.unresolved_policy)lines.push(RO.policy[t.unresolved_policy]||t.unresolved_policy);
  }else if(e.control){lines.push('No training update: '+(e.control.mode||e.control.action)+(fin(e.control.rollback_epoch)?' · checkpoint rolled back to E'+e.control.rollback_epoch:''));}
  else lines.push('No training result recorded');
  lines.push(roIntakeText(e));
  const ns=roReasons({'over budget':e.not_selected_reasons?.['training-task-budget'],'duplicate task':e.not_selected_reasons?.['cross-miner-task-collision']});
  if(ns)lines.push('Not selected: '+ns);
  if(e.strip)lines.push('Stripped: '+(roReasons(e.strip.reasons)||fmt(e.strip.stripped,0)));
  if(fin(e.rejected_submissions))lines.push(fmt(e.rejected_submissions,0)+' rejected submissions');
  return lines;
 }
 function roIntakeLines(e){
  const lines=[roIntakeText(e).replace(/^[\d,]+ submitted → /,'')],r=e.not_selected_reasons||{},ns=RO.intake[1].get(e);
  const why=[];
  if(fin(r['training-task-budget'])&&r['training-task-budget'])why.push(fmt(r['training-task-budget'],0)+' over budget');
  if(fin(r['cross-miner-task-collision'])&&r['cross-miner-task-collision'])why.push(fmt(r['cross-miner-task-collision'],0)+' duplicate task');
  if(e.strip&&e.strip.stripped)why.push(fmt(e.strip.stripped,0)+' banned/stripped');
  if(!e.training&&e.control&&fin(e.groups_sealed))why.push(fmt(e.groups_sealed,0)+' sealed but no update ('+(e.control.action||'control')+(fin(e.control.rollback_epoch)?' to E'+e.control.rollback_epoch:'')+')');
  if(fin(ns))lines.push('Not selected: '+fmt(ns,0)+' groups'+(why.length?' ('+why.join(' · ')+')':''));
  return lines;
 }
 /* Stacked epoch bars with optional right-axis line, regime bands and event markers. */
 function roChart(svg,tipEl,rows,bars,o){
  svg.replaceChildren();tipEl.hidden=true;
  const {W,H}=box(svg,480,o.height);svg.setAttribute('viewBox','0 0 '+W+' '+H);
  const L=46,R=o.line?W-44:W-10,T=o.top||12,B=H-40,w=R-L,n=rows.length;
  const vals=rows.map(r=>bars.map(s=>{const v=s.get(r);return fin(v)?v:null;}));
  const tot=vals.map(v=>v.reduce((a,b)=>a+(b||0),0));
  let hi=Math.max(1,...tot);{const mag=10**Math.floor(Math.log10(hi)),step=[1,2,2.5,5,10].map(k=>k*mag).find(k=>k*4>=hi)||mag*10;hi=Math.ceil(hi/step)*step;}
  const x=i=>L+w*(i+.5)/n,y=v=>B-(B-T)*v/hi;
  const focusI=(()=>{const s=rows.findIndex(r=>r.index===selectedIndex);return s>=0?s:n-1;})();
  // regime bands
  if(o.bands){let i=0;while(i<n){const k=o.bands(rows[i]);let j=i;while(j+1<n&&o.bands(rows[j+1])===k)j++;
   if(k){const x0=L+w*i/n,x1=L+w*(j+1)/n;svg.append(el('rect',{x:x0,y:T,width:x1-x0,height:B-T,fill:'var(--hover)',opacity:.55}));
    if(x1-x0>60)svg.append(el('text',{x:(x0+x1)/2,y:T+10,'text-anchor':'middle',opacity:.85},k));}
   i=j+1;}}
  for(const t of [0,hi/2,hi])svg.append(el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'sp-guide'}),el('text',{x:L-8,y:y(t)+4,'text-anchor':'end'},fmt(t,0)));
  const bw=Math.max(1,w/n*.78);
  rows.forEach((r,i)=>{let base=0;bars.forEach((s,si)=>{const v=vals[i][si];if(!v)return;
   svg.append(el('rect',{x:x(i)-bw/2,y:y(base+v),width:bw,height:Math.max(0,y(base)-y(base+v)),fill:'var('+s.color+')',opacity:(s.opacity||1)*(i===focusI?1:.72),class:'sp-rise'}));base+=v;});});
  if(o.line){const lv=rows.map(o.line.get),lmax=o.line.max(lv),ly=v=>B-(B-T)*v/lmax;
   for(const t of [0,lmax/2,lmax])svg.append(el('text',{x:R+6,y:ly(t)+4,'text-anchor':'start'},o.line.tick(t)));
   let d='',open=false;lv.forEach((v,i)=>{if(!fin(v)){open=false;return;}d+=(open?' L':' M')+x(i).toFixed(1)+' '+ly(v).toFixed(1);open=true;});
   svg.append(el('path',{d,class:'sp-line',stroke:'var('+o.line.color+')'}));
   lv.forEach((v,i)=>{if(fin(v))svg.append(el('circle',{cx:x(i),cy:ly(v),r:i===focusI?4:(n>40?1.6:2.5),class:'sp-dot',stroke:'var('+o.line.color+')'}));});}
  // event markers
  for(const m of o.markers||[]){const i=rows.findIndex(r=>r.index===m.index);if(i<0)continue;const xm=L+w*i/n;
   svg.append(el('line',{x1:xm,x2:xm,y1:m.label?(m.y||T)+2:T,y2:B,stroke:'var(--muted)','stroke-dasharray':'3 3','stroke-width':1}));
   const tx=el('text',{x:m.side==='left'?xm-4:xm+4,y:(m.y||T)+(m.dy||0),'text-anchor':m.side==='left'?'end':'start'},m.label);tx.style.fill='var(--ink)';svg.append(tx);}
  const every=Math.max(1,Math.ceil(n/Math.max(4,Math.round(8*W/480))));
  rows.forEach((r,i)=>{if(i===0||i===n-1||i===focusI||(i%every===0&&Math.abs(i-focusI)>every/2&&n-1-i>every/2))svg.append(el('text',{x:x(i),y:B+18,'text-anchor':'middle'},'E'+r.index));
   const hit=el('rect',{x:x(i)-w/n/2,y:T,width:w/n,height:B-T+22,fill:'transparent',tabindex:0,role:'img','aria-label':'E'+r.index+': '+o.aria(r)});
   const show=()=>{tipEl.replaceChildren(h('span','','Epoch '+r.index),h('b','',o.headline(r)));for(const line of o.lines(r))tipEl.append(h('small','',line));tipEl.hidden=false;};
   hit.addEventListener('pointerenter',show);hit.addEventListener('pointermove',show);hit.addEventListener('focus',show);
   hit.addEventListener('pointerleave',()=>tipEl.hidden=true);hit.addEventListener('blur',()=>tipEl.hidden=true);svg.append(hit);});
  svg.append(el('text',{x:(L+R)/2,y:H-5,'text-anchor':'middle'},o.axis||'Epoch'));
 }
 function roSetup(p){
  if(p.ro)return p.ro;
  const s=p.section,plot=s.querySelector('.sp-plot'),legend=s.querySelector('.sp-legend');
  const funnel=h('div','sp-funnel');plot.before(funnel);
  const sub=h('h3','sp-subhead','Intake · groups per epoch');
  const plot2=h('div','sp-plot sp-plot-intake');plot2.innerHTML='<svg role="group" aria-label="Intake groups per epoch"></svg><output class="sp-tip" hidden></output>';
  const legend2=h('div','legend sp-legend');
  const note=h('p','sp-note');
  legend.after(sub,plot2,legend2,note);
  p.ro={funnel,svg2:plot2.querySelector('svg'),tip2:plot2.querySelector('output'),legend2,note};
  return p.ro;
 }
 function roLegend(L,items,extra){L.replaceChildren();for(const it of items){const span=h('span');const i=h('i','swatch');i.style.background='var('+it.color+')';if(it.opacity)i.style.opacity=it.opacity;if(it.line){i.style.height='2px';}span.append(i,document.createTextNode(it.label));L.append(span);}if(extra){const s=h('span','',extra);L.append(s);}}
 function drawRollouts(finals){
  const p=P.rollouts,ro=roSetup(p);
  const all=finals.filter(e=>e.training||e.control||(e.phase!=='collecting'&&fin(e.groups_sealed)));
  const rows=windowed(p,all),r=focusRow(rows),t=r?.training;
  const cp=r?capPct(r):null;
  setReading(p,t&&t.rollouts?pct(t.correct/t.rollouts,1):'—',t?(fin(cp)?pct(cp,1)+' hit '+capOf(r)+' cap':'cap hits not recorded')+(fin(t.mean_output_tokens)?' · mean '+fmt(t.mean_output_tokens,0)+' tok':''):(r?.control?'no update':''),r?'Epoch '+r.index+' · '+(t?fmt(t.rollouts,0)+' rollouts · '+(RO.policy[t.unresolved_policy]||''):(r.control?'rolled back to E'+r.control.rollback_epoch:'no training result')):'');
  // funnel for the focused epoch
  ro.funnel.replaceChildren();
  if(r){const step=(k,v,sub)=>{const d=h('div','sp-fstep');d.append(h('span','',k),h('b','',v));if(sub)d.append(h('small','',sub));return d;};
   const arrow=()=>h('i','sp-farrow','→');
   const ns=r.not_selected_reasons||{};
   ro.funnel.append(
    step('Submitted',fmt(r.groups_total,0),fin(r.submissions)?fmt(r.submissions,0)+' submissions · '+fmt(r.rejected_submissions,0)+' rejected':''),arrow(),
    step('Sealed in batch',fmt(r.groups_sealed,0),fin(r.groups_not_selected)?'−'+fmt(ns['training-task-budget']||0,0)+' over budget · −'+fmt(ns['cross-miner-task-collision']||0,0)+' dup':''),arrow(),
    step('Trained',t?fmt(t.groups,0):(r.control?'0':'—'),r.strip?'−'+fmt(r.strip.stripped,0)+' stripped'+(roReasons(r.strip.reasons)?' ('+Object.keys(r.strip.reasons).join(', ')+')':''):(r.control?'rolled back to E'+r.control.rollback_epoch:'')),arrow(),
    step('Rollouts',t?fmt(t.rollouts,0):'—',t?fmt(t.correct,0)+' pass · '+fmt(t.wrong,0)+' fail · '+fmt(t.unresolved,0)+' unresolved'+(fin(t.hit_cap)?' ('+fmt(t.hit_cap,0)+' hit cap)':''):''));}
  // Clean 3-series rollouts chart (2026-10-25): Pass / Fail (wrong + unresolved) / Cap. Missing epochs render as empty slots.
  const cap=capOf(rows.find(e=>e.training?.output_token_cap)||null);
  const byIdx=new Map(rows.map(e=>[e.index,e])),idxs=rows.map(e=>e.index),slots=[];
  if(idxs.length)for(let k=Math.min(...idxs);k<=Math.max(...idxs);k++)slots.push(byIdx.get(k)||{index:k});
  const RO3=[{label:'Pass',color:'--performance',get:t=>t.correct},
   {label:'Fail',color:'--unchecked',get:t=>{const u=fin(t.unresolved_uncapped)?t.unresolved_uncapped:t.unresolved;return (fin(t.wrong)?t.wrong:0)+(fin(u)?u:0);}},
   {label:'Cap',color:'--rejected',get:t=>t.hit_cap}];
  const cp3=(r&&fin(capPct(r)))?capPct(r):([...rows].reverse().map(capPct).find(fin)??null);p.section.querySelector('.reading').textContent=fin(cp3)?pct(cp3,0)+' cap':'—';
  roChart(p.svg,p.tip,slots,RO3.map(s=>({...s,get:e=>e.training?s.get(e.training):null})),{height:230,top:12,
   headline:e=>e.training?fmt(e.training.rollouts,0)+' rollouts':(e.control?'no update':'not recorded'),lines:e=>e.training||e.control?roLines(e):[],
   aria:e=>e.training?['pass '+e.training.correct,'fail '+RO3[1].get(e.training),'cap '+(fin(e.training.hit_cap)?e.training.hit_cap:'not recorded')].join(', '):'no training update'});
  roLegend(p.section.querySelector('.sp-legend'),RO3);
  roChart(ro.svg2,ro.tip2,rows,RO.intake,{height:230,
   headline:e=>fmt(e.groups_total,0)+' groups submitted',lines:e=>roIntakeLines(e),
   aria:e=>roIntakeText(e)});
  roLegend(ro.legend2,RO.intake);
  ro.note.textContent='Every sampled rollout is counted. Passed = correct answer, failed = wrong answer, unresolved = no gradable answer (including outputs truncated at the '+cap+'-token cap). Since E125 unresolved rollouts are trained with reward 0; shaded epochs dropped or masked them instead. Intake bars total the groups listed in sealed submissions; only the trained slice entered the update. A missing bar or line point means that epoch did not record the field.';
 }

 function drawEpochCharts(){
  const finals=epochsAll();
  const trained=finals.filter(e=>e.training);
  // Groups
  {const p=P.groups,rows=windowed(p,finals.filter(e=>fin(e.groups_total))),r=focusRow(rows);
   setReading(p,r?fmt(r.groups_total,0):'—','groups',r?'Epoch '+r.index+' · '+r.phase:'');
   chart(p,rows,[{label:'Groups',type:'bar',color:'--accepted',get:e=>e.groups_total,fmt:v=>fmt(v,0)}],{tick:v=>fmt(v,0),headline:e=>fmt(e.groups_total,0)+' groups',extra:e=>[fmt(e.submissions,0)+' sealed submissions',fin(e.max_tasks)?fmt(e.max_tasks,0)+' groups trained (max)':null]});}
  // Miners
  {const p=P.miners,rows=windowed(p,finals.filter(e=>fin(e.submitting_miners))),r=focusRow(rows);
   setReading(p,r?fmt(r.submitting_miners,0):'—','miners',r?'Epoch '+r.index+(fin(r.eligible_miners)?' · '+fmt(r.eligible_miners,0)+' registered':''):'');
   chart(p,rows,[{label:'Submitting miners',type:'bar',color:'--incentive',get:e=>e.submitting_miners,fmt:v=>fmt(v,0)}],{tick:v=>fmt(v,0),extra:e=>[fmt(e.eligible_miners,0)+' registered miners',fmt(e.rejected_submissions,0)+' rejected submissions']});}
  // Rollouts (intake funnel + outcomes)
  drawRollouts(finals);
  // Grad norm
  {const p=P.grad,rows=windowed(p,trained),r=focusRow(rows);
   setReading(p,r?fmt(r.training.grad_norm,4):'—','',r?'Epoch '+r.index+' · step '+fmt(r.training.optimizer_step,0):'');
   chart(p,rows,[{label:'Grad norm',type:'line',color:'--incentive',get:e=>e.training.grad_norm,fmt:v=>fmt(v,4)}],{min:0,tick:v=>fmt(v,3)});}
  // Loss
  {const p=P.loss,rows=windowed(p,trained),r=focusRow(rows);
   setReading(p,r?fmt(r.training.loss,5):'—','',r?'Epoch '+r.index:'');
   chart(p,rows,[{label:'Loss',type:'line',color:'--rejected',get:e=>e.training.loss,fmt:v=>fmt(v,5)}],{tick:v=>fmt(v,3)});}
  // Timings (minutes)
  {const p=P.timing,src=finals.filter(e=>e.timing&&fin(e.timing.collect_seconds)),rows=p.view===0?src.slice(-20):src,r=rows.find(e=>e.index===selectedIndex)||rows.filter(e=>fin(e.timing.total_seconds)).pop(),T=r?.timing||{};
   const total=fin(T.total_seconds)?T.total_seconds:null;
   setReading(p,total!==null?mins(total)+' min':'—',fin(T.cycle_seconds)?'next epoch after '+mins(T.cycle_seconds)+' min':r?r.phase:'',r?'Epoch '+r.index+' · start → finalized':'');
   const m=k=>e=>fin(e.timing[k])?e.timing[k]/60:null;
   const tot=e=>fin(e.timing.total_seconds)&&e.timing.total_seconds>0?e.timing.total_seconds/60:null,trn=e=>fin(e.timing.train_seconds)?e.timing.train_seconds/60:0;
   const gat=e=>tot(e)===null?null:Math.min(fin(e.timing.collect_seconds)?e.timing.collect_seconds/60:0,tot(e)),tr2=e=>tot(e)===null?null:Math.min(trn(e),tot(e)-gat(e));
   const gp=e=>{if(tot(e)===null)return null;const g=gateOf(e.index),d=g?gspan(g.timestamps?.post_probe):null;return d===null?0:Math.max(0,Math.min(d,tot(e)-gat(e)-tr2(e)));};
   chart(p,rows,[{label:'Gather',type:'bar',color:'--incentive',get:gat,fmt:v=>fmt(v,0)+' min'},{label:'Train',type:'bar',color:'--performance',get:tr2,fmt:v=>fmt(v,0)+' min'},{label:'Post-probe (gate)',type:'bar',color:'--accepted',get:gp,fmt:v=>fmt(v,0)+' min'},{label:'Other',type:'bar',color:'--unchecked',get:e=>tot(e)===null?null:Math.max(0,tot(e)-gat(e)-tr2(e)-(gp(e)||0)),fmt:v=>fmt(v,0)+' min'}],{stacked:true,tick:v=>fmt(v,0)+'m',headline:e=>fin(e.timing.total_seconds)?mins(e.timing.total_seconds)+' min':e.phase,extra:e=>['Prepare '+fmt(m('seal_prepare_seconds')(e),0)+' · finalize '+fmt(m('finalize_seconds')(e),0)+' min',utc(e.start)+' start',...gateLines(gateOf(e.index))]});}
 }


 /* ---- Env probe (/probe/series.json aggregates only) ---- */
 /* Chart is driven by the configured miner-env list (today mining.json has only affine_math).
    Historical multi-env points stay in series.json; we plot whatever history exists for these keys. */
 const MINER_ENVS=['affine_math'];
 const ENV_PALETTE=['#85c8b4','#86baff','#d6b580','#d18e91','#a78bfa','#67e8f9','#fbbf24','#fb7185','#34d399','#60a5fa','#c4b5fd','#fda4af','#94a3b8','#f0abfc'];
 const envShort=name=>String(name||'').replace(/^affine_/,'');
 const envColor=(names,name)=>ENV_PALETTE[Math.max(0,names.indexOf(name))%ENV_PALETTE.length];
 let probeData=null;

 function probePoints(){
  const pts=(probeData?.points||[]).filter(p=>fin(p.epoch_index)&&p.envs&&typeof p.envs==='object');
  return pts.slice().sort((a,b)=>a.epoch_index-b.epoch_index||(a.finished_at||0)-(b.finished_at||0));
 }
 function probeEnvNames(pts){
  /* Prefer configured miner envs that appear in any point; fall back to latest-point keys if list empty. */
  const present=new Set();
  for(const p of pts)for(const k of Object.keys(p.envs||{}))present.add(k);
  const fromMiner=MINER_ENVS.filter(e=>present.has(e));
  if(fromMiner.length)return fromMiner.slice();
  const latest=pts[pts.length-1];
  return latest?Object.keys(latest.envs||{}).sort((a,b)=>envShort(a).localeCompare(envShort(b))):[];
 }
 function probePrimary(point,names){
  if(!point)return null;
  if(names.length===1){const e=point.envs?.[names[0]];return fin(e?.p)?e.p:null;}
  const vals=names.map(n=>point.envs?.[n]?.p).filter(fin);
  return vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null;
 }
 function probeLegend(names,extra){
  const L=P.probetrend.section.querySelector('.sp-legend');L.replaceChildren();
  for(const name of names){
   const span=h('span');const i=h('i','swatch');i.style.background=envColor(names,name);
   span.append(i,document.createTextNode(envShort(name)));L.append(span);
  }
  if(extra){const s=h('span','');s.textContent=extra;L.append(s);}
 }

 function drawProbeTrends(p,pts,names){
  const svg=p.svg;svg.replaceChildren();
  const {W,H}=box(svg);svg.setAttribute('viewBox','0 0 '+W+' '+H);p.tip.hidden=true;
  const usable=pts.some(pt=>names.some(n=>fin(pt.envs?.[n]?.p)));
  empty(p,usable?null:'No probe measurements yet.');if(!usable)return;
  const L=40,R=W-10,T=10,B=H-36,w=R-L,n=Math.max(pts.length,1);
  const x=i=>L+w*(i+.5)/n,y=v=>B-(B-T)*v;
  for(const t of [0,.25,.5,.75,1]){
   svg.append(el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'sp-guide'}));
   svg.append(el('text',{x:L-8,y:y(t)+4,'text-anchor':'end'},Math.round(t*100)+'%'));
  }
  names.forEach(name=>{
   const color=envColor(names,name);let d='',open=false;
   pts.forEach((pt,i)=>{const v=pt.envs?.[name]?.p;if(!fin(v)){open=false;return;}d+=(open?' L':' M')+x(i).toFixed(1)+' '+y(v).toFixed(1);open=true;});
   if(d)svg.append(el('path',{d,class:'sp-line probe-line',stroke:color}));
   pts.forEach((pt,i)=>{
    const e=pt.envs?.[name];if(!fin(e?.p))return;
   });
  });
  const every=Math.max(1,Math.ceil(n/Math.max(4,Math.round(7*W/480))));
  pts.forEach((pt,i)=>{
   if(i===0||i===n-1||i%every===0)svg.append(el('text',{x:x(i),y:B+16,'text-anchor':'middle'},'E'+pt.epoch_index));
   const hit=el('rect',{x:x(i)-w/n/2,y:T,width:Math.max(w/n,8),height:B-T+20,fill:'transparent',tabindex:0,role:'img','aria-label':'E'+pt.epoch_index});
   const show=()=>{
    const lines=names.map(name=>{const e=pt.envs?.[name];if(!e||!fin(e.p))return null;return envShort(name)+': '+pct(e.p,1)+' ±'+pct(e.se,1)+' · n='+fmt(e.n,0)+(fin(e.u)?' · u='+fmt(e.u,0):'');}).filter(Boolean);
    const head=pct(probePrimary(pt,names),1)+(names.length===1?'':' mean');
    tip(p,'Epoch '+pt.epoch_index,head,[...lines,'Finished '+ago(pt.finished_at),'step '+fmt(pt.optimizer_step,0)]);
   };
   hit.addEventListener('pointerenter',show);hit.addEventListener('pointermove',show);hit.addEventListener('focus',show);
   hit.addEventListener('pointerleave',()=>p.tip.hidden=true);hit.addEventListener('blur',()=>p.tip.hidden=true);svg.append(hit);
  });
  svg.append(el('text',{x:(L+R)/2,y:H-5,'text-anchor':'middle'},'Epoch'));
 }

 /* Net-change rows: pre-gate from the probe series (checkpoint vs parent), gated from the gate ledger (C vs H on the gate sample). */
 function netRows(){
  const ents=(gateData?.entries||[]).filter(g=>fin(g.epoch)&&fin(g.candidate_pass)&&fin(g.head_pass));
  const firstGate=ents.length?Math.min(...ents.map(g=>g.epoch)):Infinity;
  const pts=probePoints(),names=probeEnvNames(pts),rows=[],seen=new Set();
  pts.forEach((pt,i)=>{
   const id=pt.checkpoint_id;if(!id||seen.has(id))return;seen.add(id);
   const ep=pt.epoch_index-1,step=pt.optimizer_step;if(!(ep<firstGate)||!fin(step))return;
   let par=null;for(let j=i-1;j>=0;j--){if(pts[j].optimizer_step===step-1&&pts[j].checkpoint_id!==id){par=pts[j];break;}}
   const pc=probePrimary(pt,names),pp=probePrimary(par,names);if(!par||!fin(pc)||!fin(pp))return;
   const e=names.length===1?pt.envs?.[names[0]]:null,ep0=names.length===1?par.envs?.[names[0]]:null;
   rows.push({index:ep,kind:'pre',net:pc-pp,c:pc,hp:pp,cid:id,hid:par.checkpoint_id,n:e?.n,hn:ep0?.n,step,probeEpoch:pt.epoch_index,parentEpoch:par.epoch_index});
  });
  for(const g of ents)rows.push({index:g.epoch,kind:'gate',net:g.candidate_pass-g.head_pass,c:g.candidate_pass,hp:g.head_pass,cid:g.candidate,hid:g.head,n:g.n,g});
  return rows.sort((a,b)=>a.index-b.index);
 }
 const pp=(v,d=2)=>fin(v)?(v>0?'+':v<0?'−':'')+Math.abs(100*v).toFixed(d)+' pp':'—';
 const ruleText=t=>t?'z '+(t.z_inclusive?'≥ ':'> ')+fmt(t.z_min,1)+' · cap-hit < '+pct(t.cap_max,0)+' · n ≥ '+fmt(t.min_tasks,0):'—';
 /* Net change bars in the shared chart style (t695u): linear axis clipped to ±3 pp, capped off-scale bars with a small
    chevron, verdict as a small dot above gated bars (filled = promoted, ring = rejected) and a hollow bar for rejected. */
 function drawNet(p,rows){
  const svg=p.svg,{W,H}=box(svg);svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+W+' '+H);p.tip.hidden=true;
  empty(p,rows.length?null:'No net-change measurements yet.');if(!rows.length)return;
  const LIM=0.03,L=46,R=W-10,T=12,B=H-40,w=R-L,n=rows.length,y=v=>B-(B-T)*(Math.max(-LIM,Math.min(LIM,v))+LIM)/(2*LIM),x=i=>L+w*(i+.5)/n;
  for(const t of [-LIM,0,LIM])svg.append(el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:t===0?'sp-zero':'sp-guide'}),el('text',{x:L-8,y:y(t)+4,'text-anchor':'end'},t===0?'0':pp(t,0)));
  const g0=rows.findIndex(r=>r.kind==='gate');
  if(g0>0){const gx=L+w*g0/n;svg.append(el('line',{x1:gx,x2:gx,y1:T,y2:B,class:'sp-guide'}),el('text',{x:gx+4,y:T+8,'text-anchor':'start'},'gate'));}
  const sel=rows.findIndex(r=>r.index===selectedIndex),focus=sel>=0?sel:n-1,bw=Math.max(1,w/n*.78);
  rows.forEach((r,i)=>{
   const pos=r.net>=0,c='var(--'+(pos?'performance':'rejected')+')',y0=y(0),y1=y(r.net),top=Math.min(y0,y1),hgt=Math.max(1,Math.abs(y1-y0));
   const rej=r.kind==='gate'&&r.g.verdict!=='promoted',op=r.kind==='pre'?(i===focus?.6:.35):(i===focus?1:.72);
   const bar=el('rect',{x:x(i)-bw/2,y:top,width:bw,height:hgt,class:'sp-rise'});
   bar.style.fill=rej?'transparent':c;bar.style.opacity=op;if(rej){bar.style.stroke=c;bar.style.strokeWidth='1.2';}
   svg.append(bar);
   if(Math.abs(r.net)>LIM&&bw>=5){const ey=pos?T:B,d=pos?-1:1,hx=Math.min(4,bw/2);  /* off-scale: capped bar + chevron */
    const m=el('path',{d:'M'+(x(i)-hx)+' '+(ey-d*0)+' L'+x(i)+' '+(ey+d*4)+' L'+(x(i)+hx)+' '+(ey-d*0),fill:'none'});m.style.stroke=c;m.style.strokeWidth='1.2';m.style.opacity=op;svg.append(m);}
   if(r.kind==='gate'){const ok=!rej,dy=Math.max(T+3,top-5),dot=el('circle',{cx:x(i),cy:pos?dy:Math.max(T+3,y0-5),r:2.5});
    const vc='var(--'+(ok?'performance':'rejected')+')';dot.style.fill=ok?vc:'transparent';dot.style.stroke=vc;dot.style.strokeWidth='1.2';svg.append(dot);}
  });
  const every=Math.max(1,Math.ceil(n/Math.max(4,Math.round(8*W/480))));
  rows.forEach((r,i)=>{
   if(i===0||i===n-1||i===focus||(i%every===0&&Math.abs(i-focus)>every/2&&n-1-i>every/2))svg.append(el('text',{x:x(i),y:B+18,'text-anchor':'middle'},'E'+r.index));
   const hit=el('rect',{x:x(i)-w/n/2,y:T,width:w/n,height:B-T+22,fill:'transparent',tabindex:0,role:'img','aria-label':'E'+r.index+' net '+pp(r.net)+(r.kind==='gate'?' '+r.g.verdict:' pre-gate')});
   const show=()=>{
    if(r.kind==='gate'){const g=r.g,ok=g.verdict==='promoted';
     tip(p,'Epoch '+r.index+' · '+(ok?'promoted':'rejected'),pp(r.net),['C '+String(r.cid).slice(0,8)+' '+pct(r.c,2)+' vs H '+String(r.hid).slice(0,8)+' '+pct(r.hp,2)+' · '+fmt(g.n,0)+' gate-sample tasks, batch removed',
      'z '+fmt(g.z,2)+' · cap-hit '+pct(g.candidate_cap_rate,1)+' · n '+fmt(g.n,0),'Threshold: '+ruleText(g.thresholds),
      (!ok&&r.net>=0)||(ok&&r.net<0)?'The gate tests paired z (pass unless clearly worse), not the sign of the net':null].filter(Boolean));}
    else tip(p,'Epoch '+r.index+' · pre-gate',pp(r.net),['Full probe '+String(r.cid).slice(0,8)+' '+pct(r.c,2)+' vs parent '+String(r.hid).slice(0,8)+' '+pct(r.hp,2),
      'Independent task draws; no gate verdict'+(Math.abs(r.net)>LIM?' · off scale (axis clipped at ±3 pp)':'')]);
   };
   hit.addEventListener('pointerenter',show);hit.addEventListener('pointermove',show);hit.addEventListener('focus',show);
   hit.addEventListener('pointerleave',()=>p.tip.hidden=true);hit.addEventListener('blur',()=>p.tip.hidden=true);svg.append(hit);
  });
  svg.append(el('text',{x:(L+R)/2,y:H-5,'text-anchor':'middle'},'Epoch'));
 }
 function drawProbe(){
  const p=P.probe;p.svg.style.display='';
  p.section.querySelector('.sp-plot .sp-table')?.remove();
  const rows=netRows();
  if(!rows.length){setReading(p,'—','',probeData?'No net-change data yet':'Probe series unavailable');empty(p,probeData?'No net-change data yet.':'Probe series unavailable.');p.svg.replaceChildren();return;}
  const r=rows.find(x=>x.index===selectedIndex)||rows[rows.length-1];
  setReading(p,pp(r.net),'','');
  drawNet(p,rows);
 }

 function drawGate(){
  const p=P.gate,plot=p.section.querySelector('.sp-plot');plot.querySelector('.sp-table')?.remove();
  const ents=(gateData?.entries||[]).map(g=>Object.assign({index:g.epoch},g)),cur=gateData?.current,c=gateData?.counts||{};
  const curTxt=cur&&fin(cur.epoch)?'E'+cur.epoch+' '+(cur.stage||'')+(cur.candidate?' · C '+String(cur.candidate).slice(0,8):''):'';
  const r=ents.find(g=>g.index===selectedIndex)||ents[ents.length-1];
  setReading(p,r?(r.verdict==='promoted'?'GREEN':'RED'):'—',r?'E'+r.index+' · z '+fmt(r.z,2)+' · cap '+pct(r.candidate_cap_rate,1):'',(fin(c.total)?fmt(c.promoted,0)+' promoted · '+fmt(c.rejected,0)+' rejected':'')+(curTxt?' · now '+curTxt:''));
  if(!ents.length){p.svg.style.display='';p.svg.replaceChildren();empty(p,gateData?'No verdicts yet. Pass unless clearly worse (z ≥ −1.5, cap-hit < 20%). Live from E152'+(curTxt?' · now '+curTxt:'')+'.':'Gate ledger unavailable.');return;}
  /* Grid tiles hide view buttons: the expanded tile (chart-expand.js) shows the full verdict table. */
  if(p.view===1||p.section.classList.contains('cx-open')){p.svg.style.display='none';empty(p,null);
   const t=h('table','sp-table');t.innerHTML='<thead><tr><th>Epoch</th><th>Verdict</th><th>H → C</th><th>z</th><th>Cap-hit</th><th>n</th><th>Decided</th><th>Reason</th></tr></thead>';const tb=h('tbody');
   for(const g of ents.slice().reverse()){const tr=h('tr');['E'+g.epoch,g.verdict==='promoted'?'GREEN promoted':'RED rejected',String(g.head).slice(0,8)+' → '+String(g.candidate).slice(0,8),fmt(g.z,2),pct(g.candidate_cap_rate,1),fmt(g.n,0),utc(g.timestamps?.decided_at),g.reason||'—'].forEach(v=>tr.append(h('td','',v)));tb.append(tr);}
   t.append(tb);plot.append(t);return;}
  p.svg.style.display='';
  chart(p,ents,[{label:'Promoted · cap-hit',type:'bar',color:'--performance',get:g=>g.verdict==='promoted'?g.candidate_cap_rate:null,fmt:v=>fin(v)?pct(v,1):'—'},{label:'Rejected · cap-hit',type:'bar',color:'--rejected',get:g=>g.verdict==='rejected'?g.candidate_cap_rate:null,fmt:v=>fin(v)?pct(v,1):'—'}],{min:0,tick:v=>pct(v,0),headline:g=>g.verdict==='promoted'?'GREEN promoted':'RED rejected',extra:g=>[...gateLines(g),g.reason?'Reason: '+g.reason:null]});
 }

 /* Performance (Jacob t697u): held-out solve rate (site-metrics.json heldout, 32 fixed problems) and env probe pass rate
    (probe/series.json) per checkpoint, aligned on the training epoch that produced the checkpoint (probe head epoch − 1). */
 function perfRows(){
  const by=new Map(),key=id=>String(id||'').slice(0,12);
  for(const h of (data?.heldout||[]).filter(r=>fin(r.score)&&fin(r.produced_by_epoch))){
   const k=key(h.checkpoint)||('h'+h.produced_by_epoch);if(!by.has(k))by.set(k,{index:h.produced_by_epoch,ck:k});Object.assign(by.get(k),{ho:h.score,hoN:h.count,hoS:h.successes,hoAt:h.evaluated_at});}
  const pts=probePoints(),names=probeEnvNames(pts);
  for(const pt of pts){const k=key(pt.checkpoint_id),v=probePrimary(pt,names);if(!fin(v))continue;
   if(!by.has(k))by.set(k,{index:pt.epoch_index-1,ck:k});const o=by.get(k);if(fin(o.pr))continue;
   const e=names.length===1?pt.envs?.[names[0]]:null;Object.assign(o,{pr:v,prN:e?.n,prSe:e?.se,prAt:pt.finished_at,step:pt.optimizer_step});}
  return [...by.values()].sort((a,b)=>a.index-b.index||(a.prAt||a.hoAt||0)-(b.prAt||b.hoAt||0));
 }
 function drawProbeTrend(){
  const p=P.probetrend,rows=perfRows();p.svg.style.display='';
  if(!rows.length){setReading(p,'—','','Performance data unavailable');empty(p,'Performance data unavailable.');p.svg.replaceChildren();return;}
  const lastPr=rows.filter(r=>fin(r.pr)).pop();
  setReading(p,lastPr?pct(lastPr.pr,1):'—','',lastPr?'Env probe, checkpoint from E'+lastPr.index:'');
  chart(p,rows,[{key:'ho',label:'Held-out',type:'line',color:'--performance',get:r=>fin(r.ho)?r.ho:null,fmt:v=>fin(v)?pct(v,1):'—'},
    {key:'pr',label:'Env probe',type:'line',color:'--incentive',get:r=>fin(r.pr)?r.pr:null,fmt:v=>fin(v)?pct(v,1):'—'}],
   {min:0,max:1,connect:3,tick:v=>Math.round(v*100)+'%',title:r=>'Checkpoint from E'+r.index+(r.ck?' · '+r.ck.slice(0,8):''),axis:'Checkpoint (training epoch that produced it)',
    headline:r=>fin(r.pr)?pct(r.pr,1)+' probe':fin(r.ho)?pct(r.ho,1)+' held-out':'—',
    extra:r=>[fin(r.ho)?'Held-out: '+r.hoS+' / '+r.hoN+' fixed problems':'Held-out: not evaluated',fin(r.pr)?'Env probe: n='+fmt(r.prN,0)+' ±'+pct(r.prSe,1)+' · step '+fmt(r.step,0):'Env probe: not probed']});
  p.setLegend([['Held-out (32 fixed)','--performance'],['Env probe (fresh tasks)','--incentive']]);
 }

 function render(){try{drawGate();}catch(error){console.error('site-metrics gate',error);}try{drawProbe();}catch(error){console.error('site-metrics net',error);}try{drawProbeTrend();}catch(error){console.error('site-metrics probe',error);}if(data){for(const f of [drawWeights,drawMining,drawEpochCharts]){try{f();}catch(error){console.error('site-metrics',error);}}}}

 // Follow the shared desktop epoch slider (same rows as network.js).
 let sharedRows=[];
 window.addEventListener('affine:snapshot',e=>{sharedRows=(e.detail?.epochs||[]).filter(r=>r.source==='live-reward-math'&&fin(r.start)&&Number.isInteger(r.batches)).sort((a,b)=>a.start-b.start);});
 const slider=document.getElementById('desktop-epoch-slider');
 const follow=()=>{const row=sharedRows[Number(slider.value)];const m=row&&String(row.id).match(/-e(\d+)$/);selectedIndex=m?Number(m[1]):null;render();};
 slider?.addEventListener('input',follow);
 document.getElementById('desktop-epoch-previous')?.addEventListener('click',()=>setTimeout(follow));
 document.getElementById('desktop-epoch-next')?.addEventListener('click',()=>setTimeout(follow));
 window.addEventListener('affine:theme',render);
 window.addEventListener('affine:chart-resize',render);

 let fingerprint='';
 async function refresh(){
  try{const r=await fetch('/site-metrics.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('unavailable');
   const next=await r.json();const key=JSON.stringify([next.epochs,next.heldout,next.weights,next.mining]);data=next;
   if(key!==fingerprint){fingerprint=key;render();}else{drawMining();try{drawProbe();drawProbeTrend();}catch{}}
  }catch{if(!data)for(const p of Object.values(P)){if(p.id==='probe')continue;setReading(p,'—','','Data temporarily unavailable');empty(p,'Data temporarily unavailable.');}}
 }
 let probeFingerprint='';
 async function refreshProbe(){
  try{const r=await fetch('/probe/series.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('unavailable');
   const next=await r.json();const key=JSON.stringify([next.updated_at,next.points?.length,next.points?.slice(-1)?.[0]?.checkpoint_id]);
   probeData=next;if(key!==probeFingerprint){probeFingerprint=key;drawProbe();try{drawProbeTrend();}catch{}}
  }catch{if(!probeData){{setReading(P.probe,'—','','Probe series unavailable');empty(P.probe,'Probe series unavailable.');}}}
 }
 let gateFingerprint='';
 async function refreshGate(){
  try{const r=await fetch('/gate.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('unavailable');
   const next=await r.json();const key=JSON.stringify([next.entries?.length,next.entries?.slice(-1),next.current]);
   gateData=next;if(key!==gateFingerprint){gateFingerprint=key;render();}
  }catch{if(!gateData){try{drawGate();}catch{}}}
 }
 refresh();refreshProbe();refreshGate();setInterval(refresh,30000);setInterval(refreshProbe,60000);setInterval(refreshGate,30000);
})();
