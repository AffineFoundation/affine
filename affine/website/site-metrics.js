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
  heldout:panel('heldout','Held-out score','performance',['All','Last 20'],[['Held-out solve rate per checkpoint','--performance']]),
  mining:panel('mining','Mining status','incentive',[],[]),
  groups:panel('groups','Groups per epoch','accepted',['All','Last 20'],[['Fresh groups','--accepted'],['Reused groups','--unchecked']]),
  miners:panel('miners','Miners per epoch','incentive',['All','Last 20'],[['Miners with a sealed submission','--incentive']]),
  rollouts:panel('rollouts','Training rollouts','performance',['All','Last 20'],[['Kept','--performance'],['Dropped unresolved','--rejected']]),
  grad:panel('grad','Gradient norm','incentive',['All','Last 20'],[['Pre-clip gradient norm per optimizer step','--incentive']]),
  loss:panel('loss','Training loss','rejected',['All','Last 20'],[['Objective loss per optimizer step','--rejected']]),
  timing:panel('timing','Epoch timings','unchecked',['Last 20','All'],[['Collect','--incentive'],['Wait + prepare','--unchecked'],['Train','--performance'],['Finalize','--accepted']]),
 };
 place(P.weights,'#main>.incentive-section');
 place(P.heldout,'#main>section[aria-labelledby="evaluation-title"]');
 place(P.mining,'#panel-heldout');
 place(P.groups,'#main>section[aria-labelledby="batch-title"]');
 for(const k of ['miners','rollouts','grad','loss','timing'])main.append(P[k].section);

 let data=null,selectedIndex=null;
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
  const svg=p.svg;svg.replaceChildren();svg.setAttribute('viewBox','0 0 480 230');p.tip.hidden=true;
  const vals=rows.map(r=>series.map(s=>s.get(r)));
  const usable=vals.some(v=>v.some(fin));empty(p,usable?null:(opts.emptyText||'No measurements yet.'));if(!usable)return;
  const stackTotals=vals.map(v=>opts.stacked?v.reduce((a,b)=>a+(fin(b)?b:0),0):Math.max(...v.filter(fin)));
  let lo=opts.min!==undefined?opts.min:Math.min(0,...vals.flat().filter(fin));
  let hi=opts.max!==undefined?opts.max:Math.max(...stackTotals.filter(fin));
  if(hi===lo)hi=lo+1;
  if(opts.max===undefined){const span=hi-lo,mag=10**Math.floor(Math.log10(span)),step=[1,2,2.5,5,10].map(n=>n*mag).find(n=>n*4>=span)||mag*10;hi=Math.ceil(hi/step)*step;if(lo<0)lo=Math.floor(lo/step)*step;}
  const L=46,R=470,T=12,B=190,w=R-L,n=rows.length,x=i=>L+w*(i+.5)/n,y=v=>B-(B-T)*(v-lo)/(hi-lo);
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
   if(s.type!=='line')return;let d='',open=false;
   rows.forEach((r,i)=>{const v=vals[i][si];if(!fin(v)){open=false;return;}d+=(open?' L':' M')+x(i).toFixed(1)+' '+y(v).toFixed(1);open=true;});
   svg.append(el('path',{d,class:'sp-line',stroke:'var('+s.color+')'}));
   rows.forEach((r,i)=>{const v=vals[i][si];if(fin(v))svg.append(el('circle',{cx:x(i),cy:y(v),r:i===focus?4:(n>40?1.6:2.5),class:'sp-dot',stroke:'var('+s.color+')'}));});
  });
  const label=opts.label||(r=>'E'+r.index);
  const every=Math.max(1,Math.ceil(n/8));
  rows.forEach((r,i)=>{if(i===0||i===n-1||i===focus||(i%every===0&&Math.abs(i-focus)>every/2&&n-1-i>every/2))svg.append(el('text',{x:x(i),y:B+18,'text-anchor':'middle'},label(r)));
   const hit=el('rect',{x:x(i)-w/n/2,y:T,width:w/n,height:B-T+22,fill:'transparent',tabindex:0,role:'img','aria-label':label(r)+': '+series.map((s,si)=>s.label+' '+fmt(vals[i][si],4)).join(', ')});
   const show=()=>{const lines=series.map((s,si)=>s.label+': '+(s.fmt?s.fmt(vals[i][si]):fmt(vals[i][si],4)));if(opts.extra)lines.push(...[].concat(opts.extra(r)).filter(Boolean));tip(p,opts.title?opts.title(r):'Epoch '+r.index,opts.headline?opts.headline(r):(series[0].fmt?series[0].fmt(vals[i][0]):fmt(vals[i][0],4)),lines);};
   hit.addEventListener('pointerenter',show);hit.addEventListener('pointermove',show);hit.addEventListener('focus',show);
   hit.addEventListener('pointerleave',()=>p.tip.hidden=true);hit.addEventListener('blur',()=>p.tip.hidden=true);
   svg.append(hit);});
  svg.append(el('text',{x:(L+R)/2,y:225,'text-anchor':'middle'},opts.axis||'Epoch'));
 }

 function focusRow(rows){return rows.find(r=>r.index===selectedIndex)||rows[rows.length-1];}

 function drawWeights(){
  const p=P.weights,w=data?.weights;
  if(!w||!Array.isArray(w.paid)){setReading(p,'—','', 'Weights unavailable');empty(p,'v9 weights unavailable.');p.svg.replaceChildren();return;}
  const a=w.assessment||{},chain=w.chain||{},perDay=fin(chain.miner_emission_rao_per_tempo)&&fin(chain.tempo)?chain.miner_emission_rao_per_tempo*(7200/chain.tempo)/1e9:null;
  setReading(p,String(w.paid.length),'paid UIDs',(w.chain_status==='submitted'?'Set on chain':'Status '+(w.chain_status||'—'))+' · block '+fmt(w.block,0)+' · '+ago(w.run_at));
  const sm=a.smoothing;
  const L=p.section.querySelector('.sp-legend');L.replaceChildren(h('span','',(a.rule||'v9')+(sm?' · '+sm.rule+' half-life '+sm.half_life_epochs+' epochs · E'+sm.epochs_used?.[0]+'–E'+sm.epochs_used?.[1]:'')+' · scored E'+fmt(a.score_epoch,0)+' · '+fmt(a.eligible_uids,0)+' eligible UIDs · 100% of weight to miners'));
  const svg=p.svg;svg.replaceChildren();p.tip.hidden=true;
  const rows=w.paid.slice(0,24);empty(p,rows.length?null:'No UID is paid in this window.');
  const plot=p.section.querySelector('.sp-plot');
  plot.querySelector('.sp-table')?.remove();
  if(p.view===1){svg.style.display='none';
   const t=h('table','sp-table');t.innerHTML='<thead><tr><th>UID</th><th>Weight</th><th>Est. α / day</th><th>Last tempo α</th></tr></thead>';const tb=h('tbody');
   for(const r of w.paid){const tr=h('tr');[String(r.uid),pct(r.weight,2),perDay?fmt(r.weight*perDay,1):'—',fin(r.last_tempo_emission_rao)?fmt(r.last_tempo_emission_rao/1e9,2):'—'].forEach(v=>tr.append(h('td','',v)));tb.append(tr);}
   t.append(tb);plot.append(t);return;}
  svg.style.display='';
  const height=Math.max(230,22+rows.length*16);svg.setAttribute('viewBox','0 0 480 '+height);
  const max=Math.max(...rows.map(r=>r.weight),0.01),L0=58,R0=400;
  rows.forEach((r,i)=>{const yy=10+i*16,wd=(R0-L0)*r.weight/max;
   svg.append(el('text',{x:L0-8,y:yy+10,'text-anchor':'end'},'UID '+r.uid),el('rect',{x:L0,y:yy,width:Math.max(1,wd),height:12,fill:'var(--accepted)',class:'sp-grow'}),el('text',{x:L0+wd+6,y:yy+10},pct(r.weight,1)));
   const hit=el('rect',{x:0,y:yy-2,width:480,height:16,fill:'transparent',tabindex:0,role:'img','aria-label':'UID '+r.uid+' weight '+pct(r.weight,2)});
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
  const L=p.section.querySelector('.sp-legend');L.replaceChildren(h('span','',(last?last.count+' fixed held-out problems per checkpoint':'Held-out')+(lastEpoch&&last&&lastEpoch.index>last.produced_by_epoch?' · E'+lastEpoch.index+' checkpoint pending':'')));
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

 function drawEpochCharts(){
  const finals=epochsAll();
  const trained=finals.filter(e=>e.training);
  // Groups
  {const p=P.groups,rows=windowed(p,finals.filter(e=>fin(e.groups_total))),r=focusRow(rows);
   setReading(p,r?fmt(r.groups_total,0):'—',r?pct(r.groups_total?r.groups_fresh/r.groups_total:null,0)+' fresh':'',r?'Epoch '+r.index+' · '+r.phase:'');
   chart(p,rows,[{label:'Fresh',type:'bar',color:'--accepted',get:e=>e.groups_fresh,fmt:v=>fmt(v,0)},{label:'Reused',type:'bar',color:'--unchecked',get:e=>e.groups_reused,fmt:v=>fmt(v,0)}],{stacked:true,tick:v=>fmt(v,0),headline:e=>fmt(e.groups_total,0)+' groups',extra:e=>[fmt(e.submissions,0)+' sealed submissions',fin(e.max_tasks)?fmt(e.max_tasks,0)+' groups trained (max)':null]});}
  // Miners
  {const p=P.miners,rows=windowed(p,finals.filter(e=>fin(e.submitting_miners))),r=focusRow(rows);
   setReading(p,r?fmt(r.submitting_miners,0):'—','miners',r?'Epoch '+r.index+(fin(r.eligible_miners)?' · '+fmt(r.eligible_miners,0)+' registered':''):'');
   chart(p,rows,[{label:'Submitting miners',type:'bar',color:'--incentive',get:e=>e.submitting_miners,fmt:v=>fmt(v,0)}],{tick:v=>fmt(v,0),extra:e=>[fmt(e.eligible_miners,0)+' registered miners',fmt(e.rejected_submissions,0)+' rejected submissions']});}
  // Rollouts
  {const p=P.rollouts,rows=windowed(p,trained),r=focusRow(rows),t=r?.training||{};
   setReading(p,r&&fin(t.kept_rollouts)?fmt(t.kept_rollouts,0):'—',r&&fin(t.rollouts)&&t.rollouts?pct(t.kept_rollouts/t.rollouts,0)+' kept':'',r?'Epoch '+r.index+(t.drop_filter?' · unresolved rollouts dropped':''):'');
   chart(p,rows,[{label:'Kept',type:'bar',color:'--performance',get:e=>e.training.kept_rollouts,fmt:v=>fmt(v,0)},{label:'Dropped',type:'bar',color:'--rejected',get:e=>e.training.dropped_rollouts,fmt:v=>fmt(v,0)}],{stacked:true,tick:v=>fmt(v,0),headline:e=>fmt(e.training.rollouts,0)+' rollouts',extra:e=>[fmt(e.training.groups,0)+' groups · '+fmt(e.training.correct,0)+' correct · '+fmt(e.training.wrong,0)+' wrong · '+fmt(e.training.unresolved,0)+' unresolved']});}
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
   chart(p,rows,[{label:'Collect',type:'bar',color:'--incentive',get:m('collect_seconds'),fmt:v=>fmt(v,0)+' min'},{label:'Wait + prepare',type:'bar',color:'--unchecked',get:m('seal_prepare_seconds'),fmt:v=>fmt(v,0)+' min'},{label:'Train',type:'bar',color:'--performance',get:m('train_seconds'),fmt:v=>fmt(v,0)+' min'},{label:'Finalize',type:'bar',color:'--accepted',get:m('finalize_seconds'),fmt:v=>fmt(v,0)+' min'}],{stacked:true,tick:v=>fmt(v,0)+'m',headline:e=>fin(e.timing.total_seconds)?mins(e.timing.total_seconds)+' min':e.phase,extra:e=>[utc(e.start)+' start']});}
 }

 function render(){if(!data)return;for(const f of [drawWeights,drawHeldout,drawMining,drawEpochCharts]){try{f();}catch(error){console.error('site-metrics',error);}}}

 // Follow the shared desktop epoch slider (same rows as network.js).
 let sharedRows=[];
 window.addEventListener('affine:snapshot',e=>{sharedRows=(e.detail?.epochs||[]).filter(r=>r.source==='live-reward-math'&&fin(r.start)&&Number.isInteger(r.batches)).sort((a,b)=>a.start-b.start);});
 const slider=document.getElementById('desktop-epoch-slider');
 const follow=()=>{const row=sharedRows[Number(slider.value)];const m=row&&String(row.id).match(/-e(\d+)$/);selectedIndex=m?Number(m[1]):null;render();};
 slider?.addEventListener('input',follow);
 document.getElementById('desktop-epoch-previous')?.addEventListener('click',()=>setTimeout(follow));
 document.getElementById('desktop-epoch-next')?.addEventListener('click',()=>setTimeout(follow));
 window.addEventListener('affine:theme',render);

 let fingerprint='';
 async function refresh(){
  try{const r=await fetch('/site-metrics.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('unavailable');
   const next=await r.json();const key=JSON.stringify([next.epochs,next.heldout,next.weights,next.mining]);data=next;
   if(key!==fingerprint){fingerprint=key;render();}else drawMining();
  }catch{if(!data)for(const p of Object.values(P)){setReading(p,'—','','Data temporarily unavailable');empty(p,'Data temporarily unavailable.');}}
 }
 refresh();setInterval(refresh,30000);
})();
