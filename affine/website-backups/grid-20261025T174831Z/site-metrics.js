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
  probe:panel('probe','Env probe','performance',[],[]),
  mining:panel('mining','Mining status','incentive',[],[]),
  groups:panel('groups','Groups per epoch','accepted',['All','Last 20'],[['Fresh groups','--accepted'],['Reused groups','--unchecked']]),
  miners:panel('miners','Miners per epoch','incentive',['All','Last 20'],[['Miners with a sealed submission','--incentive']]),
  rollouts:panel('rollouts','Rollouts per epoch','performance',['All','Last 20'],[]),
  grad:panel('grad','Gradient norm','incentive',['All','Last 20'],[['Pre-clip gradient norm per optimizer step','--incentive']]),
  loss:panel('loss','Training loss','rejected',['All','Last 20'],[['Objective loss per optimizer step','--rejected']]),
  timing:panel('timing','Epoch timings','unchecked',['Last 20','All'],[['Collect','--incentive'],['Wait + prepare','--unchecked'],['Train','--performance'],['Finalize','--accepted']]),
 };
 place(P.weights,'#main>.incentive-section');
 place(P.heldout,'#main>section[aria-labelledby="evaluation-title"]');
 place(P.mining,'#panel-heldout');
 place(P.groups,'#main>section[aria-labelledby="batch-title"]');
 for(const k of ['miners','rollouts','grad','loss','timing'])main.append(P[k].section);
 /* Large Env probe trend chart at bottom of homepage (multi-env ready; driven by miner env list). */
 main.append(P.probe.section);
 P.probe.section.classList.add('probe-hero');
 (()=>{const note=h('p','sp-note');note.append(
   document.createTextNode('Pass rate vs epoch · fresh random tasks per checkpoint · seeded after publish · miner envs only (today: affine_math) · not the held-out evaluator. '),
   Object.assign(document.createElement('a'),{href:'/probe/series.json',textContent:'series.json'}),
   document.createTextNode(' · '),
   Object.assign(document.createElement('a'),{href:'/probe/index.json',textContent:'index.json'})
 );P.probe.section.append(note);})();

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

 /* ---- Rollouts per epoch: intake funnel (groups) + outcome of every sampled rollout ----
    Gaps (missing bars / broken line) mean the epoch's artifacts do not record that field. */
 const RO={
  out:[
   {key:'passed',label:'Passed',color:'--performance',get:t=>t.correct},
   {key:'failed',label:'Failed (wrong answer)',color:'--accepted',get:t=>t.wrong},
   {key:'unres',label:'Unresolved, under cap',color:'--unchecked',get:t=>t.unresolved_uncapped},
   {key:'unk',label:'Unresolved, cap split not recorded',color:'--unchecked',opacity:.55,get:t=>fin(t.unresolved_uncapped)?null:t.unresolved},
   {key:'cap',label:'Hit output cap',color:'--rejected',get:t=>t.hit_cap},
  ],
  intake:[
   {key:'trained',label:'Trained',color:'--performance',get:e=>e.training?e.training.groups:null},
   {key:'stripped',label:'Stripped (ban / TOPLOC gate)',color:'--rejected',get:e=>e.strip?e.strip.stripped:null},
   {key:'held',label:'Sealed, not trained (rollback)',color:'--accepted',get:e=>!e.training&&e.control&&fin(e.groups_sealed)?e.groups_sealed:null},
   {key:'dup',label:'Not selected: duplicate task',color:'--unchecked',get:e=>e.not_selected_reasons?e.not_selected_reasons['cross-miner-task-collision']||0:null},
   {key:'budget',label:'Not selected: over group budget',color:'--incentive-low',get:e=>e.not_selected_reasons?e.not_selected_reasons['training-task-budget']||0:null},
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
 /* Stacked epoch bars with optional right-axis line, regime bands and event markers. */
 function roChart(svg,tipEl,rows,bars,o){
  svg.replaceChildren();tipEl.hidden=true;
  const H=o.height,W=480;svg.setAttribute('viewBox','0 0 '+W+' '+H);
  const L=46,R=o.line?436:470,T=o.top||12,B=H-40,w=R-L,n=rows.length;
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
  const every=Math.max(1,Math.ceil(n/8));
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
  const markers=[{index:124,label:'E124 rolled back to E95 ┐',side:'left',y:11},{index:125,label:'E125+ unresolved trained at reward 0 ┐',side:'left',y:24}];
  const cap=capOf(rows.find(e=>e.training?.output_token_cap)||null);
  roChart(p.svg,p.tip,rows,RO.out.map(s=>({...s,get:e=>e.training?s.get(e.training):null})),{height:262,top:36,
   bands:e=>e.training?.unresolved_policy==='dropped'?'unresolved dropped':e.training?.unresolved_policy==='masked'?'unresolved masked':null,
   line:{get:capPct,color:'--incentive',max:v=>{const m=Math.max(.1,...v.filter(fin));return Math.ceil(m*20)/20;},tick:v=>Math.round(v*100)+'%'},
   markers,headline:e=>e.training?fmt(e.training.rollouts,0)+' rollouts':'no update',lines:roLines,
   aria:e=>e.training?['passed '+e.training.correct,'failed '+e.training.wrong,'unresolved '+e.training.unresolved,'hit cap '+(fin(e.training.hit_cap)?e.training.hit_cap:'not recorded')].join(', '):'no training update'});
  roLegend(p.section.querySelector('.sp-legend'),[...RO.out.map(s=>s.key==='cap'?{...s,label:'Hit '+cap+'-token cap'}:s).filter(s=>s.key!=='unk'||rows.some(e=>e.training&&!fin(e.training.unresolved_uncapped))),{label:'% of rollouts hitting the cap (right axis)',color:'--incentive',line:true}]);
  roChart(ro.svg2,ro.tip2,rows,RO.intake,{height:190,markers:[{index:124,label:'',side:'left'},{index:125,label:'',side:'right'}],
   headline:e=>fmt(e.groups_total,0)+' groups submitted',lines:e=>[roIntakeText(e),...roLines(e).slice(-3)],
   aria:e=>roIntakeText(e)});
  roLegend(ro.legend2,RO.intake.filter(s=>s.key!=='held'||rows.some(e=>!e.training&&e.control)));
  ro.note.textContent='Every sampled rollout is counted. Passed = correct answer, failed = wrong answer, unresolved = no gradable answer (including outputs truncated at the '+cap+'-token cap). Since E125 unresolved rollouts are trained with reward 0; shaded epochs dropped or masked them instead. Intake bars total the groups listed in sealed submissions; only the trained slice entered the update. A missing bar or line point means that epoch did not record the field.';
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
   chart(p,rows,[{label:'Collect',type:'bar',color:'--incentive',get:m('collect_seconds'),fmt:v=>fmt(v,0)+' min'},{label:'Wait + prepare',type:'bar',color:'--unchecked',get:m('seal_prepare_seconds'),fmt:v=>fmt(v,0)+' min'},{label:'Train',type:'bar',color:'--performance',get:m('train_seconds'),fmt:v=>fmt(v,0)+' min'},{label:'Finalize',type:'bar',color:'--accepted',get:m('finalize_seconds'),fmt:v=>fmt(v,0)+' min'}],{stacked:true,tick:v=>fmt(v,0)+'m',headline:e=>fin(e.timing.total_seconds)?mins(e.timing.total_seconds)+' min':e.phase,extra:e=>[utc(e.start)+' start']});}
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
  const L=P.probe.section.querySelector('.sp-legend');L.replaceChildren();
  for(const name of names){
   const span=h('span');const i=h('i','swatch');i.style.background=envColor(names,name);
   span.append(i,document.createTextNode(envShort(name)));L.append(span);
  }
  if(extra){const s=h('span','');s.textContent=extra;L.append(s);}
 }

 function drawProbeTrends(p,pts,names){
  const svg=p.svg;svg.replaceChildren();
  const W=960,H=420;svg.setAttribute('viewBox','0 0 '+W+' '+H);p.tip.hidden=true;
  const usable=pts.some(pt=>names.some(n=>fin(pt.envs?.[n]?.p)));
  empty(p,usable?null:'No probe measurements yet.');if(!usable)return;
  const L=64,R=W-24,T=24,B=H-48,w=R-L,n=Math.max(pts.length,1);
  const x=i=>L+w*(i+.5)/n,y=v=>B-(B-T)*v;
  for(const t of [0,.25,.5,.75,1]){
   svg.append(el('line',{x1:L,x2:R,y1:y(t),y2:y(t),class:'sp-guide'}));
   svg.append(el('text',{x:L-10,y:y(t)+4,'text-anchor':'end'},Math.round(t*100)+'%'));
  }
  names.forEach(name=>{
   const color=envColor(names,name);let d='',open=false;
   pts.forEach((pt,i)=>{const v=pt.envs?.[name]?.p;if(!fin(v)){open=false;return;}d+=(open?' L':' M')+x(i).toFixed(1)+' '+y(v).toFixed(1);open=true;});
   if(d)svg.append(el('path',{d,class:'sp-line probe-line',stroke:color}));
   pts.forEach((pt,i)=>{
    const e=pt.envs?.[name];if(!fin(e?.p))return;
    if(i===n-1&&fin(e.se)&&e.se>0){
     svg.append(el('line',{x1:x(i),x2:x(i),y1:y(Math.max(0,e.p-e.se)),y2:y(Math.min(1,e.p+e.se)),stroke:color,'stroke-width':'1.5',opacity:'.55'}));
    }
    svg.append(el('circle',{cx:x(i),cy:y(e.p),r:i===n-1?5:3.2,class:'sp-dot',stroke:color}));
   });
  });
  const every=Math.max(1,Math.ceil(n/10));
  pts.forEach((pt,i)=>{
   if(i===0||i===n-1||i%every===0)svg.append(el('text',{x:x(i),y:B+22,'text-anchor':'middle'},'E'+pt.epoch_index));
   const hit=el('rect',{x:x(i)-w/n/2,y:T,width:Math.max(w/n,8),height:B-T+28,fill:'transparent',tabindex:0,role:'img','aria-label':'E'+pt.epoch_index});
   const show=()=>{
    const lines=names.map(name=>{const e=pt.envs?.[name];if(!e||!fin(e.p))return null;return envShort(name)+': '+pct(e.p,1)+' ±'+pct(e.se,1)+' · n='+fmt(e.n,0)+(fin(e.u)?' · u='+fmt(e.u,0):'');}).filter(Boolean);
    const head=pct(probePrimary(pt,names),1)+(names.length===1?'':' mean');
    tip(p,'Epoch '+pt.epoch_index,head,[...lines,'Finished '+ago(pt.finished_at),'step '+fmt(pt.optimizer_step,0)]);
   };
   hit.addEventListener('pointerenter',show);hit.addEventListener('pointermove',show);hit.addEventListener('focus',show);
   hit.addEventListener('pointerleave',()=>p.tip.hidden=true);hit.addEventListener('blur',()=>p.tip.hidden=true);svg.append(hit);
  });
  svg.append(el('text',{x:(L+R)/2,y:H-8,'text-anchor':'middle'},'Epoch'));
 }

 function drawProbe(){
  const p=P.probe;p.svg.style.display='';
  p.section.querySelector('.sp-plot .sp-table')?.remove();
  const pts=probePoints();
  if(!pts.length){setReading(p,'—','','Probe series unavailable');empty(p,'Probe series unavailable.');p.svg.replaceChildren();probeLegend([]);return;}
  const names=probeEnvNames(pts),cur=pts[pts.length-1],primary=probePrimary(cur,names);
  const curEnv=names.length===1?cur.envs?.[names[0]]:null;
  const nLabel=names.length===1
    ? (envShort(names[0])+' · n='+fmt(curEnv?.n,0)+' · epoch E'+cur.epoch_index+' · updated '+ago(probeData.updated_at||cur.finished_at))
    : (names.length+' envs · epoch E'+cur.epoch_index+' · updated '+ago(probeData.updated_at||cur.finished_at));
  const sub=names.length===1?'pass rate':(names.length?'mean pass rate':'');
  setReading(p,pct(primary,1),sub,nLabel);
  probeLegend(names,'step '+fmt(cur.optimizer_step,0));
  drawProbeTrends(p,pts,names);
 }

 function render(){if(data){for(const f of [drawWeights,drawHeldout,drawMining,drawEpochCharts]){try{f();}catch(error){console.error('site-metrics',error);}}}try{drawProbe();}catch(error){console.error('site-metrics probe',error);}}

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
   if(key!==fingerprint){fingerprint=key;render();}else{drawMining();try{drawProbe();}catch{}}
  }catch{if(!data)for(const p of Object.values(P)){if(p.id==='probe')continue;setReading(p,'—','','Data temporarily unavailable');empty(p,'Data temporarily unavailable.');}}
 }
 let probeFingerprint='';
 async function refreshProbe(){
  try{const r=await fetch('/probe/series.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!r.ok)throw Error('unavailable');
   const next=await r.json();const key=JSON.stringify([next.updated_at,next.points?.length,next.points?.slice(-1)?.[0]?.checkpoint_id]);
   probeData=next;if(key!==probeFingerprint){probeFingerprint=key;drawProbe();}
  }catch{if(!probeData){setReading(P.probe,'—','','Probe series unavailable');empty(P.probe,'Probe series unavailable.');}}
 }
 refresh();refreshProbe();setInterval(refresh,30000);setInterval(refreshProbe,60000);
})();
