/* K=1 per-registration dashboard tiles, 2026-10-28 (Jacob t759u). Data: /k1.json, /scores.json, /mining.json. No other endpoint changes. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const $=(t,a={},h='')=>{const e=document.createElement(t);for(const k in a)e.setAttribute(k,a[k]);if(h)e.innerHTML=h;return e;};
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const sh=h=>h?String(h).slice(0,8):'—', hk=h=>h?String(h).slice(0,6)+'…'+String(h).slice(-4):'—';
 const pct=x=>x==null?'—':(100*x).toFixed(1)+'%', fz=z=>z==null?'—':(z>=0?'+':'')+Number(z).toFixed(2);
 const NS='http://www.w3.org/2000/svg';
 const svgEl=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 function tile(id,title,cls='',reading=''){
  let s=document.getElementById(id);if(s)return s;
  s=$('section',{id,class:'site-panel k1-tile '+cls,'aria-labelledby':id+'-t'});
  s.innerHTML='<div class="section-heading"><h2 id="'+id+'-t">'+title+'</h2><div class="measurement-reading"><span class="reading" id="'+id+'-r">'+reading+'</span></div></div>';
  return s;
 }
 function pill(v){const c=v==='green'?'green':v==='red'?'red':v==='skip'?'skip':'pending';return '<span class="k1-pill '+c+'">'+esc(v==='green'?'GREEN':v==='red'?'RED':v||'pending')+'</span>';}
 function tip(wrap){let t=wrap.querySelector('.chart-tip');if(!t){t=$('output',{class:'chart-tip','aria-live':'polite'});t.hidden=true;wrap.append(t);}return t;}
 /* 2026-10-29 (Jacob t774u): only the first and last epoch label on each x-axis, anchored to the slot edges so they never overlap or clip. */
 function edgeLabels(svg,count,slotX,slotW,y,label){if(!count)return;const at=(i,a,x)=>{const t=svgEl('text',{x,y,'text-anchor':a});t.textContent=label(i);svg.append(t);};
  if(count===1){at(0,'start',slotX(0));return;}at(0,'start',slotX(0));at(count-1,'end',slotX(count-1)+slotW);}
 function barChart(wrap,items,opt){
  wrap.querySelector('svg')?.remove();
  const W=wrap.clientWidth||400,H=wrap.clientHeight||220,EXT=(opt.refSteps&&opt.refNow!=null)?46:0,P={l:40,r:(opt.right?40:10)+EXT,t:10,b:22};
  const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':opt.label});wrap.prepend(svg);
  if(!items.length){const e=wrap.querySelector('.empty')||wrap.appendChild($('p',{class:'empty'}));e.textContent=opt.empty;return;}
  wrap.querySelector('.empty')?.remove();
  const lo=Math.min(0,...items.map(i=>i.v),opt.min??0),hi=Math.max(...items.map(i=>i.v),opt.max??0,1e-9);
  const y=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/(hi-lo||1));
  for(const g of opt.ticks(lo,hi)){svg.append(svgEl('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:g===0?'k1-axis':'k1-grid'}));const t=svgEl('text',{x:P.l-6,y:y(g)+3,'text-anchor':'end'});t.textContent=opt.fmt(g);svg.append(t);}
  const n=Math.max(items.length,opt.slots||0),bw=(W-P.l-P.r)/n,tp=tip(wrap);
  items.forEach((it,i)=>{
   const x=P.l+i*bw+bw*.18,w=Math.max(2,bw*.64),y0=y(0),y1=y(it.v);
   const r=svgEl('rect',{x,y:Math.min(y0,y1),width:w,height:Math.max(1,Math.abs(y1-y0)),fill:it.c,rx:1});
   if(it.hollow){r.setAttribute('fill','none');r.setAttribute('stroke',it.c);}
   svg.append(r);
   if(it.dot!=null&&opt.right){const yy=P.t+(H-P.t-P.b)*(1-it.dot/(opt.rightMax||1));svg.append(svgEl('circle',{cx:x+w/2,cy:yy,r:3,fill:'var(--bg)',stroke:'var(--accepted)','stroke-width':1.5}));}
   const hit=svgEl('rect',{x:P.l+i*bw,y:P.t,width:bw,height:H-P.t-P.b,fill:'transparent'});
   hit.addEventListener('mouseenter',()=>{tp.innerHTML=it.tip;tp.hidden=false;const L=Math.min(Math.max(0,x-60),W-220);tp.style.left=L+'px';tp.style.top='4px';});
   hit.addEventListener('mouseleave',()=>{tp.hidden=true;});svg.append(hit);
  });
  if(opt.refSteps){const yv=v=>y(Math.min(hi,Math.max(lo,v)));let d='',pv=null,px=null;const xc=i=>P.l+i*bw+bw/2;
   opt.refSteps.forEach((v,i)=>{if(v==null){pv=null;return;}d+=(pv==null?'M':'L')+`${xc(i)},${yv(v)}`;pv=v;px=xc(i);});
   if(d)svg.append(svgEl('path',{d,fill:'none',stroke:'var(--performance)','stroke-width':1.6,'stroke-linejoin':'round',opacity:.9}));
   if(opt.refNow!=null){const x0=px??P.l,x1=W-P.r+EXT-2;
    svg.append(svgEl('path',{d:`M${x0},${yv(pv??opt.refNow)}L${x1},${yv(opt.refNow)}`,fill:'none',stroke:'var(--performance)','stroke-width':1.6,'stroke-dasharray':'3 3',opacity:.9}));
    svg.append(svgEl('circle',{cx:x1,cy:yv(opt.refNow),r:2.5,fill:'var(--performance)'}));
    const t=svgEl('text',{x:x1,y:yv(opt.refNow)-6,'text-anchor':'end',fill:'var(--performance)','font-size':9,stroke:'var(--bg)','stroke-width':3,'paint-order':'stroke'});t.textContent=opt.refLabel||'';svg.append(t);}}
  else if(opt.ref!=null&&opt.ref>=lo&&opt.ref<=hi){const yr=y(opt.ref);svg.append(svgEl('line',{x1:P.l,x2:W-P.r,y1:yr,y2:yr,stroke:'var(--performance)','stroke-dasharray':'4 3','stroke-width':1,opacity:.8}));const t=svgEl('text',{x:W-P.r-2,y:yr-4,'text-anchor':'end',fill:'var(--performance)','font-size':9,stroke:'var(--bg)','stroke-width':3,'paint-order':'stroke'});t.textContent=opt.refLabel||'';svg.append(t);}
  edgeLabels(svg,items.length,i=>P.l+i*bw,bw,H-6,i=>items[i].x);
  if(opt.right){for(const [v,yy] of [[opt.rightMax,P.t],[0,H-P.b]]){const t=svgEl('text',{x:W-P.r+4,y:yy+3,fill:'var(--accepted)'});t.textContent=v;svg.append(t);}}
 }
 function render(k1,scores,mining,lrj,zbj){
  const turns=(k1.turns||[]).slice().sort((a,b)=>a.epoch_index-b.epoch_index);
  const done=turns.filter(t=>t.status==='turn');
  const q=k1.queue||{};const payees=k1.payees||[];
  const anchor=main.firstElementChild;
  // 1. status strip
  const st=tile('panel-k1-status','K=1 network','k1-wide');
  const head=(mining.training&&mining.training.head)||{};const last=turns[turns.length-1]||{};
  const ep=(mining.training&&mining.training.epoch_index)||'';
  const HK=q.queue_rule==='hotkey-once'||q.queue_rule==='hotkey-registration-once';
  const RULE={'hotkey-registration-once':'Each hotkey gets exactly one turn, permanently, and so does each registration (UID + registration block). Pass or fail, the oldest-registered unserved entry with a valid batch file (until the old-format cutoff, 09:00 UTC Oct 31, also a ready old-format batch) takes the turn; entries without one are skipped, not charged, and keep their place. To try again, register a new hotkey; don\'t deregister a paying hotkey. ','hotkey-once':'Each hotkey gets exactly one turn, permanently, pass or fail. The oldest-registered unserved hotkey with a ready batch (≥8 tasks) at the seal trains; others are skipped and keep their place. Register a new hotkey to try again; don\'t deregister a paying hotkey. '}[q.queue_rule]||'One turn per registration (UID + hotkey + registration block), pass or fail. Oldest unserved registration with a ready batch (≥8 tasks) at the seal trains; others are skipped and keep their place. Register again to submit again. ';
  const greens=done.filter(t=>t.verdict==='green').length,reds=done.filter(t=>t.verdict==='red').length;
  st.querySelector('.reading').innerHTML=pill(last.verdict||last.status);
  st.querySelector('dl')?.remove();st.querySelector('.k1-rule')?.remove();
  st.insertAdjacentHTML('beforeend','<dl class="k1-stats">'+
   `<div><dt>Head</dt><dd title="${esc(head.checkpoint)}">${sh(head.checkpoint)}<small>step ${esc(head.optimizer_step??'—')}</small></dd></div>`+
   `<div><dt>Epoch</dt><dd>E${esc(ep)}<small>${esc(mining.status||'')}</small></dd></div>`+
   `<div><dt>Last turn</dt><dd>${last.uid!=null?'UID '+esc(last.uid):'—'}<small>E${esc(last.epoch_index??'—')}</small></dd></div>`+
   `<div><dt>Turns</dt><dd>${done.length}<small>${greens} green · ${reds} red</small></dd></div>`+
   `<div><dt>In line</dt><dd>${esc(q.queue_length??'—')}<small>ready at E${esc(q.readiness_epoch??'—')}</small></dd></div>`+
   (HK?`<div><dt>Hotkeys</dt><dd>${esc(q.hotkeys_unserved??'—')}<small>unserved · ${esc(q.hotkeys_served??'—')} served</small></dd></div>`:`<div><dt>Registrations</dt><dd>${esc(q.registrations_unserved??'—')}<small>unserved · ${esc(q.registrations_served??'—')} served</small></dd></div>`)+
   `<div><dt>Payees</dt><dd>${payees.length}<small>of 10 · 1/N each</small></dd></div>`+
   '</dl><p class="k1-rule">'+RULE+'GREEN = held-out z &gt; 1 and cap-hit &lt; 20%; the last 10 distinct GREEN UIDs share the weight. <a href="/llms.txt">Rules</a> · <a href="/k1.json">k1.json</a></p>'+(mining.batch_file?'<p class="k1-rule"><strong>Mine:</strong> upload one batch file ('+esc(mining.batch_file.format)+': prompts, your rollouts, your rewards) to your private per-hotkey object, any time. Format, rules and upload: <a href="/mining.json">mining.json</a> → batch_file · helper <a href="/ff/affine_batch.py">affine_batch.py</a> · <a href="/ff/status.json">file checks</a>'+(mining.old_format_cutoff&&Date.now()/1000<mining.old_format_cutoff.at_unix?' · old-format batches trained until '+esc(mining.old_format_cutoff.at_cr):'')+'</p>':''));
  // 2. next in line
  const qu=tile('panel-k1-queue','Next in line');qu.querySelector('.reading').textContent=q.queue_length??'—';
  qu.querySelector('.flip-table-wrap')?.remove();
  qu.insertAdjacentHTML('beforeend','<div class="flip-table-wrap"><table class="flip-table"><thead><tr><th class="num">#</th><th class="num">UID</th><th>Hotkey</th><th class="num">Reg block</th><th class="num">Ready</th><th class="num">Order</th></tr></thead><tbody>'+
   ((q.next||[]).map((r,i)=>`<tr><td class="num">${r.line??i}</td><td class="num">${esc(r.uid)}</td><td class="k1-hash" title="${esc(r.hotkey)}">${hk(r.hotkey)}</td><td class="num">${esc(r.registration_block)}</td><td class="num">${r.file_valid?'file':r.old_format_tasks_last_seal?esc(r.old_format_tasks_last_seal)+' old':esc(r.last_seal_tasks??'—')}</td><td class="num">${esc(r.position)}</td></tr>`).join('')||'<tr><td colspan="6" class="flip-loading">No unserved registration has a valid batch file (or a ready old-format batch) now.</td></tr>')+
   '</tbody></table></div>');
  qu.querySelector("h2").title=(q.queue_rule==='hotkey-registration-once'?'Unserved entries (one turn per hotkey and per UID + registration block, permanently)':'Unserved hotkeys (one turn per hotkey, permanently)')+' with a valid batch file now (Ready = file; until the old-format cutoff also an old-format batch at the last seal, Ready = its task count), in turn order (oldest registration block first). Order = position among all unserved hotkeys/registrations. A forecast: the turn decides.';
  // 3. payees
  const pa=tile('panel-k1-payees','Payees (last 10 GREEN)');
  const share={};for(const m of (scores.miners||[]))share[m.uid]=m.share_now;
  pa.querySelector('.reading').textContent=payees.length;pa.querySelector('.flip-table-wrap')?.remove();
  pa.insertAdjacentHTML('beforeend','<div class="flip-table-wrap"><table class="flip-table"><thead><tr><th class="num">UID</th><th>Weight</th><th>Hotkey</th><th class="num">GREEN at</th><th class="num">z</th></tr></thead><tbody>'+
   (payees.map(p=>{const s=share[p.uid]||0;return `<tr><td class="num">${esc(p.uid)}</td><td><span class="k1-bar" style="width:${Math.round(60*s)}px"></span>${pct(s)}</td><td class="k1-hash" title="${esc(p.hotkey)}">${hk(p.hotkey)}${p.still_registered===false?' (dereg)':''}</td><td class="num">E${esc(p.green_epoch)}</td><td class="num">${fz(p.z)}</td></tr>`;}).join('')||'<tr><td colspan="5" class="flip-loading">No GREEN turn yet (pre-K=1 floor applies).</td></tr>')+
   '</tbody></table></div>');
  pa.querySelector("h2").title='Weight on chain (scores.json, set hourly): 1/N to each of the last up to 10 distinct GREEN UIDs. RED and skipped turns earn nothing.';
  // 4. gate z per turn
  const gz=tile('panel-k1-z','Gate z per turn');const gw=gz.querySelector('.k1-chart')||gz.appendChild($('div',{class:'chart-wrap k1-chart'}));
  gz.querySelector('.reading').textContent=done.length?fz(done[done.length-1].z):'—';
  if(!gz.querySelector('.legend'))gz.insertAdjacentHTML('beforeend','<div class="legend chart-legend"><span><i class="swatch" style="background:var(--performance)"></i>GREEN</span><span><i class="swatch" style="background:var(--rejected)"></i>RED</span><span><i class="swatch" style="border:1px solid var(--accepted)"></i>Pending</span><span><i class="swatch" style="width:12px;height:0;border-top:2px solid var(--performance);background:none"></i>z bar</span></div>');
  gz.querySelector("h2").title='z = gain/SE of candidate vs head on the held-out round (prior-v1 baseline). GREEN needs gain > 0, z > z bar and cap-hit < 33%. z bar (line) = the exact bar each turn was judged against: max(0.25, 1.0 - 0.75 × h/12), h = hours since the last GREEN (clock: the candidate\'s training-result time), back to 1.00 after a GREEN; the dashed end runs to the bar now (gate.json z_bar).';
  // 5. pass rates
  const pr=tile('panel-k1-pass','Pass rate per turn');const pw=pr.querySelector('.k1-chart')||pr.appendChild($('div',{class:'chart-wrap k1-chart'}));
  if(!pr.querySelector('.legend'))pr.insertAdjacentHTML('beforeend','<div class="legend chart-legend"><span><i class="swatch" style="background:var(--guide)"></i>Head</span><span><i class="swatch" style="background:var(--performance)"></i>Candidate</span></div>');
  pr.querySelector('.reading').textContent=done.length?pct(done[done.length-1].c_pass):'—';
  pr.querySelector('h2').title='One slot per epoch; head and candidate pass rate of each decided turn, as the gate scored them (wrong, unresolved and cap hits fail). To E270: the turn’s ~1,000-task private gate sample. From E272 (held-out gate): the round’s kept held-out tasks (~160–210), a harder set, so the level drops; published when the round closes. Gaps: no turn that epoch, E274 fail-closed (no evaluation), the open round, or a pending decision. Hover for details.';
  // 6. batch size per turn
  const bs=tile('panel-k1-batch','Batch per turn');const bwrap=bs.querySelector('.k1-chart')||bs.appendChild($('div',{class:'chart-wrap k1-chart'}));
  if(!bs.querySelector('.legend'))bs.insertAdjacentHTML('beforeend','<div class="legend chart-legend"><span><i class="swatch" style="background:var(--incentive)"></i>Rollouts</span></div>');
  bs.querySelector('.reading').textContent=done.length?(done[done.length-1].tasks+' × '+(done[done.length-1].rollouts/done[done.length-1].tasks|0)):'—';
  // 6b. learning rate per turn (Jacob pt1039u 2026-11-01; data /k1/lr.json from affine-training-params)
  const lrT=(lrj&&lrj.turns)||[];const lrBy=new Map(lrT.map(r=>[r.epoch,r]));
  const fe=v=>{if(v==null)return'—';const e=Math.floor(Math.log10(v)),m=v/Math.pow(10,e);return(Math.abs(m-Math.round(m))<1e-9?Math.round(m):+m.toFixed(2))+'e'+e;};
  const lt=tile('panel-k1-lr','Learning rate per turn');const lwrap=lt.querySelector('.k1-chart')||lt.appendChild($('div',{class:'chart-wrap k1-chart'}));
  if(!lt.querySelector('.legend'))lt.insertAdjacentHTML('beforeend','<div class="legend chart-legend"><span><i class="swatch" style="background:var(--performance)"></i>GREEN</span><span><i class="swatch" style="background:var(--rejected)"></i>RED</span><span><i class="swatch" style="border:1px solid var(--accepted)"></i>Pending</span><span><i class="swatch" style="border:1px dashed var(--rejected)"></i>Miner lr private (RED)</span></div>');
  {const lastLr=[...lrT].reverse().find(r=>r.lr!=null);lt.querySelector('.reading').textContent=lastLr?fe(lastLr.lr):'—';}
  lt.querySelector('h2').title='The learning rate the trainer applied to each turn batch (trainer report TRAINING-RESULT: train_params.lr, else the optimizer lr). Default 5e-7; a miner may set lr in [1.25e-7, 1e-6] in an affine-batch-v2 file. A miner-chosen lr on a RED turn stays private, like the rest of its train params (hollow marker at the bottom). Log scale.';
  // 7. lineage
  const li=tile('panel-k1-lineage','Head lineage');li.querySelector('.k1-lineage')?.remove();
  let rows=[],cur=null;
  for(const t of done){if(cur!==t.head){rows.push(`<li><span>E${t.epoch_index}</span><span class="k1-node">head ${sh(t.head)}</span><span></span></li>`);cur=t.head;}
   const g=t.verdict==='green';rows.push(`<li><span>E${t.epoch_index}</span><span class="k1-node">UID ${esc(t.uid)} → ${g?'<b>'+sh(t.candidate)+'</b>':'<s>'+sh(t.candidate)+'</s>'}</span>${pill(t.verdict)}</li>`);if(g)cur=t.candidate;}
  li.querySelector('.reading').textContent=sh(head.checkpoint);
  li.insertAdjacentHTML('beforeend','<ul class="k1-lineage">'+rows.reverse().join('')+'</ul>');
  li.querySelector("h2").title='Each turn trains one step from the current head. GREEN promotes the candidate to head; RED discards it (struck through).';
  // 8. turns table
  const tt=tile('panel-k1-turns','Turns','k1-wide');tt.querySelector('.reading').textContent=turns.length;tt.querySelector('.flip-table-wrap')?.remove();
  tt.insertAdjacentHTML('beforeend','<div class="flip-table-wrap"><table class="flip-table"><thead><tr><th class="num">Epoch</th><th class="num">UID</th><th>Hotkey</th><th class="num">Reg block</th><th>Rule</th><th class="num">Skipped</th><th class="num">Tasks</th><th class="num">Rollouts</th><th class="num">Head pass</th><th class="num">Cand pass</th><th class="num">z</th><th class="num">Cap-hit</th><th>Verdict</th><th>Head → candidate</th><th>Decided</th></tr></thead><tbody>'+
   turns.slice().reverse().map(t=>t.status!=='turn'?`<tr><td class="num">E${t.epoch_index}</td><td colspan="4">${esc(t.reason||'no turn')}</td><td class="num">${esc(t.skipped_not_ready??'—')}</td><td colspan="7"></td><td>${pill('skip')}</td><td></td></tr>`:
   `<tr><td class="num">E${t.epoch_index}</td><td class="num">${esc(t.uid)}</td><td class="k1-hash" title="${esc(t.hotkey)}">${hk(t.hotkey)}</td><td class="num">${esc(t.reg_block??'—')}</td><td class="k1-hash">${esc({'hotkey-registration-once':'hotkey + reg once','hotkey-once':'per hotkey','registration-once':'per registration','fewest-turns':'fewest-turns'}[t.queue_rule]||esc(t.queue_rule||'—'))}</td><td class="num">${esc(t.skipped_not_ready??'—')}</td><td class="num">${esc(t.tasks)}</td><td class="num">${esc(t.rollouts)}</td><td class="num">${pct(t.h_pass)}</td><td class="num">${pct(t.c_pass)}</td><td class="num">${fz(t.z)}</td><td class="num">${pct(t.cap)}</td><td>${pill(t.verdict)}</td><td class="k1-hash">${sh(t.head)} → ${sh(t.candidate)}</td><td class="k1-hash">${esc((t.decided_at_cr||'').replace(' CR (UTC-6)','').slice(5))}</td></tr>`).join('')+
   '</tbody></table></div>');
  // place tiles: status first, then charts, then tables; ahead of existing tiles
  st.remove();/* pt931u 2026-10-31: K=1 network text tile removed from the page */for(const s of [gz,pr,bs,lt,qu,pa,li,tt]){if(s.parentNode!==main||s.dataset.placed!=='1'){anchor&&anchor!==s?anchor.before(s):main.append(s);s.dataset.placed='1';}}
  // charts (after layout)
  requestAnimationFrame(()=>{
   const lab=t=>'E'+t.epoch_index;
   const pg=(mining.training&&mining.training.promotion_gate)||{},ho=pg.heldout||{};const zbar=Number((zbj&&zbj.current)??(ho.current_round&&ho.current_round.bar_z)??ho.bar_z??1);
   const zbE=new Map(((zbj&&zbj.turns)||[]).map(r=>[r.epoch,r.z_bar]));const zsteps=done.map(t=>{const v=zbE.get(t.epoch_index);return v==null?((t.verdict==='green'||t.verdict==='red')?null:zbar):Number(v);});
   {const fa=zbj&&(zbj.floor_at_cr||(zbj.next_step&&zbj.next_step.at_cr));gz.querySelector('.reading').title='z bar now '+zbar.toFixed(3)+(zbj&&zbj.formula?' · '+zbj.formula:'')+(fa?' · floor '+(zbj.floor??0.25)+' at '+String(fa).replace(' (UTC-6)',''):'');}
   barChart(gw,done.map(t=>({x:lab(t),v:t.z??0,c:t.verdict==='green'?'var(--performance)':t.verdict==='red'?'var(--rejected)':'var(--accepted)',hollow:!(t.verdict==='green'||t.verdict==='red'),
    tip:`<strong>E${t.epoch_index} · UID ${esc(t.uid)}</strong><span>z</span> ${fz(t.z)}  ${esc((t.verdict||'pending').toUpperCase())}\n<span>z bar</span> ${zbE.has(t.epoch_index)?Number(zbE.get(t.epoch_index)).toFixed(3):'—'}\n<span>head</span> ${pct(t.h_pass)} <span>cand</span> ${pct(t.c_pass)}\n<span>cap-hit</span> ${pct(t.cap)} <span>n</span> ${esc(t.n??'—')}`})),
    {label:'Gate z per turn',empty:'No K=1 turn decided yet.',refSteps:zsteps,refNow:zbar,refLabel:'z bar now '+zbar.toFixed(2),min:-3,max:3,slots:10,fmt:v=>(v>0?'+':'')+v,ticks:(lo,hi)=>{const s=Math.ceil(Math.max(-lo,hi)/3)||1;const o=[];for(let v=Math.ceil(lo/s)*s;v<=hi;v+=s)o.push(v);return o;}});
   // pass chart (pt904u 2026-10-31): one slot per epoch from the first to the last turn, head + candidate pass rate per decided turn.
   // Lines break at gaps (no turn, fail-closed, pending, held-out round still open) and where the measurement set changes (E272: held-out gate).
   pw.querySelector('svg')?.remove();
   {const W=pw.clientWidth||400,H=pw.clientHeight||220,P={l:40,r:10,t:10,b:22};const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});pw.prepend(svg);
    const byE=new Map(done.map(t=>[t.epoch_index,t]));const e0=done.length?done[0].epoch_index:0,e1=done.length?done[done.length-1].epoch_index:-1;
    const slots=[];for(let e=e0;e<=e1;e++)slots.push({e,t:byE.get(e)||null});
    const basisLab=b=>b==='heldout'?'held-out round, kept tasks':b==='gate-sample'?'gate sample (~1,000 tasks, turn batch removed)':'—';
    const pv=done.flatMap(t=>[t.h_pass,t.c_pass]).filter(v=>v!=null);const lo=pv.length?Math.max(0,Math.floor((Math.min(...pv)-.03)*20)/20):0,hi=pv.length?Math.min(1,Math.ceil((Math.max(...pv)+.02)*20)/20):1;
    const ys=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/((hi-lo)||1)),yc=v=>P.t+(H-P.t-P.b)*(1-v/0.4);
    const st=(hi-lo)>.3?.1:.05;for(let g=lo;g<=hi+1e-9;g+=st){svg.append(svgEl('line',{x1:P.l,x2:W-P.r,y1:ys(g),y2:ys(g),class:'k1-grid'}));const t=svgEl('text',{x:P.l-6,y:ys(g)+3,'text-anchor':'end'});t.textContent=Math.round(g*100)+'%';svg.append(t);}
    const n=Math.max(slots.length,10),bw=(W-P.l-P.r)/n,tp=tip(pw);
    // measurement-set change marker
    const fh=slots.findIndex(s=>s.t&&s.t.pass_basis==='heldout');
    if(fh>0){const x=P.l+fh*bw;svg.append(svgEl('line',{x1:x,x2:x,y1:P.t,y2:H-P.b,stroke:'var(--muted)','stroke-dasharray':'2 3',opacity:.6}));const t=svgEl('text',{x:x+3,y:P.t+9,fill:'var(--muted)','font-size':9});t.textContent='held-out set →';svg.append(t);}
    const ser=(k,c)=>{const segs=[];let cur=[],pb=null;slots.forEach((s,i)=>{const t=s.t,v=t?t[k]:null;
      if(v==null||(pb!==null&&t.pass_basis!==pb)){if(cur.length)segs.push(cur);cur=[];}
      if(v!=null){cur.push([P.l+i*bw+bw/2,ys(v)]);pb=t.pass_basis;}});if(cur.length)segs.push(cur);
     for(const pts of segs){if(pts.length>1)svg.append(svgEl('polyline',{points:pts.map(p=>p.join(',')).join(' '),fill:'none',stroke:c,'stroke-width':1.5}));for(const p of pts)svg.append(svgEl('circle',{cx:p[0],cy:p[1],r:3,fill:'var(--bg)',stroke:c,'stroke-width':1.5}));}};
    ser('h_pass','var(--guide)');ser('c_pass','var(--performance)');
    slots.forEach((s,i)=>{const t=s.t;
     const hit=svgEl('rect',{x:P.l+i*bw,y:P.t,width:bw,height:H-P.t-P.b,fill:'transparent'});hit.addEventListener('mouseenter',()=>{
      tp.innerHTML=!t?`<strong>E${s.e}</strong>no K=1 turn this epoch (no candidate, no gate run)`:
       `<strong>E${t.epoch_index} · UID ${esc(t.uid)} · ${esc((t.verdict||'pending').toUpperCase())}</strong><span>head</span> ${pct(t.h_pass)}\n<span>candidate</span> ${pct(t.c_pass)}\n<span>cap-hit</span> ${pct(t.cap)} (GREEN needs &lt; 20%)\n<span>set</span> ${esc(basisLab(t.pass_basis))}${t.n!=null?' · n '+esc(t.n):''}`+(t.pass_note&&t.h_pass==null?`\n<span>no pass rate</span> ${esc(t.pass_note)}`:'');
      tp.hidden=false;tp.style.left=Math.min(P.l+i*bw,W-220)+'px';tp.style.top='4px';});hit.addEventListener('mouseleave',()=>tp.hidden=true);svg.append(hit);});
    edgeLabels(svg,slots.length,i=>P.l+i*bw,bw,H-6,i=>'E'+slots[i].e);
    if(!done.length){const e=pw.querySelector('.empty')||pw.appendChild($('p',{class:'empty'}));e.textContent='No K=1 turn decided yet.';}}
   const smax=Math.max(1,...done.map(t=>t.skipped_not_ready||0));
   barChart(bwrap,done.map(t=>({x:lab(t),v:t.rollouts||0,c:'var(--incentive)',
    tip:`<strong>E${t.epoch_index} · UID ${esc(t.uid)}</strong><span>tasks</span> ${esc(t.tasks)} <span>rollouts</span> ${esc(t.rollouts)}\n<span>skipped ahead</span> ${esc(t.skipped_not_ready??'—')} <span>rule</span> ${esc(t.queue_rule||'')}`})),
    {label:'Batch per turn',empty:'No K=1 turn yet.',slots:10,max:72,fmt:v=>v,ticks:(lo,hi)=>{const s=hi>500?250:hi>100?50:24;const o=[];for(let v=0;v<=hi;v+=s)o.push(v);return o;}});
   // lr chart: one slot per epoch over the same range as the pass chart, log y
   lwrap.querySelector('svg')?.remove();
   {const W=lwrap.clientWidth||400,H=lwrap.clientHeight||220,P={l:44,r:10,t:10,b:22};const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`});lwrap.prepend(svg);
    const byE=new Map(done.map(t=>[t.epoch_index,t]));const e0=done.length?done[0].epoch_index:0,e1=done.length?done[done.length-1].epoch_index:-1;
    const slots=[];for(let e=e0;e<=e1;e++)slots.push({e,t:byE.get(e)||null,r:lrBy.get(e)||null});
    const vals=lrT.map(r=>r.lr).filter(v=>v!=null);let lo=Math.min(1e-7,...vals),hi=Math.max(1.2e-6,...vals);
    const L=Math.log10,ys=v=>P.t+(H-P.t-P.b)*(1-(L(v)-L(lo))/((L(hi)-L(lo))||1));
    for(const g of [1.25e-7,2.5e-7,5e-7,1e-6]){if(g<lo||g>hi)continue;svg.append(svgEl('line',{x1:P.l,x2:W-P.r,y1:ys(g),y2:ys(g),class:'k1-grid'}));const t=svgEl('text',{x:P.l-6,y:ys(g)+3,'text-anchor':'end'});t.textContent=fe(g);svg.append(t);}
    const n=Math.max(slots.length,10),bw=(W-P.l-P.r)/n,tp=tip(lwrap),rr=Math.max(2,Math.min(3,bw/2));
    slots.forEach((s,i)=>{const r=s.r;if(!r)return;const x=P.l+i*bw+bw/2,v=(s.t&&s.t.verdict)||r.verdict;
     const c=v==='green'?'var(--performance)':v==='red'?'var(--rejected)':'var(--accepted)';
     if(r.lr==null){svg.append(svgEl('circle',{cx:x,cy:H-P.b-4,r:rr,fill:'none',stroke:'var(--rejected)','stroke-dasharray':'1.5 1','stroke-width':1.2}));return;}
     svg.append(svgEl('circle',{cx:x,cy:ys(r.lr),r:rr,fill:v==='green'||v==='red'?c:'var(--bg)',stroke:c,'stroke-width':1.2}));});
    slots.forEach((s,i)=>{const t=s.t,r=s.r;
     const hit=svgEl('rect',{x:P.l+i*bw,y:P.t,width:bw,height:H-P.t-P.b,fill:'transparent'});hit.addEventListener('mouseenter',()=>{
      tp.innerHTML=!r?`<strong>E${s.e}</strong>${t?'no trainer lr published yet':'no K=1 turn this epoch'}`:
       `<strong>E${s.e}${t?' · UID '+esc(t.uid):''} · ${esc(((t&&t.verdict)||r.verdict||'pending').toUpperCase())}</strong><span>lr</span> ${r.lr==null?'private (miner-chosen, RED)':fe(r.lr)}\n<span>source</span> ${esc(r.source==='miner'?'miner-chosen':r.source==='default'?'default':'miner-chosen')}`;
      tp.hidden=false;tp.style.left=Math.min(P.l+i*bw,W-220)+'px';tp.style.top='4px';});hit.addEventListener('mouseleave',()=>tp.hidden=true);svg.append(hit);});
    edgeLabels(svg,slots.length,i=>P.l+i*bw,bw,H-6,i=>'E'+slots[i].e);
    if(!lrT.length){const e=lwrap.querySelector('.empty')||lwrap.appendChild($('p',{class:'empty'}));e.textContent='No learning rate published yet.';}}
  });
 }
 // 9. post-decision artifacts + gate reveals + replay (t764u transparency; data: /k1/turns/index.json, /gate/reveal/index.json, /k1/replay/results.json)
 async function arts(){
  const [ix,rv,rp]=await Promise.all([j('/k1/turns/index.json').catch(()=>null),j('/gate/reveal/index.json').catch(()=>null),j('/k1/replay/results.json').catch(()=>null)]);
  if(!ix)return;
  const rev={},rep={};for(const r of (rv&&rv.reveals)||[])rev[r.epoch]=r;for(const r of (rp&&rp.results)||[])rep[r.epoch]=r;
  const ok=v=>v===true?'<span class="k1-pill green">✓</span>':v===false?'<span class="k1-pill red">✗</span>':'—';
  const a=tile('panel-k1-artifacts','Turn data & gate reveals','k1-wide');a.querySelector('.reading').textContent=(ix.turns||[]).length;
  a.querySelector('.flip-table-wrap')?.remove();a.querySelector('.k1-rule')?.remove();
  a.insertAdjacentHTML('beforeend','<div class="flip-table-wrap"><table class="flip-table"><thead><tr><th class="num">Epoch</th><th class="num">UID</th><th>Verdict</th><th>Candidate</th><th class="num">Tasks</th><th>Turn data</th><th>Gate reveal</th><th>Commitment</th><th>Verdict recomputed</th><th class="num">Replay: elements off</th><th class="num">Replay: max diff</th></tr></thead><tbody>'+
   (ix.turns||[]).slice().reverse().map(t=>{const r=rev[t.epoch]||{},p=rep[t.epoch];return `<tr><td class="num">E${esc(t.epoch)}</td><td class="num">${esc(t.uid)}</td><td>${pill(t.decision)}</td><td class="k1-hash" title="${esc(t.candidate)}">${sh(t.candidate)}</td><td class="num">${esc(t.tasks)}</td><td><a href="${esc(t.turn)}">turn.json</a></td><td>${r.url?`<a href="${esc(r.url)}">E${esc(t.epoch)} reveal</a>`:'—'}</td><td>${ok(r.url?(r.commitment_matches&&r.indices_match_seed):null)}</td><td>${ok(r.url?r.decision_matches:null)}</td><td class="num">${p&&p.candidate_mismatched_elements!=null?esc(p.candidate_mismatched_elements.toLocaleString()):'—'}</td><td class="num">${p&&p.candidate_max_abs_diff!=null?Number(p.candidate_max_abs_diff).toExponential(1):'—'}</td></tr>`;}).join('')+
   '</tbody></table></div><p class="k1-rule">Published only after each gate decision: the trained batch (signed dispatch rows and the miner documents), the trainer report, the candidate file hashes (GREEN and RED), and the gate seed, sample and per-task grades. <a href="/k1/spec/step-recipe.json">Step recipe</a> · <a href="/k1/replay/README.md">Replay a step</a> · <a href="/gate/reveal/verify_gate_reveal.py">verify_gate_reveal.py</a> · <a href="/k1/dev/README.md">Dev split &amp; scorer</a></p>');
  a.querySelector('h2').title='Commitment = sha256(version|seed|ids) matches the hash published when the epoch opened and the ids are the draw the seed implies. Verdict recomputed = gate_core.decide on the revealed per-task grades gives the published verdict. Replay = our one-GPU replay of the step vs the published candidate (bf16 elements that differ, max abs difference).';
  const tt=document.getElementById('panel-k1-turns');if(tt&&tt.nextElementSibling!==a)tt.after(a);
 }
 async function j(u){const r=await fetch(u+(u.includes('?')?'&':'?')+'t='+Date.now(),{cache:'no-store'});if(!r.ok)throw new Error(u+' '+r.status);return r.json();}
 async function load(){
  try{const [k1,scores,mining,lrj,zbj]=await Promise.all([j('/k1.json'),j('/scores.json').catch(()=>({})),j('/mining.json').catch(()=>({})),j('/k1/lr.json').catch(()=>null),j('/gate/zbar.json').catch(()=>null)]);render(k1,scores,mining,lrj,zbj);document.getElementById('panel-k1-artifacts')?.remove();}
  catch(e){console.warn('k1',e);}
 }
 load();setInterval(load,60000);
 let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(load,250);});
})();
