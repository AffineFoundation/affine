/* Kings / Queue / History: three full-width submission tables with one column schema and one renderer
   (Jacob pt1100u 2026-11-02; replaces the Next in line, Payees, Head lineage and Turns tiles; layout after teutonic.ai).
   Data: /k1.json, /scores.json, /mining.json (existing) + /k1/tables.json (build_k1_tables.py: gain, decided_at, GREEN-only
   train params and file links, coldkeys, queue file checks, epoch timing). RED and pending train params stay private. */
(()=>{
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const clip=(s,a=6,b=4)=>{s=String(s||'');return s.length>a+b+1?s.slice(0,a)+'…'+s.slice(-b):(s||'—');};
 const sh=h=>h?String(h).slice(0,8):'—';
 const pct=x=>x==null?'—':(100*x).toFixed(1)+'%';
 const sgn=(x,d=2)=>x==null||!isFinite(x)?'—':(x>0?'+':x<0?'−':'')+Math.abs(x).toFixed(d);
 const fnum=v=>{if(typeof v!=='number'||!isFinite(v))return esc(v);if(v!==0&&(Math.abs(v)<1e-3||Math.abs(v)>=1e5)){const e=Math.floor(Math.log10(Math.abs(v))),m=v/Math.pow(10,e);return(Math.abs(m-Math.round(m))<1e-9?Math.round(m):+m.toFixed(2))+'e'+e;}return String(+v.toPrecision(4));};
 const dur=s=>{if(s==null||!isFinite(s))return'—';s=Math.max(0,s);const m=Math.round(s/60);if(m<1)return'<1m';if(m<60)return m+'m';const h=Math.floor(m/60);if(h<48)return h+'h'+(m%60?' '+(m%60)+'m':'');const d=Math.floor(h/24);return d+'d'+(h%24?' '+(h%24)+'h':'');};
 const ago=t=>t?dur(Date.now()/1000-t)+' ago':'—';
 const crf=new Intl.DateTimeFormat('en-CA',{timeZone:'America/Costa_Rica',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
 const cr=t=>t?crf.format(new Date(t*1000)).replace(',','')+' CR (UTC-6)':'';
 const crParse=s=>{const m=/^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)/.exec(s||'');return m?Date.parse(m[1]+'T'+m[2]+'-06:00')/1000:null;};
 const pill=v=>{const c=v==='green'?'green':v==='red'?'red':v==='skip'?'skip':'pending';return '<span class="k1-pill '+c+'">'+esc(v==='green'?'GREEN':v==='red'?'RED':v||'pending')+'</span>';};
 const key=(v,a,b)=>v?`<button type="button" class="k1t-key" data-copy="${esc(v)}" title="${esc(v)} (click to copy)">${esc(clip(v,a,b))}</button>`:'<span class="k1t-mute">—</span>';
 const priv=(t,red)=>`<span class="k1t-private${red?' red':''}" title="${esc(t)}">private</span>`;
 const mute=(s,t)=>`<span class="k1t-mute"${t?` title="${esc(t)}"`:''}>${s}</span>`;

 /* hparams: only the knobs that differ from the validator default, compact; the full set in the tooltip. Generic over keys,
    so new optional train knobs show up as soon as mining.json batch_file.train.fields lists them. */
 function hparams(r,D){
  const h=r.hp||{};
  if(h.state==='private')return priv(h.why,h.red);
  if(h.state==='defaults')return mute('defaults',h.why||'No miner train params: validator defaults'+(Object.keys(D).length?' ('+Object.entries(D).map(([k,v])=>k+' '+fnum(v)).join(' · ')+')':''));
  if(h.state!=='public'||!h.train)return mute('—');
  const all={...D,...h.train},diff=Object.keys(all).filter(k=>h.train[k]!==undefined&&(D[k]===undefined||Math.abs(h.train[k]-D[k])>1e-12*Math.max(1,Math.abs(D[k]))));
  const full=Object.keys(all).sort().map(k=>k+' '+fnum(all[k])+(D[k]!==undefined&&!diff.includes(k)?' (default)':D[k]!==undefined?' (default '+fnum(D[k])+')':'')).join('\n');
  const txt=diff.length?diff.sort().map(k=>`<span class="k1t-hk">${esc(k)}</span> ${fnum(h.train[k])}`).join(' <span class="k1t-sep">·</span> '):'<span class="k1t-mute">defaults</span>';
  return `<span class="k1t-hp" title="${esc(full)}">${txt}</span>`;
 }
 const batch=r=>r.tasks?`<span title="${esc(r.tasks)} tasks · ${esc(r.rollouts??'—')} rollouts">${esc(r.tasks)} × ${r.rollouts?+(r.rollouts/r.tasks).toFixed(1):'—'}</span>`:mute('—');
 const file=r=>r.file?`<a href="${esc(r.file)}" title="The batch file this turn trained on${r.file_bytes?' ('+(r.file_bytes/1e6).toFixed(1)+' MB)':''}">batch.json</a>`:r.fileState==='private'?priv(r.fileWhy||'RED and pending files stay private',r.verdict==='red'&&r.status==='turn'):mute('—');

 /* one schema: shared columns + per-table columns */
 const C={
  rank:{h:'#',w:36,c:'num',r:r=>mute(r.rank)},
  line:{h:'#',w:36,c:'num',r:r=>mute(r.line),t:'Place among the members with a valid file now (0 = next turn).'},
  epoch:{h:'Epoch',w:64,c:'num',r:r=>r.turnUrl?`<a href="${esc(r.turnUrl)}" title="turn.json">E${esc(r.epoch)}</a>`:'E'+esc(r.epoch)},
  uid:{h:'UID',w:52,c:'num',r:r=>esc(r.uid??'—')},
  hot:{h:'Hotkey',w:104,r:r=>key(r.hotkey)+(r.dereg?' '+mute('dereg','No longer registered'):'')},
  cold:{h:'Coldkey',w:104,r:r=>key(r.coldkey),t:'On-chain owner of the hotkey.'},
  batch:{h:'Batch',w:84,c:'num',r:batch,t:'Tasks × mean rollouts per task (totals in the tooltip).'},
  hp:{h:'Hparams',w:150,r:(r,D)=>hparams(r,D),t:'Train params that differ from the validator defaults (all of them in the tooltip). Public for GREEN turns only; RED, pending and queued files stay private.'},
  file:{h:'File',w:86,r:file,t:'The trained batch file, published for GREEN turns only.'},
  greenAt:{h:'GREEN at',w:132,c:'num',r:r=>`<span title="${esc(cr(r.decided_at))}">E${esc(r.green_epoch)} <span class="k1t-mute">${ago(r.decided_at)}</span></span>`},
  z:{h:'z',w:60,c:'num',r:r=>r.z==null?mute('—'):`<span title="z = gain / SE = ${esc(r.z.toFixed(4))}">${sgn(r.z)}</span>`},
  gain:{h:'Gain',w:76,c:'num',r:r=>r.gain==null?mute('—'):`<span class="${r.gain>0?'k1t-up':'k1t-dn'}" title="Held-out pass-rate gain vs the head: ${esc(r.gain.toFixed(5))}${r.se!=null?' ± SE '+esc(r.se.toFixed(5)):''}">${sgn(100*r.gain)} pp</span>`,t:'Candidate minus head pass rate on the held-out round, in percentage points.'},
  weight:{h:'Weight',w:104,r:r=>`<span class="k1-bar" style="width:${Math.round(44*(r.share||0))}px"></span>${pct(r.share)}`,t:'Weight on chain now (scores.json): 1/N to each of the last up to 10 distinct GREEN UIDs.'},
  head:{h:'Head hash',w:92,r:r=>`<span class="k1-hash" title="${esc(r.candidate)}">${sh(r.candidate)}</span>${r.isHead?' <span class="k1-pill green k1t-tag">head</span>':''}`,t:'The checkpoint this GREEN turn produced.'},
  reg:{h:'Reg block',w:88,c:'num',r:r=>`<span title="${r.regAgo?'registered ~'+esc(r.regAgo)+' ago':''}">${esc(r.registration_block??'—')}</span>`},
  order:{h:'Order',w:60,c:'num',r:r=>esc(r.position??'—'),t:'Position among all unserved hotkeys/registrations (oldest registration block first).'},
  eta:{h:'Est. eval',w:84,c:'num',r:r=>r.eta==null?mute('—'):`<span title="${esc(r.etaWhy)}">~${dur(r.eta)}</span>`,t:'Estimated time until this turn is evaluated: rest of the current epoch plus one median epoch per place ahead. A forecast; a file that turns invalid or a skip changes it.'},
  sub:{h:'Submitted',w:92,c:'num',r:r=>r.checked_at?`<span title="${esc('file checked '+cr(r.checked_at)+(r.format?' · '+r.format:''))}">${ago(r.checked_at)}</span>`:mute('—'),t:'Time since the current batch file was uploaded (first checked).'},
  result:{h:'Result',w:84,r:r=>pill(r.status==='turn'?r.verdict:'skip')},
  margin:{h:'vs bar',w:96,c:'num',r:r=>{if(r.z==null)return mute('—');const m=r.z-1,w=Math.round(Math.min(1,Math.abs(m)/3)*22);return `<span class="k1t-m ${m>0?'k1t-up':'k1t-dn'}" title="z ${esc(r.z.toFixed(3))} vs the bar z > 1.00: ${m>0?'cleared by':'short by'} ${esc(Math.abs(m).toFixed(3))}"><i style="width:${w}px"></i>${sgn(m)}</span>`;},t:'How close the turn came to the GREEN bar: z − 1 (positive = cleared). GREEN also needs gain > 0 and cap-hit < 33%.'},
  cap:{h:'Cap-hit',w:68,c:'num',r:r=>r.cap==null?mute('—'):`<span class="${r.cap>=0.33?'k1t-dn':''}" title="Share of the candidate's held-out answers that hit the token cap (RED at ≥ 33%)">${pct(r.cap)}</span>`},
  evald:{h:'Evaluated',w:92,c:'num',r:r=>r.decided_at?`<span title="${esc(cr(r.decided_at))}">${ago(r.decided_at)}</span>`:mute(r.status==='turn'?'pending':'—')},
 };
 const SHARED=['uid','hot','cold','batch','hp','file'];
 const T={
  kings:{cols:['rank',...SHARED,'greenAt','z','gain','weight','head'],empty:'No GREEN turn yet.'},
  queue:{cols:['line',...SHARED,'reg','order','eta','sub'],empty:'No unserved registration has a valid batch file now.'},
  history:{cols:['epoch',...SHARED,'result','gain','z','margin','cap','evald'],empty:'No turn yet.'},
 };
 const head=n=>'<colgroup>'+T[n].cols.map(k=>`<col style="width:${C[k].w}px">`).join('')+'</colgroup><thead><tr>'+T[n].cols.map(k=>`<th class="${C[k].c||''}"${C[k].t?` title="${esc(C[k].t)}"`:''}>${esc(C[k].h)}</th>`).join('')+'</tr></thead>';
 window.K1T_HEAD=head;/* used once to write the static shells in index.html */
 function fill(n,rows,D,reading,title,more){
  const s=document.getElementById('k1-'+n);if(!s)return;
  const box=s.querySelector('.k1t-scroll'),tb=box&&box.querySelector('tbody');if(!tb)return;
  const tbl=box.querySelector('table');const sig=T[n].cols.join(',');if(tbl.dataset.cols!==sig){tbl.querySelector('colgroup')?.remove();tbl.querySelector('thead')?.remove();tbl.insertAdjacentHTML('afterbegin',head(n));tbl.dataset.cols=sig;}
  tbl.style.minWidth=T[n].cols.reduce((a,k)=>a+C[k].w,0)+'px';
  tb.innerHTML=(rows.length?rows.map(r=>'<tr>'+T[n].cols.map(k=>`<td class="${C[k].c||''}">${C[k].r(r,D)}</td>`).join('')+'</tr>').join(''):`<tr><td colspan="${T[n].cols.length}" class="flip-loading">${esc(T[n].empty)}</td></tr>`)+(more?`<tr class="k1t-more"><td colspan="${T[n].cols.length}">${more}</td></tr>`:'');
  s.querySelector('.reading').textContent=reading;if(title)s.querySelector('h2').title=title;
  if(box.classList.contains('afx-wait')){box.classList.remove('afx','afx-wait');if(!matchMedia('(prefers-reduced-motion:reduce)').matches){tb.classList.add('afx-in');tb.addEventListener('animationend',()=>tb.classList.remove('afx-in'),{once:true});}}
 }

 function render(k1,scores,mining,X){
  X=X||{};const D=X.train_defaults||((mining.batch_file||{}).train||{}).fields&&Object.fromEntries(Object.entries(mining.batch_file.train.fields).map(([k,v])=>[k,v.default]))||{};
  const CK=X.coldkeys||{},PT=X.turns||{},QF=X.queue_files||{},now=Date.now()/1000;
  const turns=(k1.turns||[]).slice().sort((a,b)=>b.epoch_index-a.epoch_index);
  const byE=new Map(turns.map(t=>[t.epoch_index,t]));
  const headCk=((mining.training||{}).head||{}).checkpoint;
  const turnRow=t=>{const p=PT[t.epoch_index]||{},v=t.verdict,g=v==='green';
   const r={...t,epoch:t.epoch_index,coldkey:CK[t.hotkey],gain:p.gain,se:p.se,decided_at:p.decided_at||crParse(t.decided_at_cr),
    turnUrl:(v==='green'||v==='red')?'/k1/turns/'+t.epoch_index+'/turn.json':null,file:g?p.file:null,file_bytes:p.file_bytes,
    fileState:g?(p.file?'public':'none'):'private',fileWhy:v==='red'?'RED: the miner file stays private':'Pending: published only if the turn is GREEN'};
   r.hp=g?(p.train?{state:'public',train:p.train}:{state:'defaults'}):{state:'private',red:v==='red',why:v==='red'?'RED: train params stay private':'Pending: train params are published only if the turn is GREEN'};
   if(t.status!=='turn'){r.hp={};r.fileState='none';}
   return r;};
  // Kings: the paid GREEN UIDs, newest GREEN first
  const share={};for(const m of (scores.miners||[]))share[m.uid]=m.share_now;
  const kings=(k1.payees||[]).slice().sort((a,b)=>b.green_epoch-a.green_epoch).map((p,i)=>{const t=byE.get(p.green_epoch);const r=t?turnRow(t):{hp:{}};
   return {...r,rank:i+1,uid:p.uid,hotkey:p.hotkey,coldkey:CK[p.hotkey],green_epoch:p.green_epoch,z:p.z??r.z,share:share[p.uid]||0,dereg:p.still_registered===false,isHead:!!(r.candidate&&r.candidate===headCk)};});
  fill('kings',kings,D,kings.length,'The last up to 10 distinct GREEN UIDs, newest first. Each gets 1/N of the weight (scores.json, set hourly). Their train params and batch files are public.');
  // Queue
  const q=k1.queue||{},E=X.epoch||{},med=E.median_s,blk=X.chain_block||q.chain_block;
  const rest=med&&E.opened_at?Math.max(0,E.opened_at+med-now):med?med/2:null;
  const queue=(q.next||[]).map((r,i)=>{const f=QF[r.hotkey]||{},line=r.line??i;const eta=rest==null?null:rest+(line+1)*med;
   return {...r,line,coldkey:CK[r.hotkey],tasks:f.groups,rollouts:f.rollouts,checked_at:f.checked_at,format:f.format,
    hp:f.format==='affine-batch-v1'?{state:'defaults',why:'affine-batch-v1 file: validator defaults'}:{state:'private',why:'Train params in a queued file stay private; published only if its turn is GREEN'},
    fileState:'private',fileWhy:'Queued files stay private; published only if the turn is GREEN',
    regAgo:blk&&r.registration_block?dur((blk-r.registration_block)*12):null,
    eta,etaWhy:med?`turn starts in ~${dur(eta-med)}, verdict ~${dur(eta)} (median epoch ${dur(med)} over the last ${E.sample||'?'} epochs; ${line} ahead with a valid file)`:''};});
  const nReady=q.queue_length??queue.length,nUn=q.registrations_unserved,rest2=nReady-queue.length;
  const more=(rest2>0?`+${rest2} more with a valid file now (only the next ${queue.length} are published)`:'')+(nUn!=null&&nUn>nReady?(rest2>0?' · ':'')+`${nUn-nReady} unserved registrations without a valid file are skipped and keep their place`:'');
  fill('queue',queue,D,`${queue.length} of ${nReady} ready`,(q.queue_rule==='hotkey-registration-once'?'Unserved entries (one turn per hotkey and per UID + registration block, permanently)':'Unserved hotkeys (one turn per hotkey, permanently)')+' with a valid batch file now, in the gate\'s pick order: open grants first, then registration block, then UID; at each seal the first member with a valid file takes the turn. '+nReady+' have a valid file now; the next '+queue.length+' are listed. Order = position among all '+(nUn??'')+' unserved entries. A forecast: a file uploaded or invalidated before the seal changes it.',more);
  // History: every turn, newest first
  const hist=turns.map(turnRow);
  const nd=turns.filter(t=>t.status==='turn'),ng=nd.filter(t=>t.verdict==='green').length;
  fill('history',hist,D,nd.length,`Every K=1 turn, newest first (${ng} GREEN · ${nd.filter(t=>t.verdict==='red').length} RED). GREEN = gain > 0, z > 1 and cap-hit < 33% on the held-out round. RED train params and files stay private.`);
 }
 // click a key to copy its full value
 document.addEventListener('click',e=>{const b=e.target.closest&&e.target.closest('.k1t-key');if(!b)return;const v=b.dataset.copy;
  const done=()=>{b.classList.add('copied');setTimeout(()=>b.classList.remove('copied'),900);};
  if(navigator.clipboard&&window.isSecureContext)navigator.clipboard.writeText(v).then(done,()=>{});});
 async function j(u){const r=await (window.AFX&&AFX.take(u)||fetch(u+(u.includes('?')?'&':'?')+'t='+Date.now(),{cache:'no-store'}));if(!r.ok)throw new Error(u+' '+r.status);return r.json();}
 async function load(){
  try{const [k1,scores,mining,X]=await Promise.all([j('/k1.json'),j('/scores.json').catch(()=>({})),j('/mining.json').catch(()=>({})),j('/k1/tables.json').catch(()=>null)]);render(k1,scores,mining,X);}
  catch(e){console.warn('k1-tables',e);}
 }
 if(document.getElementById('k1-history')){load();setInterval(load,60000);}
})();
