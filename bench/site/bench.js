/* Benchmarks (Jacob 2026-11-01; bench-v2 2026-11-05; chart fix + reported rows 2026-11-05):
   Sparklines: x = king epoch ascending (oldest left); bench-v1 and bench-v2 are separate series
   (muted grey dots for v1, accent line+dots for v2); never one line across protocols.
   Headline = latest complete bench-v2 king. Incomplete runs not plotted.
   Reported rows (kind=reported): developer-published scores, badge + footnote, not in Avg/frontier. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
 const pct=x=>x==null?'—':(100*x).toFixed(1);
 const PARAMS={'genesis':9.41e9,'qwen3.5-9b':9.65e9,'qwen3-8b':8.19e9,'deepseek-r1-0528-qwen3-8b':8.19e9,'r1-distill-qwen-7b':7.62e9,
  'r1-distill-llama-8b':8.03e9,'nemotron-nano-9b-v2':8.89e9,'glm-z1-9b':9.40e9,'qwen3.5-4b':4.66e9,'qwen3-14b':14.77e9,'qwen3-4b':4.02e9,
  'r1-distill-qwen-14b':14.77e9,'openreasoning-nemotron-7b':7.62e9,'acereason-nemotron-1.1-7b':7.62e9,'skywork-or1-7b':7.62e9,'mimo-7b-rl':7.83e9,
  'llama-nemotron-nano-8b-v1':8.03e9,'openthinker3-7b':7.62e9,'phi-4-reasoning':14.66e9,'phi-4-mini-reasoning':3.84e9,'deepscaler-1.5b':1.78e9,
  'r1-distill-qwen-1.5b':1.78e9,'gpt-oss-20b':20.91e9,'intellect-3':106e9,'glm-4.5-air':null};
 const SHORT={'deepseek-r1-0528-qwen3-8b':'R1-0528-Qwen3-8B','r1-distill-qwen-7b':'R1-Qwen-7B','r1-distill-llama-8b':'R1-Llama-8B',
  'r1-distill-qwen-14b':'R1-Qwen-14B','r1-distill-qwen-1.5b':'R1-Qwen-1.5B','nemotron-nano-9b-v2':'Nemotron-Nano-9B','glm-z1-9b':'GLM-Z1-9B',
  'openreasoning-nemotron-7b':'OR-Nemotron-7B','acereason-nemotron-1.1-7b':'AceReason-7B','llama-nemotron-nano-8b-v1':'Nemotron-Nano-8B',
  'phi-4-reasoning':'Phi-4-R-14B','phi-4-mini-reasoning':'Phi-4-mini-R','deepscaler-1.5b':'DeepScaleR-1.5B',
  'intellect-3':'INTELLECT-3','glm-4.5-air':'GLM-4.5-Air'};
 const gb=p=>p==null?'':(p/1e9).toFixed(1)+'B';
 const kname=m=>'Affine-'+String(m.king).slice(0,8)+'-9B';
 const name=m=>{const n=esc(SHORT[m.model_id]||m.label);const repo=m.hf&&String(m.hf).split('@')[0];
  const t=esc((repo||m.label)+(PARAMS[m.model_id]?' · '+gb(PARAMS[m.model_id])+' params':''));
  return (repo?`<a href="https://huggingface.co/${esc(repo)}" target="_blank" rel="noopener" title="${t}">${n}</a>`:n)+(PARAMS[m.model_id]?` <span>${gb(PARAMS[m.model_id])}</span>`:'');};
 const proto=r=>{const ev=String((r&&r.env_version)||''); if(ev.includes('bench-v2'))return 'v2'; if(ev.includes('bench-v1'))return 'v1';
  if(r&&r.model_kind==='reported')return 'reported'; if((r&&r.budget)!=null&&r.budget>=65536)return 'v2'; return 'v1';};
 const prefer=(a,b)=>{ // higher protocol wins; complete preferred
  if(!a)return b; if(!b)return a;
  const pa=proto(a),pb=proto(b);
  if(pa!==pb) return (pb==='v2'?b:a);
  if(a.complete===false&&b.complete!==false)return b;
  if(b.complete===false&&a.complete!==false)return a;
  return (b.run_at||0)>=(a.run_at||0)?b:a;
 };
 let box=document.getElementById('bench-table');
 if(!box){box=document.createElement('section');box.id='bench-table';box.setAttribute('aria-label','Benchmarks');main.parentNode.insertBefore(box,main);}
 function table(d){
  const B=(d.benchmarks||[]).filter(b=>b.enabled).sort((a,b)=>(a.suite===b.suite?0:d.suites.indexOf(a.suite)-d.suites.indexOf(b.suite))||a.order-b.order);
  // All protocol rows (results + results_prior); R = preferred complete for table cells
  const allRes=[...(d.results||[]),...(d.results_prior||[])];
  const byProto={}; // mid -> bid -> {v1,v2,reported}
  for(const r of allRes){
    if(r.complete===false) continue; // never plot/show incomplete
    const p=proto(r); const mid=r.model_id, bid=r.benchmark_id;
    (byProto[mid]||(byProto[mid]={}))[bid]||(byProto[mid][bid]={});
    byProto[mid][bid][p]=r;
  }
  const R={};
  for(const mid of Object.keys(byProto)){
    R[mid]={};
    for(const bid of Object.keys(byProto[mid])){
      const bag=byProto[mid][bid];
      R[mid][bid]=prefer(prefer(bag.v1, bag.v2), bag.reported);
    }
  }
  const ms=d.models||[],gen=ms.find(m=>m.kind==='genesis');
  // Table: kings newest-first by epoch
  const kings=ms.filter(m=>m.kind==='king'&&(R[m.model_id]||m.status==='evaluating'))
    .sort((a,b)=>(b.epoch??0)-(a.epoch??0));
  const reported=ms.filter(m=>m.kind==='reported').sort((a,b)=>(a.priority??999)-(b.priority??999));
  const AB=B.filter(b=>b.in_avg!==false);
  const have=m=>AB.filter(b=>((R[m.model_id]||{})[b.id]||{}).score!=null);
  const full=m=>R[m.model_id]&&have(m).length>0&&PARAMS[m.model_id]!=null;
  const avg=m=>{
    if(m.kind==='reported') return null; // Avg n/a for reported
    const h=have(m);return h.length?h.reduce((t,b)=>t+R[m.model_id][b.id].score,0)/h.length:null;
  };
  // Frontier excludes reported
  const cand=ms.filter(m=>(m.kind==='reference'||m.kind==='genesis')&&full(m));
  const bsc=(m,id)=>id==='_avg'?avg(m):(((R[m.model_id]||{})[id]||{}).score??null);
  const front=(m,id)=>{const x=bsc(m,id);return x!=null&&!cand.some(o=>{const y=bsc(o,id);return o!==m&&y!=null&&PARAMS[o.model_id]<=PARAMS[m.model_id]&&y>x;});};
  const refs=cand.filter(m=>m.kind==='reference'&&['_avg',...AB.map(b=>b.id)].some(id=>front(m,id)))
   .sort((a,b)=>PARAMS[b.model_id]-PARAMS[a.model_id]||avg(b)-avg(a));
  const all=[...kings,...reported,...refs,...(gen?[gen]:[])];
  const AVG={id:'_avg'};const val=(m,b)=>b===AVG?(R[m.model_id]?avg(m):null):((R[m.model_id]||{})[b.id]||{}).score??null;
  const r1=x=>Math.round(1000*x);
  const span={};for(const b of [...B,AVG]){const v=all.map(m=>val(m,b)).filter(x=>x!=null);if(v.length)span[b.id]=[Math.min(...v),Math.max(...v)];}
  const STAR='<svg class="max-ic" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 .6l1.3 2.8 3 .3-2.3 2 .7 3L5 7.2 2.3 8.7l.7-3-2.3-2 3-.3z"/></svg>';
  const shade=f=>{const d=1-f,k=Math.min(1,Math.max(0,(d-.1)/.75)),h=Math.round(130*(1-Math.pow(k,.75)));return`hsl(${h} 65% 45% / ${(.30+.14*Math.abs(.5-k)*2).toFixed(2)})`;};
  const cell=(m,b)=>{
    const x=val(m,b),c=b===AVG?' avg':'';
    if(x==null)return`<td class="na${c}">${b===AVG&&m.kind==='reported'?'n/a':(b===AVG&&!R[m.model_id]&&(m.status==='evaluating'||m.status==='queued')?'…':'—')}</td>`;
    const rr=b===AVG?null:(R[m.model_id]||{})[b.id],inc=rr&&rr.complete===false,pp=inc&&rr.errors>.01*rr.samples_expected;
    const pt=inc?`partial run: ${rr.samples} of ${rr.samples_expected} samples (${rr.errors} errored)`:'',dg=pp?'<sup title="'+pt+'">†</sup>':'';
    const cap=rr&&rr.cap_rate!=null?`cut-off ${(100*rr.cap_rate).toFixed(1)}%`:'';
    const bud=rr&&rr.budget!=null?`budget ${Number(rr.budget).toLocaleString()}`:(m.budget!=null?`budget ${Number(m.budget).toLocaleString()}`:'');
    const rep=m.kind==='reported'?`reported, not run on Affine${m.source?'; '+m.source:''}`:'';
    if(m.kind==='reported'){const t=[rep,m.source_note,bud].filter(Boolean).join('; ');return`<td class="reported"${t?` title="${esc(t)}"`:''}>${pct(x)}</td>`;}
    if(b.in_avg===false){const t=['display only, not in Avg',bud,cap,pt].filter(Boolean).join('; ');return`<td class="disp" title="${esc(t)}">${pct(x)}${dg}</td>`;}
    const [lo,hi]=span[b.id];const f=hi>lo?(x-lo)/(hi-lo):1;const top=r1(x)===r1(hi);const tt=[top?'column best':'',bud,cap,pt].filter(Boolean).join('; ');
    return`<td class="${top?'best':''}${c}" style="background:linear-gradient(${shade(f)},${shade(f)}),var(--bg)"${tt?` title="${esc(tt)}"`:''}>${top?STAR:''}${pct(x)}${dg}</td>`;
  };
  const rowMeta=m=>{
    if(m.kind==='reported') return (m.protocol||'reported')+' · not run on Affine';
    const rs=Object.values(R[m.model_id]||{});const caps=rs.map(r=>r.cap_rate).filter(x=>x!=null);
    const budget=m.budget??(rs.find(r=>r.budget!=null)||{}).budget;const cap=caps.length?caps.reduce((a,b)=>a+b,0)/caps.length:null;
    const bits=[];if(budget!=null)bits.push('budget '+Number(budget).toLocaleString()+' tok');
    if(cap!=null)bits.push('cut-off '+(100*cap).toFixed(1)+'%');if(m.profile)bits.push(m.profile);
    return bits.join(' · ');
  };
  const row=(m,cls='')=>{
    const meta=rowMeta(m);
    let label;
    if(m.kind==='king') label=`E${esc(m.epoch)} <span>${esc(kname(m))}</span>`;
    else if(m.kind==='reported'){
      const tip=esc([m.source_note,m.source].filter(Boolean).join(' '));
      label=`${name(m)} <span class="bench-reported-badge" title="${tip}">reported, not run</span>`;
      cls=(cls+' reported').trim();
    } else label=name(m);
    return`<tr class="${cls}"><th scope="row" title="${esc(meta)}">${label}${meta?` <span class="bench-meta">${esc(meta)}</span>`:''}</th>${cell(m,AVG)}${B.map(b=>cell(m,b)).join('')}</tr>`;
  };
  const suites=[...new Set(B.map(b=>b.suite))];
  const head=suites.length>1?`<tr><th></th><th class="avg"></th>${suites.map(s=>`<th colspan="${B.filter(b=>b.suite===s).length}" class="suite">${esc(s)}</th>`).join('')}</tr>`:'';
  const prev=box.querySelector('.bench-scroll'),keep=prev?[prev.scrollTop,prev.scrollLeft]:null;
  const cols=[AVG,...B],lab=b=>b===AVG?'Avg':b.label;

  // Sparks: x = epoch ascending; separate v1 / v2 series (complete only)
  const kingById=Object.fromEntries(kings.map(m=>[m.model_id,m]));
  const scoredKings=kings.filter(m=>m.epoch!=null).sort((a,b)=>a.epoch-b.epoch);
  function seriesFor(col, wantProto){
    const pts=[];
    for(const m of scoredKings){
      let v=null, se=null, r=null;
      if(col===AVG){
        // avg over in-avg benches of that protocol only
        const scores=[];
        for(const b of AB){
          const bag=(byProto[m.model_id]||{})[b.id]||{};
          const rr=bag[wantProto];
          if(rr&&rr.score!=null&&rr.complete!==false) scores.push(rr.score);
        }
        if(!scores.length) continue;
        v=scores.reduce((a,b)=>a+b,0)/scores.length;
      } else {
        const bag=(byProto[m.model_id]||{})[col.id]||{};
        r=bag[wantProto];
        if(!r||r.score==null||r.complete===false) continue;
        v=r.score; se=r.se??null;
      }
      pts.push({t:m.epoch, v, se, m, proto:wantProto});
    }
    return pts;
  }
  SP=cols.map(b=>{
    const v1=seriesFor(b,'v1'), v2=seriesFor(b,'v2');
    // headline = latest complete v2 point, else latest v1
    const headPts=v2.length?v2:v1;
    const latest=headPts.length?headPts[headPts.length-1]:null;
    return {b, lab:lab(b), v1, v2, latest};
  });
  const sparks=SP.map((s,i)=>{
    const l=s.latest;
    const n=s.v1.length+s.v2.length;
    return`<div class="bench-spark" data-i="${i}"><div class="section-heading"><h2 title="${esc(s.lab)}: king scores by epoch; grey line = older stricter bench (v1), accent = bench-v2 (64K). Headline=latest complete v2.">${esc(s.lab)}</h2><span class="reading" title="${l?`E${l.m.epoch} ${l.proto}`:''}">${l?pct(l.v)+'%':'—'}</span></div><div class="chart-wrap k1-chart bs-plot" role="img" aria-label="${esc(s.lab)} king score history (${n} points)"></div></div>`;
  }).join('');

  const footnote=reported.length?`<p class="bench-footnote">Rows marked <em>reported, not run</em> show developer-published scores (not Affine-graded). ${reported.map(m=>`<a href="${esc(m.source||('#') )}" target="_blank" rel="noopener">${esc(m.label)}</a>`).join(' · ')}.</p>`:'';

  box.innerHTML=`<div class="bench-sparks" aria-label="King score history per benchmark">${sparks}</div><div class="bench-scroll"><table><thead>${head}<tr><th scope="col">Model</th><th scope="col" class="avg" title="mean of the row's scored benchmark columns (empty cells left out; n/a for reported)">Avg</th>${B.map(b=>`<th scope="col"${b.in_avg===false?' class="disp" title="display only: not counted in Avg yet"':''}>${esc(b.label)}</th>`).join('')}</tr></thead><tbody>`+
   kings.map(m=>row(m)).join('')+`</tbody><tfoot>`+reported.map(m=>row(m)).join('')+refs.map(m=>row(m)).join('')+(gen?row(gen,'genesis'):'')+`</tfoot></table></div>${footnote}`;
  layout();
  const sc=box.querySelector('.bench-scroll');if(keep){sc.scrollTop=keep[0];sc.scrollLeft=keep[1];}
 }
 let SP=[];
 const NS='http://www.w3.org/2000/svg',el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
 function spark(card,s){
  const plot=card.querySelector('.bs-plot');plot.textContent='';
  const v1=s.v1||[], v2=s.v2||[]; // full history kept (Jacob 2026-11-05): v1 as a thin muted line, v2 accent
  const all=[...v1,...v2].sort((a,b)=>a.t-b.t);
  const W=plot.clientWidth||200,H=plot.clientHeight||90,P={l:30,r:6,t:6,b:16};
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});plot.append(svg);
  if(!all.length){const e=document.createElement('p');e.className='empty';e.textContent='No king scored yet.';plot.append(e);return;}
  const t0=all[0].t, t1=all[all.length-1].t;
  let lo=Math.min(...all.map(p=>p.v)), hi=Math.max(...all.map(p=>p.v));
  const st=hi-lo>.2?.1:hi-lo>.08?.05:hi-lo>.03?.02:.01;lo=Math.max(0,Math.floor(lo/st)*st);hi=Math.min(1,Math.ceil(hi/st)*st);if(hi<=lo)hi=Math.min(1,lo+st);
  const x=t=>t1>t0?P.l+(W-P.l-P.r)*((t-t0)/(t1-t0)):(P.l+W-P.r)/2;
  const y=v=>P.t+(H-P.t-P.b)*(1-(v-lo)/(hi-lo));
  for(const g of [lo,(lo+hi)/2,hi]){svg.append(el('line',{x1:P.l,x2:W-P.r,y1:y(g),y2:y(g),class:'k1-grid'}));
   const tx=el('text',{x:P.l-5,y:y(g)+3,'text-anchor':'end'});{const v=100*g;tx.textContent=(Math.abs(v-Math.round(v))<1e-6?v.toFixed(0):v.toFixed(1))+'%';}svg.append(tx);}
  // v1: muted grey dots only (no connecting line across eras)
  const C1='var(--text-muted, #8a8a8a)', C2='var(--performance)';
  if(v1.length>1)svg.append(el('polyline',{points:v1.map(p=>x(p.t).toFixed(1)+','+y(p.v).toFixed(1)).join(' '),fill:'none',stroke:C1,'stroke-width':1,'stroke-linejoin':'round',opacity:'0.7'}));
  // v2: accent line + hollow/filled points
  if(v2.length>1)svg.append(el('polyline',{points:v2.map(p=>x(p.t).toFixed(1)+','+y(p.v).toFixed(1)).join(' '),fill:'none',stroke:C2,'stroke-width':1.5,'stroke-linejoin':'round'}));
  v2.forEach((p,i)=>svg.append(el('circle',{cx:x(p.t),cy:y(p.v),r:i===v2.length-1?2.8:2.0,fill:i===v2.length-1?C2:'var(--bg)',stroke:C2,'stroke-width':1.2})));
  const at=(e,a,xx)=>{const tx=el('text',{x:xx,y:H-3,'text-anchor':a});tx.textContent='E'+e;svg.append(tx);};
  at(all[0].m.epoch,'start',P.l); if(all.length>1) at(all[all.length-1].m.epoch,'end',W-P.r);
  // hover over union of points
  const hl=el('circle',{r:3.2,fill:C2,stroke:'var(--bg)','stroke-width':1,'pointer-events':'none',visibility:'hidden'});svg.append(hl);
  let tp=document.createElement('output');tp.className='chart-tip';tp.hidden=true;plot.append(tp);
  const pts=all; const xs=pts.map(p=>x(p.t));
  pts.forEach((p,i)=>{
    const l=i?(xs[i-1]+xs[i])/2:P.l, rr=i<pts.length-1?(xs[i]+xs[i+1])/2:W-P.r;
    const h=el('rect',{x:l,y:P.t,width:Math.max(1,rr-l),height:H-P.t-P.b,fill:'transparent'});
    h.addEventListener('mouseenter',()=>{
      hl.setAttribute('cx',xs[i]);hl.setAttribute('cy',y(p.v));hl.setAttribute('fill',p.proto==='v2'?C2:C1);hl.setAttribute('visibility','visible');
      const prot=p.proto==='v2'?'bench-v2 (64K)':'bench-v1 (32K-era)';
      tp.innerHTML=`<strong>E${esc(p.m.epoch)} · ${esc(kname(p.m))}</strong><span>${esc(s.lab)}</span> ${pct(p.v)}%${p.se!=null?' ±'+(100*p.se).toFixed(1):''}\n<span>${esc(prot)}</span>`;
      tp.hidden=false;const tw=tp.offsetWidth;tp.style.left=Math.min(Math.max(0,xs[i]-tw/2),Math.max(0,W-tw))+'px';tp.style.top=(y(p.v)>H/2?2:H/2)+'px';
    });
    h.addEventListener('mouseleave',()=>{tp.hidden=true;hl.setAttribute('visibility','hidden');});
    svg.append(h);
  });
 }
 function fit(){const sc=box.querySelector('.bench-scroll'),t=sc&&sc.querySelector('table');if(!t)return;const rs=t.tBodies[0]?t.tBodies[0].rows:[];
  if(rs.length<=5){sc.style.maxHeight='none';return;}
  let h=t.tHead.offsetHeight+(t.tFoot?t.tFoot.offsetHeight:0)+2;for(let i=0;i<5;i++)h+=rs[i].offsetHeight;sc.style.maxHeight=h+'px';}
 function layout(){box.querySelectorAll('.bench-spark').forEach(c=>{const s=SP[+c.dataset.i];if(s)spark(c,s);});fit();}
 let rt=0;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(layout,150);});
 let data=null;
 async function load(){try{const r=await (window.AFX&&AFX.take('/benchmarks.json')||fetch('/benchmarks.json',{cache:'no-store'}));if(!r.ok)throw 0;data=await r.json();table(data);}catch(e){}}
 load();setInterval(load,120000);
})();
