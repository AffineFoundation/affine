/* Layout only: every graph is rendered by the shared phone chart code. */
(()=>{
 const media=matchMedia('(min-width:1000px)'),market=document.querySelector('.market-prices');
 const home=document.createComment('market prices home');market.before(home);
 const readings=['tao-price','alpha-price','batches-total'].map(id=>document.getElementById(id));
 function fit(){
  for(const reading of readings){
   reading.parentElement.style.removeProperty('--desktop-reading-size');
   if(!media.matches)continue;
   const available=reading.parentElement.clientWidth;
   for(let i=0;i<4;i++){
    const width=reading.getBoundingClientRect().width,size=parseFloat(getComputedStyle(reading).fontSize);
    if(!width||width<=available-2)break;
    reading.parentElement.style.setProperty('--desktop-reading-size',Math.max(12,size*(available-4)/width)+'px');
   }
  }
 }
 function adapt(){
  if(media.matches)document.querySelector('header').insertBefore(market,document.querySelector('header nav'));
  else home.after(market);
  fit();
 }
 const observer=new MutationObserver(fit);
 readings.forEach(reading=>observer.observe(reading,{childList:true,characterData:true,subtree:true}));
 media.addEventListener('change',adapt);window.addEventListener('resize',fit);document.fonts.ready.then(fit);adapt();
})();

/* Appearance uses the same controls and chart palette on every viewport. */
(()=>{
 const root=document.documentElement,button=document.getElementById('theme-toggle');
 function sync(){const light=root.dataset.theme==='light';button.setAttribute('aria-pressed',String(light));button.setAttribute('aria-label',light?'Switch to dark mode':'Switch to light mode');button.title=button.getAttribute('aria-label');document.querySelector('meta[name="theme-color"]').content=light?'#ffffff':'#000000';}
 function apply(theme){if(theme==='light')root.dataset.theme='light';else delete root.dataset.theme;sync();window.dispatchEvent(new Event('affine:theme'));}
 button.addEventListener('click',()=>{const theme=root.dataset.theme==='light'?'dark':'light';try{localStorage.setItem('affine:theme',theme);}catch{}apply(theme);});
 window.addEventListener('storage',event=>{if(event.key==='affine:theme')apply(event.newValue);});sync();
})();

/* Desktop detail row. Counts use only the current public run projection. */
(()=>{
 const media=matchMedia('(min-width:1000px)'),ns='http://www.w3.org/2000/svg',main=document.getElementById('main');
 const valid=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0;
 const number=n=>n.toLocaleString('en-GB',{maximumFractionDigits:1});
 const compact=n=>n>=1000000?(n/1000000).toFixed(1)+'m':n>=1000?(n/1000).toFixed(1)+'k':number(n);
 const specs=[
  {id:'samples',title:'Submitted samples',color:'batch',legend:'Submitted rollouts · K + L per batch',views:['Cumulative','Per epoch']},
  {id:'activity',title:'Active miners',color:'accepted',legend:'UIDs with submitted batches',views:['Trend','Bars']},
  {id:'inclusion',title:'Included batches',color:'performance',legend:'Included / submitted · training admission',views:['Rate','Count']},
  {id:'distribution',title:'Contribution distribution',color:'incentive',legend:'Miners grouped by submitted batch count',views:['Histogram','Cumulative']}
 ];
 const panels=specs.map(spec=>{
  const section=document.createElement('section');section.className='desktop-detail';section.setAttribute('aria-labelledby','detail-'+spec.id);section.style.setProperty('--detail-color','var(--'+spec.color+')');
  section.innerHTML='<div class="section-heading"><h2 id="detail-'+spec.id+'">'+spec.title+'</h2><span class="reading">—</span></div><div class="detail-controls"><span class="detail-context">Loading</span><div class="miner-view-control" role="group" aria-label="'+spec.title+' view"></div></div><div class="detail-plot"><svg role="group" aria-label="'+spec.title+'"></svg><p class="empty" hidden></p><output class="detail-tip" hidden></output></div><div class="legend detail-legend"><i class="swatch"></i><span>'+spec.legend+'</span></div>';
  const p={spec,section,view:0,svg:section.querySelector('svg'),tip:section.querySelector('output'),key:null};
  spec.views.forEach((label,i)=>{const b=document.createElement('button');b.type='button';b.textContent=label;b.setAttribute('aria-pressed',String(i===0));b.addEventListener('click',()=>{p.view=i;p.key=null;draw(p);});section.querySelector('[role="group"].miner-view-control').append(b);});
  main.append(section);return p;
 });
 let snapshot=null,rows=[];
 function element(tag,attrs={},text){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
 function selectedIndex(){return Math.min(rows.length-1,Math.max(0,Number(document.getElementById('desktop-epoch-slider').value)||0));}
 function active(r){return Array.isArray(r.grid)?r.grid.filter(n=>valid(n)&&n>0).length:null;}
 function samples(r){return r.batches_available!==false&&valid(r.batches)&&valid(r.k)&&valid(r.l)&&r.k+r.l>0?r.batches*(r.k+r.l):null;}
 function included(r){return valid(r.learner_eligible)&&r.learner_eligible<=r.batches?r.learner_eligible:null;}
 function epoch(r){return String(r.id).match(/-(\d+)$/)?.[1]||String(r.id);}
 function draw(p){
  if(!media.matches||!snapshot)return;
  const index=selectedIndex(),selected=rows[index],view=p.view,id=p.spec.id;
  const key=JSON.stringify([snapshot.training_run?.run_id,rows,index,view]);if(key===p.key)return;p.key=key;
  let points=[],headline=null,unit='',context=selected?'Epoch '+epoch(selected):'No epochs';
  if(id==='samples'){
   let sum=0,complete=true;
   points=rows.map(r=>{const n=samples(r);if(n===null)complete=false;else sum+=n;return{x:epoch(r),y:view?n:complete?sum:null,description:'Submitted samples',row:r};});headline=points[index]?.y;context=(view?'Epoch ':'Through epoch ')+(selected?epoch(selected):'—');
  }else if(id==='activity'){points=rows.map(r=>({x:epoch(r),y:active(r),description:'Submitting miners',row:r}));headline=points[index]?.y;unit='miners';}
  else if(id==='inclusion'){
   points=rows.map(r=>{const n=included(r);return{x:epoch(r),y:n===null?null:view?n:r.batches>0?100*n/r.batches:null,description:view?'Included batches':'Included %',row:r};});headline=points[index]?.y;unit=view?'batches':'%';
  }else if(selected&&Array.isArray(selected.grid)){
   const buckets=new Map();for(const n of selected.grid)if(valid(n)&&n>0)buckets.set(n,(buckets.get(n)||0)+1);
   let sum=0;points=[...buckets].sort((a,b)=>a[0]-b[0]).map(([n,c])=>({x:n,y:view?(sum+=c):c,description:view?'Miners with ≤ '+n+' batches':n+' batches per miner',row:selected}));headline=active(selected);unit='miners';
  }
  const section=p.section;section.querySelector('.reading').textContent=headline===null||headline===undefined?'—':number(headline)+(unit==='%'?'%':'');section.querySelector('.reading').setAttribute('aria-label',headline==null?'Measurement unavailable':number(headline)+' '+unit);
  section.querySelector('.detail-context').textContent=context;
  section.querySelectorAll('.miner-view-control button').forEach((b,i)=>{b.setAttribute('aria-pressed',String(i===view));b.disabled=!rows.length;});
  p.tip.hidden=true;p.svg.replaceChildren();p.svg.setAttribute('viewBox','0 0 480 230');
  const usable=points.filter(n=>valid(n.y)),empty=section.querySelector('.empty');empty.hidden=usable.length>0;empty.textContent=id==='inclusion'?'Training admission unavailable.':'No measurements yet.';if(!usable.length)return;
  const peak=Math.max(1,...usable.map(n=>n.y)),magnitude=10**Math.floor(Math.log10(peak)),niceMax=[1,2,5,10].map(n=>n*magnitude).find(n=>n>=peak);
  const max=id==='inclusion'&&!view?100:niceMax,L=42,R=468,T=12,B=190,w=R-L;
  const x=i=>L+w*(i+.5)/points.length,y=n=>B-(B-T)*n/max;
  for(const tick of[0,max/2,max]){const yy=y(tick);p.svg.append(element('line',{x1:L,x2:R,y1:yy,y2:yy,class:'detail-guide'}),element('text',{x:L-8,y:yy+4,'text-anchor':'end'},compact(tick)+(id==='inclusion'&&!view?'%':'')));}
  const bars=id==='distribution'&&!view||id==='activity'&&view||id==='samples'&&view;
  if(!bars){let d='',open=false;points.forEach((n,i)=>{if(!valid(n.y)){open=false;return;}d+=(open?' L':' M')+x(i)+' '+y(n.y);open=true;});p.svg.append(element('path',{d,class:'detail-line detail-roll-in'}));}
  points.forEach((n,i)=>{
   if(!valid(n.y))return;
   const chosen=id==='distribution'||i===index;
   if(bars)p.svg.append(element('rect',{x:x(i)-w/points.length*.44,y:y(n.y),width:w/points.length*.88,height:B-y(n.y),class:'detail-bar detail-rise',opacity:chosen?1:.65}));
   else p.svg.append(element('circle',{cx:x(i),cy:y(n.y),r:chosen?4:2.5,class:'detail-dot'}));
   if(id==='distribution'||points.length<=8||i===0||i===points.length-1||i===index)p.svg.append(element('text',{x:x(i),y:B+18,'text-anchor':'middle'},String(n.x)));
   const hit=element('rect',{x:x(i)-w/points.length/2,y:T,width:w/points.length,height:B-T+22,fill:'transparent',tabindex:0,role:'img','aria-label':'Epoch '+epoch(n.row)+', '+n.description+': '+number(n.y)});
   const show=()=>{if(p.dismissed)return;p.tip.replaceChildren();const title=document.createElement('span');title.textContent=id==='distribution'?n.description:'Epoch '+epoch(n.row);const b=document.createElement('b');b.textContent=number(n.y)+(id==='inclusion'&&!view?'%':'');p.tip.append(title,b);if(id==='samples'){const s=document.createElement('small');s.textContent=number(n.row.batches)+' batches · '+(n.row.k+n.row.l)+' rollouts / batch';p.tip.append(s);}p.tip.hidden=false;};
   hit.addEventListener('pointerenter',show);hit.addEventListener('pointermove',()=>{p.dismissed=false;show();});hit.addEventListener('focus',()=>{p.dismissed=false;show();});hit.addEventListener('pointerleave',()=>p.tip.hidden=true);hit.addEventListener('blur',()=>p.tip.hidden=true);hit.addEventListener('keydown',e=>{if(e.key==='Escape'){p.dismissed=true;p.tip.hidden=true;}if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();const hits=[...p.svg.querySelectorAll('[tabindex]')],j=hits.indexOf(hit);hits[Math.max(0,Math.min(hits.length-1,j+(e.key==='ArrowRight'?1:-1)))].focus();}});p.svg.append(hit);
  });
  p.svg.append(element('text',{x:(L+R)/2,y:225,'text-anchor':'middle'},id==='distribution'?'Batches per miner':'Epoch'));
 }
 function render(){panels.forEach(draw);}
 window.addEventListener('affine:snapshot',event=>{snapshot=event.detail;rows=snapshot.epochs.filter(r=>r.source==='live-reward-math'&&valid(r.start)&&valid(r.batches)).sort((a,b)=>a.start-b.start);render();});
 document.getElementById('desktop-epoch-slider').addEventListener('input',render);
 media.addEventListener('change',()=>{panels.forEach(p=>p.key=null);render();});
})();

/* Full task-index map and task × epoch matrix, scoped to one corpus version. */
(()=>{
 const media=matchMedia('(min-width:1000px)'),section=document.createElement('section');section.className='desktop-detail task-concentration';section.style.setProperty('--detail-color','var(--performance)');section.setAttribute('aria-labelledby','task-title');
 section.innerHTML='<div class="section-heading"><h2 id="task-title">Task concentration</h2><span class="task-reading">—</span></div><div class="task-controls"><span class="task-scope">Current corpus</span><div class="miner-view-control task-metric" role="group" aria-label="Task measurement"><button type="button" aria-pressed="true">Submitted</button><button type="button" aria-pressed="false">Verified solves</button></div><div class="miner-view-control task-view" role="group" aria-label="Task concentration view"><button type="button" aria-pressed="true">Corpus</button><button type="button" aria-pressed="false">Epochs</button></div><label>Task <input class="task-lookup" type="number" min="0" value="0" aria-label="Inspect corpus task index"></label></div><div class="task-scroll"><canvas tabindex="0" role="group" aria-label="Corpus task concentration. Enter a task index to inspect its epoch history."></canvas><p class="task-empty" hidden></p></div><div class="task-footer"><div class="legend"><span><i class="swatch task-zero"></i>Unobserved</span><span><i class="swatch task-low"></i>Less</span><span><i class="swatch task-high"></i>More</span><span class="task-concentration-stat"></span></div><output class="task-inspection" aria-live="polite"></output></div><p class="task-note">Submitted counts are declared batches. Verified solves count reward-1 rollouts in fully audited accepted batches.</p>';
 document.getElementById('main').append(section);
 const canvas=section.querySelector('canvas'),input=section.querySelector('input'),output=section.querySelector('output'),scroll=section.querySelector('.task-scroll');let data=null,metric='submitted',view=0,selected=0,layout=null;
 const color=name=>getComputedStyle(document.documentElement).getPropertyValue('--'+name).trim();const integer=n=>Number.isInteger(n)&&n>=0;
 const epoch=r=>String(r.id).match(/-(\d+)$/)?.[1]||String(r.id),format=n=>n.toLocaleString('en-GB');
 function inspect(index){if(!layout)return;selected=Math.max(0,Math.min(layout.corpus.size-1,index));input.value=selected;const values=layout.rows.map(r=>{const n=r.task_concentration?.tasks.find(t=>t.environment===layout.corpus.environment&&t.index===selected);return {epoch:epoch(r),value:layout.available(r)?n?.[metric]||0:null};});const observed=values.filter(n=>n.value>0),total=observed.reduce((s,n)=>s+n.value,0);output.textContent='Task '+selected+' · '+format(total)+' '+(metric==='submitted'?'batches':'verified solves')+' · '+observed.length+' epochs'+(observed.length?' · '+observed.map(n=>'E'+n.epoch+': '+n.value).join(' · '):' · No recorded results')+(values.some(n=>n.value===null)?' · Some epochs unavailable':'');}
 function draw(){
  if(!media.matches||!data)return;
  const all=data.epochs.filter(r=>r.source==='live-reward-math').sort((a,b)=>a.start-b.start),cut=Math.min(all.length-1,Number(document.getElementById('desktop-epoch-slider').value)||0),row=all[cut];
  const corpus=row?.task_concentration?.corpora.find(c=>c.environment==='affine_math')||row?.task_concentration?.corpora[0];
  const empty=section.querySelector('.task-empty');layout=null;canvas.hidden=false;
  if(!corpus){canvas.hidden=true;empty.hidden=false;empty.textContent='Task records unavailable for this epoch.';section.querySelector('.task-reading').textContent='—';output.textContent='';return;}
  const rows=all.slice(0,cut+1).filter(r=>r.task_concentration?.corpora.some(c=>c.key===corpus.key));
  const available=r=>r.task_concentration?.[metric==='submitted'?'submitted_available':'solves_available']===true;
  const totals=new Map();rows.forEach(r=>{if(!available(r))return;for(const t of r.task_concentration.tasks)if(t.environment===corpus.environment&&integer(t.index)&&integer(t[metric])&&t[metric]>0)totals.set(t.index,(totals.get(t.index)||0)+t[metric]);});
  const tasks=[...totals.keys()].sort((a,b)=>a-b),max=Math.max(1,...totals.values()),total=[...totals.values()].reduce((s,n)=>s+n,0),top=[...totals.values()].sort((a,b)=>b-a).slice(0,10).reduce((s,n)=>s+n,0);
  section.querySelector('.task-reading').textContent=rows.some(available)?format(tasks.length)+' / '+format(corpus.size)+' tasks':'—';section.querySelector('.task-scope').textContent=corpus.environment+' · through epoch '+epoch(row);section.querySelector('.task-concentration-stat').textContent=total?'Top 10 tasks: '+(100*top/total).toFixed(1)+'%':'';
  input.max=corpus.size-1;section.querySelectorAll('.task-metric button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===(metric==='submitted'?0:1))));section.querySelectorAll('.task-view button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===view)));
  empty.hidden=rows.some(available)&&(!view||tasks.length>0);empty.textContent=metric==='solves'?'No verified solve outcomes yet. Submitted view shows task activity.':'No submitted task results yet.';
  if(!empty.hidden){canvas.hidden=true;output.textContent='';return;}
  const width=Math.max(600,scroll.clientWidth),cols=view?rows.length:Math.max(32,Math.floor(width/11)),cellWidth=view?Math.max(36,(width-60)/cols):width/cols,cellHeight=view?16:9,height=view?32+Math.max(1,tasks.length)*cellHeight:Math.ceil(corpus.size/cols)*cellHeight;
  const drawWidth=view?Math.max(width,60+cols*cellWidth):width,dpr=devicePixelRatio||1;canvas.width=drawWidth*dpr;canvas.height=height*dpr;canvas.style.width=drawWidth+'px';canvas.style.height=height+'px';const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const zero=color('zero'),high=color('performance');ctx.font='10px "DM Mono",monospace';ctx.textBaseline='middle';
  const fill=(x,y,value,unknown=false)=>{ctx.fillStyle=unknown?color('unchecked'):zero;ctx.globalAlpha=unknown?.22:1;ctx.fillRect(x,y,cellWidth-2,cellHeight-2);if(value>0){ctx.fillStyle=high;ctx.globalAlpha=.3+.7*Math.sqrt(value/max);ctx.fillRect(x,y,cellWidth-2,cellHeight-2);}ctx.globalAlpha=1;};
  if(!view){for(let i=0;i<corpus.size;i++)fill(i%cols*cellWidth,Math.floor(i/cols)*cellHeight,totals.get(i)||0);}
  else{rows.forEach((r,i)=>{ctx.fillStyle=color('muted');ctx.fillText('E'+epoch(r),60+i*cellWidth+4,14);});const lookups=rows.map(r=>new Map(r.task_concentration.tasks.filter(t=>t.environment===corpus.environment).map(t=>[t.index,t[metric]])));tasks.forEach((task,j)=>{ctx.fillStyle=color('muted');ctx.fillText('#'+task,0,32+j*cellHeight+6);rows.forEach((r,i)=>fill(60+i*cellWidth,32+j*cellHeight,lookups[i].get(task)||0,!available(r)));});}
  layout={corpus,rows,available,cols,cellWidth,cellHeight,tasks};canvas.classList.remove('task-enter');void canvas.offsetWidth;canvas.classList.add('task-enter');inspect(selected);
 }
 section.querySelectorAll('.task-metric button').forEach((b,i)=>b.addEventListener('click',()=>{metric=i?'solves':'submitted';draw();}));section.querySelectorAll('.task-view button').forEach((b,i)=>b.addEventListener('click',()=>{view=i;draw();}));input.addEventListener('input',()=>{const value=Number(input.value);if(integer(value))inspect(value);});
 canvas.addEventListener('pointermove',event=>{if(!layout)return;const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;const index=view?layout.tasks[Math.floor((y-32)/layout.cellHeight)]:Math.floor(y/layout.cellHeight)*layout.cols+Math.floor(x/layout.cellWidth);if(integer(index)&&index<layout.corpus.size)inspect(index);});
 canvas.addEventListener('keydown',e=>{if(!layout)return;const step={ArrowLeft:-1,ArrowRight:1,ArrowUp:-layout.cols,ArrowDown:layout.cols}[e.key];if(step){e.preventDefault();inspect(selected+step);}});
 window.addEventListener('affine:snapshot',e=>{data=e.detail;draw();});window.addEventListener('affine:theme',draw);window.addEventListener('resize',draw);media.addEventListener('change',draw);
})();
