/* Uniform chart grid, 2026-10-25: give the rollout intake chart its own tile. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 function split(){
  const ro=document.getElementById('panel-rollouts');if(!ro)return;
  const plot=ro.querySelector(':scope>.sp-plot-intake');if(!plot)return;
  const legends=ro.querySelectorAll(':scope>.sp-legend'),legend2=legends.length>1?legends[legends.length-1]:null;
  let tile=document.getElementById('panel-intake');
  if(!tile){
   tile=document.createElement('section');tile.className='site-panel';tile.id='panel-intake';
   tile.setAttribute('aria-labelledby','sp-intake');tile.style.setProperty('--sp-color','var(--performance)');
   tile.innerHTML='<div class="section-heading"><h2 id="sp-intake">Rollout intake</h2></div>';
   ro.after(tile);
  }
  tile.append(plot);if(legend2)tile.append(legend2);
  ro.querySelector(':scope>.sp-subhead')?.remove();
 }
 for(const id of ['panel-weights','panel-mining'])document.getElementById(id)?.remove();
 {const tm=document.getElementById('panel-timing'),ho=document.getElementById('panel-heldout');if(tm&&ho)ho.before(tm);}
 {let prev=document.getElementById('panel-heldout');for(const el of [document.getElementById('panel-probe'),document.getElementById('panel-miners'),main.querySelector(':scope>section[aria-labelledby="batch-title"]'),document.getElementById('panel-groups')]){if(prev&&el){prev.after(el);prev=el;}}}
 new MutationObserver(split).observe(main,{childList:true,subtree:true});split();
})();
/* Title explanations, 2026-10-25: hover/focus/tap a tile title; the popover is placed outside the tile. */
(()=>{
 const main=document.getElementById('main');if(!main)return;
 const STALE=' Its source, network-data.json, is currently stale, so recent epochs are missing.';
 const T={
  'incentive-title':['Alpha paid to each miner UID in one chain tempo (360 blocks), sorted from highest to lowest; only UIDs that received emission are shown. The headline is the total paid to miners that tempo.',1],
  'flip-credit-title':['For each published epoch, the env probe compares every trained task before and after that epoch\u2019s checkpoint. Fail\u2192pass tasks earn credit (green bars), pass\u2192fail tasks are debited (red bars). The line is the net flip credit summed over all miners; only tasks that were failing beforehand and passed sampling/TOPLOC verification can earn credit.'],
  'flip-share-title':['Live flip-scoring leaderboard (scores.json). Share is each miner\u2019s current share of miner weight; Flip score is its smoothed net flip credit. f\u2192p and p\u2192f count its fail\u2192pass and pass\u2192fail tasks, and Eligible counts tasks it submitted while the probe was still failing them.'],
  'evaluation-title':['Mean reward of the evaluator on a fixed set of 32 affine_math tasks (same seed every run, 8,192-token output budget), one point per completed checkpoint evaluation. Higher means more of the fixed tasks solved.',1],
  'sp-heldout':['Share of a fixed 32-problem affine_math set solved by each checkpoint, labelled by the training epoch that produced it. These problems come from the held-out split, which never becomes miner tasks or training data, so this measures generalization rather than training fit.'],
  'sp-probe':['Pass rate of each new checkpoint on fresh random affine_math tasks, drawn with a seed fixed only after the checkpoint is published. It samples thousands of tasks per checkpoint, so it is far less noisy than the 32-problem held-out score. Pass = correct \u00f7 (correct + wrong + unresolved).'],
  'sp-miners':['Number of distinct miners whose submission was sealed into each epoch\u2019s frozen population. Hover a bar for registered miners and rejected submissions.'],
  'batch-title':['Total groups miners submitted each epoch, from the signed frozen population. Passed / failed / pending is the audit status; in the current mode every group is reported as pending.',1],
  'sp-groups':['Rollout groups submitted per epoch (each group is one task\u2019s set of rollouts). Fresh groups are new this epoch; reused groups were carried over from a miner\u2019s earlier submission.'],
  'sp-rollouts':['Every rollout the trainer used in each epoch, stacked by outcome: Pass = correct answer, Fail = wrong or no gradable answer, Cap = output hit the 8,192-token limit. The headline is the share of the latest trained epoch\u2019s rollouts that hit the cap. Empty slots are epochs with no training result (for example, rolled back).'],
  'sp-intake':['What happened to the groups sealed each epoch. Trained went into the update; Stripped were removed by the ban / TOPLOC verification gate; Rollback were sealed but not trained because the epoch was rolled back. Dup and Budget were not selected: duplicate tasks across miners, or over the epoch\u2019s group budget.'],
  'sp-grad':['Size of the gradient for each epoch\u2019s optimizer step, measured before clipping. Spikes mean an unusually large update; sustained high values can signal unstable training.'],
  'sp-loss':['The GRPO training objective for each epoch\u2019s optimizer step. With group-relative rewards it sits near zero and can be negative, so jumps and trends matter more than the level.'],
  'sp-timing':['Minutes spent in each phase of the last 20 epochs: Collect (submission window open), Wait + prepare (deadline to training start), Train, and Finalize (training done to epoch finalized). The headline is the latest epoch\u2019s start-to-finalized time.'],
 };
 const pop=document.createElement('div');pop.className='tile-explain';pop.id='tile-explain';pop.setAttribute('role','tooltip');pop.hidden=true;document.body.append(pop);
 let current=null,hideTimer=null;
 function text(id){const [t,stale]=T[id];return t+(stale&&/stale/i.test(document.getElementById('connection')?.textContent||'')?STALE:'');}
 function place(title){
  const tile=title.closest('section'),R=tile.getBoundingClientRect(),vw=document.documentElement.clientWidth,vh=window.innerHeight,g=8;
  const hb=document.querySelector('header');const top0=(hb&&getComputedStyle(hb).position==='sticky')?hb.getBoundingClientRect().bottom+g:g;
  pop.style.left='0px';pop.style.top='0px';pop.style.maxWidth=Math.min(320,vw-2*g)+'px';
  const P=pop.getBoundingClientRect(),w=P.width,h=P.height,clampX=x=>Math.max(g,Math.min(x,vw-w-g));
  const c=[];
  if(R.top-g-h>=top0)c.push([clampX(R.left),R.top-g-h,'above']);
  if(R.bottom+g+h<=vh-g)c.push([clampX(R.left),R.bottom+g,'below']);
  if(R.right+g+w<=vw-g)c.push([R.right+g,Math.max(top0,Math.min(R.top,vh-h-g)),'right']);
  if(R.left-g-w>=g)c.push([R.left-g-w,Math.max(top0,Math.min(R.top,vh-h-g)),'left']);
  const [x,y,side]=c[0]||(R.top-top0>vh-R.bottom?[clampX(R.left),Math.max(top0,R.top-g-h),'above']:[clampX(R.left),R.bottom+g,'below']);
  pop.style.left=x+'px';pop.style.top=y+'px';pop.dataset.side=side;
 }
 function show(title){clearTimeout(hideTimer);current=title;pop.textContent=text(title.id);pop.hidden=false;title.setAttribute('aria-expanded','true');place(title);}
 function hide(){clearTimeout(hideTimer);if(current)current.setAttribute('aria-expanded','false');current=null;pop.hidden=true;}
 let lastPointer='mouse';
 function wire(){
  for(const id of Object.keys(T)){const t=document.getElementById(id);if(!t||t.dataset.explain)continue;
   t.dataset.explain='1';t.classList.add('has-explain');t.tabIndex=0;t.setAttribute('aria-describedby','tile-explain');t.setAttribute('aria-expanded','false');
   t.addEventListener('pointerdown',e=>{lastPointer=e.pointerType;});
   t.addEventListener('mouseenter',()=>{if(lastPointer!=='touch')show(t);});
   t.addEventListener('mouseleave',()=>{if(lastPointer!=='touch')hideTimer=setTimeout(hide,80);});
   t.addEventListener('focus',()=>{if(lastPointer!=='touch')show(t);});
   t.addEventListener('blur',()=>{if(lastPointer!=='touch')hide();});
   t.addEventListener('click',e=>{if(lastPointer==='touch'){e.preventDefault();e.stopPropagation();current===t?hide():show(t);}});
   t.addEventListener('keydown',e=>{if(e.key==='Escape')hide();});}
 }
 document.addEventListener('pointerdown',e=>{if(current&&!e.target.closest('.has-explain'))hide();},true);
 window.addEventListener('scroll',()=>{if(current)place(current);},{passive:true});
 window.addEventListener('resize',()=>{if(current)place(current);});
 new MutationObserver(wire).observe(main,{childList:true,subtree:true});wire();
})();
