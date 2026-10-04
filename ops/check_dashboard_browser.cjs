// Read-only public dashboard check. Synthetic response controls stay in this browser.
const fs=require('fs'),crypto=require('crypto'),path=require('path');
const {chromium}=require('playwright');
const site=process.argv[2]||'https://affine.io',output=process.argv[3]||'state/dashboard';
fs.mkdirSync(output,{recursive:true});
const publicDirectory=path.resolve(__dirname,'../dashboard/public');
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const cohort=e=>JSON.stringify([e.dataset_id,e.taskset_hash,e.fixed_task_ids,e.seed,e.count,e.requested_count,e.harness,e.environment_version,e.model,e.model_runtime_revision,e.output_token_budget,e.policy_kind]);
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const count=n=>finite(n)&&Number.isInteger(n)&&n>=0;
const percentage=n=>`${Number((n*100).toFixed(1))}%`;
const number=n=>n.toLocaleString('en-GB');
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const assert=(condition,message)=>{if(!condition)throw Error(message);};
function expected(snapshot){
 const epochs=snapshot.epochs.filter(e=>e.source==='live-reward-math'&&finite(e.start)).sort((a,b)=>a.start-b.start);
 const ids=new Set(epochs.map(e=>e.id));
 const evaluations=snapshot.evaluations.filter(e=>ids.has(e.epoch_id)&&e.env_id==='affine_math'&&e.status==='complete'&&finite(e.timestamp)&&finite(e.mean_reward)&&e.mean_reward>=0&&e.mean_reward<=1&&count(e.count)&&e.count>0&&count(e.successes)&&e.successes<=e.count).sort((a,b)=>a.timestamp-b.timestamp);
 const latest=evaluations.at(-1);
 return {evaluations:latest?evaluations.filter(e=>cohort(e)===cohort(latest)):[],epochs:epochs.filter(e=>e.finalized&&count(e.batches))};
}
async function structure(page){
 const actual=await page.evaluate(()=>({charts:document.querySelectorAll('main .chart-wrap>svg').length,sections:document.querySelectorAll('main>section').length,removed_controls:!!document.querySelector('#grid,#metric,#environment,#evaluation,#batch-source,[data-scope]'),llms_links:document.querySelectorAll('a[href="/llms.txt"]').length,github_links:document.querySelectorAll('a[href="https://github.com/AffineFoundation/affine"]').length,overflow:document.documentElement.scrollWidth>innerWidth}));
 assert(actual.charts===2&&actual.sections===2&&!actual.removed_controls,'Current two-chart structure mismatch');
 assert(actual.llms_links>0&&actual.github_links>0&&!actual.overflow,'Document links or viewport layout mismatch');return actual;
}
async function plotted(page,kind){return page.locator(`#${kind}-chart .chart-point`).evaluateAll(nodes=>nodes.map(n=>({value:Number(n.dataset.value),record:n.dataset.record})));}
async function compare(page,snapshot){
 const rows=expected(snapshot);
 for(const [kind,entries,id,value] of [['evaluation',rows.evaluations,'run_id','mean_reward'],['batch',rows.epochs,'id','batches']]){
  assert(same(await plotted(page,kind),entries.map(e=>({value:e[value],record:e[id]}))),`${kind} chart differs from actual public JSON`);
  assert(await page.locator(`#${kind}-empty`).isVisible()===(entries.length===0),`${kind} empty-state mismatch`);
 }
 const evaluation=rows.evaluations.at(-1),epoch=rows.epochs.at(-1);
 assert(await page.locator('#evaluation-value').textContent()===(evaluation?percentage(evaluation.mean_reward):'—'),'Latest measured evaluation value mismatch');
 assert(await page.locator('#batch-value').textContent()===(epoch?number(epoch.batches):'—'),'Latest frozen batch value mismatch');
 if(evaluation){assert((await page.locator('#evaluation-reading').textContent()).includes(`${evaluation.successes} / ${evaluation.count}`),'Held-out success count mismatch');assert((await page.locator('#evaluation-note').textContent()).includes(`${evaluation.count} fixed held-out problems`),'Held-out cohort size missing');}
 return rows;
}
async function waitRendered(page){await page.waitForFunction(()=>['current','stale','unavailable'].includes(document.querySelector('#connection')?.dataset.state));}
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.AFFINE_CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const errors=[],viewports=[],controls=[],assetChecks=[];let snapshot,publicResponseSHA;
 const observe=(page,label)=>{page.on('pageerror',error=>errors.push({label,error:String(error)}));};
 try{
  for(const width of [1440,390,320]){
   const page=await browser.newPage({viewport:{width,height:width===1440?1100:900},isMobile:width<600,hasTouch:width<600});observe(page,`actual-${width}`);
   const consoleErrors=[],failedRequests=[];page.on('console',message=>{if(message.type()==='error')consoleErrors.push(message.text());});page.on('requestfailed',request=>failedRequests.push({url:request.url(),error:request.failure()?.errorText}));
   const pending=page.waitForResponse(response=>new URL(response.url()).pathname==='/network-data.json'&&response.status()===200,{timeout:30000});
   const documentResponse=await page.goto(site,{waitUntil:'domcontentloaded',timeout:30000});
   assert(documentResponse?.ok(),'Public dashboard document unavailable');
   const response=await pending,raw=await response.body(),current=JSON.parse(raw);assert(Array.isArray(current.epochs)&&Array.isArray(current.evaluations),'Invalid public snapshot');
   await waitRendered(page);await page.evaluate(()=>document.fonts.ready);
   assert((await page.locator('#connection').getAttribute('data-state'))!=='unavailable','Public records did not render');
   const actual=await structure(page),rows=await compare(page,current);
   await page.screenshot({path:path.join(output,`two-charts-${width}.png`),fullPage:true});
   const interactions=[];
   for(const [kind,entries,id] of [['evaluation',rows.evaluations,'run_id'],['batch',rows.epochs,'id']]){
    if(!entries.length)continue;
    const last=page.locator(`#${kind}-chart .chart-hit`).last();
    if(width<600)await last.tap();else await last.focus();
    assert(await page.locator(`#${kind}-tip`).isVisible(),`${kind} point inspection unavailable`);
    const tip=await page.locator(`#${kind}-tip`).textContent(),entry=entries.at(-1);
    if(kind==='evaluation')assert(tip.includes(percentage(entry.mean_reward))&&tip.includes(`${entry.successes}/${entry.count}`),'Measured evaluation tooltip mismatch');
    else for(const field of ['accepted','rejected','unchecked'])if(count(entry[field]))assert(tip.includes(`${number(entry[field])} ${field==='accepted'?'fully audited and accepted':field}`),`Batch ${field} tooltip mismatch`);
    await last.focus();await page.keyboard.press('Home');
    assert(await page.locator(`#${kind}-chart .chart-hit[tabindex="0"]`).getAttribute('data-record')===entries[0][id],'Home key did not select first point');
    await page.keyboard.press('End');assert(await page.locator(`#${kind}-chart .chart-hit[tabindex="0"]`).getAttribute('data-record')===entries.at(-1)[id],'End key did not select last point');
    await page.keyboard.press('Escape');assert(await page.locator(`#${kind}-tip`).isHidden(),'Escape did not dismiss chart inspection');
    interactions.push({kind,touch:width<600,keyboard:true});
   }
   assert(consoleErrors.length===0&&failedRequests.length===0,'Actual public browser console/request failures');
   viewports.push({width,...actual,evaluation_points:rows.evaluations.length,batch_points:rows.epochs.length,interactions,console_errors:consoleErrors,failed_requests:failedRequests,public_response_sha256:hash(raw)});
   if(!snapshot){snapshot=current;publicResponseSHA=hash(raw);
    const html=await documentResponse.body();
    // Cloudflare may append its analytics beacon only to browser HTML responses.
    // Allow that exact provider insertion, not arbitrary scripts or HTML changes.
    const beacon=/<script\b[^>]*\bsrc=["']https:\/\/static\.cloudflareinsights\.com\/beacon\.min\.js(?:\/[^"']*)?["'][^>]*>\s*<\/script>(?:\r?\n)?/g;
    const beaconCount=(html.toString('utf8').match(beacon)||[]).length,sourceHTML=html.toString('utf8').replace(beacon,'');
    assetChecks.push({file:'index.html',sha256:hash(html),source_sha256:hash(sourceHTML),cloudflare_beacon_count:beaconCount,matches_checkout:beaconCount<=1&&hash(sourceHTML)===hash(fs.readFileSync(path.join(publicDirectory,'index.html')))});
    for(const [file,selector,attribute] of [['network.css','link[rel="stylesheet"]','href'],['network.js','script[src^="/network.js"]','src']]){
     const assetURL=new URL(await page.locator(selector).getAttribute(attribute),site).href,assetResponse=await page.request.get(assetURL);assert(assetResponse.ok(),`${file} unavailable`);const bytes=await assetResponse.body();assetChecks.push({file,sha256:hash(bytes),matches_checkout:hash(bytes)===hash(fs.readFileSync(path.join(publicDirectory,file)))});
    }
    const guideResponse=await page.request.get(new URL('/llms.txt',site).href);assert(guideResponse.ok()&&/text\/plain/.test(guideResponse.headers()['content-type']||''),'Public guide unavailable or not plain text');const guide=await guideResponse.body();assetChecks.push({file:'llms.txt',sha256:hash(guide),matches_checkout:hash(guide)===hash(fs.readFileSync(path.join(publicDirectory,'llms.txt')))});
   }
   await page.close();
  }
  fs.writeFileSync(path.join(output,'asset-readback.json'),JSON.stringify(assetChecks,null,2)+'\n');
  assert(assetChecks.every(asset=>asset.matches_checkout),'Published dashboard assets or public guide differ from checkout (see asset-readback.json)');
  const localPage=async fixture=>{
   const page=await browser.newPage({viewport:{width:390,height:900},isMobile:true,hasTouch:true});observe(page,'intercepted-control');
   // Expose the existing refresh callback only inside this test page; production code is unchanged.
   await page.addInitScript(()=>{const original=window.setInterval;window.setInterval=(fn,delay)=>{if(delay===15000)window.__dashboardCheckRefresh=fn;return original(fn,delay);};});
   await page.route('**/network-data.json',route=>{const response=fixture();return route.fulfill({status:response.status||200,contentType:'application/json',body:JSON.stringify(response.body||{})});});
   await page.goto(site,{waitUntil:'domcontentloaded',timeout:30000});await waitRendered(page);return page;
  };
  for(const unavailable of [false,true]){
   const page=await localPage(()=>({status:unavailable?503:200,body:{epochs:[],evaluations:[],summary:{updated_at:Date.now()/1000}}}));
   assert(await page.locator('.chart-point').count()===0&&await page.locator('#evaluation-empty').isVisible()&&await page.locator('#batch-empty').isVisible(),'Fabricated empty/unavailable points');
   assert(await page.locator('#evaluation-value').textContent()==='—'&&await page.locator('#batch-value').textContent()==='—','Fabricated empty/unavailable headline values');
   await structure(page);await page.screenshot({path:path.join(output,unavailable?'local-unavailable.png':'local-empty.png'),fullPage:true});
   controls.push({intercepted_response:unavailable?'unavailable':'empty',points:0,headline_values_unavailable:true});await page.close();
  }
  const sample=expected(snapshot).evaluations.at(-1),sampleEpoch=expected(snapshot).epochs.at(-1)||snapshot.epochs.find(e=>e.source==='live-reward-math');
  if(sample&&sampleEpoch){
   // Isolate every comparison identity without publishing these synthetic measurements.
   for(const field of ['dataset_id','taskset_hash','fixed_task_ids','seed','count','requested_count','harness','environment_version','model','model_runtime_revision','output_token_budget','policy_kind']){
    const value=field==='fixed_task_ids'?['local-changed-task']:field==='count'?sample.count+1:typeof sample[field]==='number'?sample[field]+1:'local-changed-identity';
    const changed={...sample,[field]:value,run_id:`local-changed-${field}`,timestamp:sample.timestamp+1};
    const fixture={epochs:[sampleEpoch],evaluations:[sample,changed],summary:{updated_at:Date.now()/1000}};
    const page=await localPage(()=>({body:fixture}));await compare(page,fixture);assert(await page.locator('#evaluation-chart .chart-point').count()===1,`Different ${field} cohorts were merged`);await page.close();
   }
   controls.push({intercepted_response:'cohort-identity-mutations',independent_fields:12,each_points:1});
   const checkpointOnly={...sample,run_id:'local-checkpoint-only',checkpoint:'local-changed-checkpoint',timestamp:sample.timestamp+1};
   const qualification={...sampleEpoch,id:'local-qualification-100',source:'qualification',start:sampleEpoch.start+1,batches:999999},pending={...sampleEpoch,id:'local-current-pending-101',start:sampleEpoch.start+2,finalized:false,batches:999999};
   const fixture={epochs:[sampleEpoch,qualification,pending],evaluations:[sample,checkpointOnly,{...sample,run_id:'local-incomplete',timestamp:sample.timestamp+2,status:'pending'},{...sample,run_id:'local-unrelated',timestamp:sample.timestamp+3,epoch_id:qualification.id}],summary:{updated_at:Date.now()/1000}};
   let unavailable=false;const page=await localPage(()=>unavailable?{status:503}:{body:fixture});await compare(page,fixture);
   assert(await page.locator('#evaluation-chart .chart-point').count()===2,'Checkpoint change split an otherwise identical cohort');assert(await page.locator('#batch-chart .chart-point').count()===(sampleEpoch.finalized?1:0),'Pending or qualification work became a finalized batch point');
   const focused=page.locator('#evaluation-chart .chart-hit').first();await focused.focus();await page.evaluate(()=>window.__dashboardCheckRefresh());assert(await page.evaluate(()=>document.activeElement?.dataset.record)===''+sample.run_id,'Periodic refresh lost keyboard focus');
   unavailable=true;await page.evaluate(()=>window.__dashboardCheckRefresh());assert(await page.locator('#evaluation-chart .chart-point').count()===2,'Failed refresh discarded committed measurements');assert((await page.locator('#connection').textContent()).includes('showing snapshot'),'Failed refresh omitted last-snapshot warning');
   await page.setViewportSize({width:320,height:900});await page.waitForTimeout(150);assert((await page.locator('#connection').textContent()).includes('unavailable'),'Resize cleared unavailable state');await structure(page);
   unavailable=false;fixture.summary.updated_at=Date.now()/1000-1000;await page.evaluate(()=>window.__dashboardCheckRefresh());assert(await page.locator('#connection').getAttribute('data-state')==='stale','Stale snapshot was labelled current');
   controls.push({intercepted_response:'selection-and-refresh-fixtures',checkpoint_changes_comparable:true,qualification_and_pending_excluded:true,keyboard_focus_preserved:true,failed_refresh_keeps_snapshot:true,unavailable_state_survives_resize:true,stale_snapshot_labelled:true});await page.close();
  }else controls.push({intercepted_response:'cohort-and-refresh-fixtures',skipped:'No current complete evaluation to derive a control fixture'});
  assert(errors.length===0,'Browser page errors: '+JSON.stringify(errors));
  const receipt={passed:true,checked_at:Date.now()/1000,site,actual_page_checked:true,public_response_sha256:publicResponseSHA,current_source:'live-reward-math',selection:'latest comparable completed affine_math cohort and finalized current-run epochs',viewports,published_assets:assetChecks,local_controls:controls,page_errors:errors,chain_transactions:false,quality_improvement_claimed:false};
  fs.writeFileSync(path.join(output,'two-chart-browser-check.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify({passed:true,viewports:viewports.map(view=>view.width),evaluation_points:viewports[0].evaluation_points,batch_points:viewports[0].batch_points,published_assets_match:assetChecks.every(asset=>asset.matches_checkout),local_controls:controls.length}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1;});
