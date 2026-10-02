// Read-only public dashboard smoke check. Empty/error controls intercept responses locally.
const fs=require('fs'),crypto=require('crypto'),path=require('path');
const {chromium}=require('playwright');
const site=process.argv[2]||'https://affine.io',output=process.argv[3]||'state/dashboard';
fs.mkdirSync(output,{recursive:true});
const key=e=>JSON.stringify([e.dataset_id,e.taskset_hash,e.fixed_task_ids,e.seed,e.count,e.requested_count,e.harness,e.environment_version,e.model,e.model_runtime_revision,e.output_token_budget,e.policy_kind]);
const assert=(condition,message)=>{if(!condition)throw Error(message);};
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.AFFINE_CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 try{
  const pending=page.waitForResponse(r=>r.url().endsWith('/network-data.json')&&r.status()===200);
  await page.goto(site,{waitUntil:'domcontentloaded',timeout:30000});
  const response=await pending,raw=await response.body(),snapshot=JSON.parse(raw);
  await page.waitForFunction(()=>document.querySelector('#connection')?.textContent.includes('updated'));
  const structure=await page.evaluate(()=>({charts:document.querySelectorAll('main svg[role=img]').length,sections:document.querySelectorAll('main section').length,grid:!!document.querySelector('#grid'),metric:!!document.querySelector('#metric'),llms:document.querySelector('footer a')?.getAttribute('href')}));
  assert(structure.charts===2&&structure.sections===2&&!structure.grid&&!structure.metric&&structure.llms==='/llms.txt','Two-chart structure mismatch');
  const defaultView=await page.evaluate(()=>({environment:document.querySelector('#environment').value,source:document.querySelector('#batch-source').value,cohort:document.querySelector('#evaluation').value}));
  const mode=e=>snapshot.epochs.find(x=>x.id===e.epoch_id)?.mode||(/^(nonpayable-|test-|mock-)/.test(e.epoch_id||'')?'test':'live');
  const values=async kind=>page.locator('#'+kind+'-chart circle').evaluateAll(rows=>rows.map(x=>Number(x.dataset.value)));
  const recordIds=async kind=>page.locator('#'+kind+'-chart circle').evaluateAll(rows=>rows.map(x=>x.dataset.record));
  const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const checks=[],batchChecks=[];
  for(const scope of ['test','live']){
   await page.locator(`[data-scope="${scope}"]`).click();
   const envs=await page.locator('#environment option').evaluateAll(rows=>rows.map(x=>x.value));
   for(const env of envs){
    await page.selectOption('#environment',env);
    const cohorts=await page.locator('#evaluation option').evaluateAll(rows=>rows.map(x=>x.value));
    for(const cohort of cohorts){
     await page.selectOption('#evaluation',cohort);
     const expected=snapshot.evaluations.filter(e=>e.env_id===env&&key(e)===cohort&&e.status==='complete'&&mode(e)===scope&&Number.isFinite(e.mean_reward)).sort((a,b)=>a.timestamp-b.timestamp);
     assert(expected.length&&same(await values('evaluation'),expected.map(e=>e.mean_reward))&&same(await recordIds('evaluation'),expected.map(e=>e.run_id)),'Evaluation mismatch: '+env);
     await page.locator('#evaluation-chart .chart-hit').last().focus();
     const tip=await page.locator('#evaluation-tip').textContent();
     assert(tip.includes('reward '+expected.at(-1).mean_reward.toFixed(3))&&tip.includes(`${expected.at(-1).successes}/${expected.at(-1).count}`),'Evaluation tooltip mismatch: '+env);
     checks.push({scope,env_id:env,cohort,points:expected.length,last_reward:expected.at(-1).mean_reward});
    }
   }
   const sources=await page.locator('#batch-source option').evaluateAll(rows=>rows.map(x=>x.value));
   for(const source of sources){
    await page.selectOption('#batch-source',source);
    const expected=snapshot.epochs.filter(e=>e.mode===scope&&e.finalized&&(source==='all'||e.source===source)).sort((a,b)=>a.start-b.start);
    assert(same(await values('batch'),expected.map(e=>e.batches))&&same(await recordIds('batch'),expected.map(e=>e.id)),'Batch series mismatch: '+source);
    if(expected.length){await page.locator('#batch-chart .chart-hit').last().focus();const tip=await page.locator('#batch-tip').textContent();assert(tip.includes(`${expected.at(-1).accepted} accepted`),'Batch tooltip mismatch');}
    batchChecks.push({scope,source,points:expected.length,batches:expected.map(e=>e.batches)});
   }
  }
  await page.locator('[data-scope="test"]').click();
  if(defaultView.environment){await page.selectOption('#environment',defaultView.environment);await page.selectOption('#evaluation',defaultView.cohort);}
  await page.selectOption('#batch-source',defaultView.source);
  const viewports=[];
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1100});await page.waitForTimeout(250);
   const actual=await page.evaluate(()=>({charts:document.querySelectorAll('main svg[role=img]').length,evaluation_points:document.querySelectorAll('#evaluation-chart circle').length,batch_points:document.querySelectorAll('#batch-chart circle').length,overflow:document.documentElement.scrollWidth>innerWidth,cohort:document.querySelector('#evaluation').value,source:document.querySelector('#batch-source').value}));
   assert(actual.charts===2&&!actual.overflow&&actual.cohort===defaultView.cohort&&actual.source===defaultView.source,'Viewport/selection mismatch');
   viewports.push({width,...actual});await page.screenshot({path:path.join(output,'two-charts-'+width+'.png'),fullPage:true});
  }
  const guideResponse=await page.request.get(site+'/llms.txt'),guide=await guideResponse.text();
  assert(guideResponse.ok()&&/text\/plain/.test(guideResponse.headers()['content-type'])&&guide.includes('K=1 positive and L=1 negative')&&guide.includes('everyone scores zero')&&guide.includes('nonpayable')&&!guide.includes('https://dash.affine.io/mailbox'),'Public guide mismatch');
  const controls=[];
  for(const unavailable of [false,true]){
   const control=await browser.newPage({viewport:{width:390,height:900}});
   control.on('pageerror',e=>errors.push(String(e)));
   await control.route('**/network-data.json',route=>route.fulfill({status:unavailable?503:200,contentType:'application/json',body:JSON.stringify({epochs:[],evaluations:[],summary:{updated_at:Date.now()/1000}})}));
   await control.goto(site,{waitUntil:'domcontentloaded'});
   await control.waitForFunction(()=>document.querySelector('#connection')?.textContent!=='Connecting');
   assert(await control.locator('main circle').count()===0&&await control.locator('#evaluation-empty').isVisible()&&await control.locator('#batch-empty').isVisible(),'Fabricated empty/unavailable data');
   controls.push({intercepted_response:unavailable?'unavailable':'empty',points:0,both_empty_states:true});await control.close();
  }
  // Bind otherwise identical measurements to independently changed cohort identities.
  const sample=snapshot.evaluations.find(e=>e.status==='complete'&&e.count===32&&e.env_id==='affine_math');
  if(sample){
   const variants=[sample,...['seed','taskset_hash','output_token_budget','fixed_task_ids'].map((field,i)=>({...sample,run_id:'local-cohort-identity-control-'+i,[field]:field==='fixed_task_ids'?['changed-task']:typeof sample[field]==='number'?sample[field]+1:'changed-taskset'}))];
   const control=await browser.newPage();control.on('pageerror',e=>errors.push(String(e)));
   await control.route('**/network-data.json',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({...snapshot,evaluations:variants})}));
   await control.goto(site,{waitUntil:'domcontentloaded'});await control.waitForFunction(()=>document.querySelector('#evaluation')?.options.length===5);
   const options=await control.locator('#evaluation option').evaluateAll(rows=>rows.map(x=>x.value));
   for(const cohort of options){await control.selectOption('#evaluation',cohort);assert(await control.locator('#evaluation-chart circle').count()===1,'Independent cohorts merged');}
   controls.push({intercepted_response:'cohort-identity-mutations',independent_cohorts:5,each_points:1});await control.close();
  }
  // Local response fixtures exercise automatic/newest selection and exclusion
  // of unfinalized upload activity. They are never published as pilot metrics.
  const mathWindow=snapshot.epochs.filter(e=>e.mode==='test'&&e.finalized&&e.source==='native-math-common').sort((a,b)=>a.start-b.start).at(-1);
  if(mathWindow){
   const fakeNew={...mathWindow,id:'nonpayable-local-series-selection-control',source:'separated-hopper-math',start:mathWindow.start+1};
   const pendingUpload={...fakeNew,id:'nonpayable-local-initial-upload-control',start:fakeNew.start+1,finalized:false};
   let fixture={...snapshot,epochs:[mathWindow,pendingUpload]};
   const control=await browser.newPage();control.on('pageerror',e=>errors.push(String(e)));
   await control.route('**/network-data.json',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(fixture)}));
   await control.goto(site,{waitUntil:'domcontentloaded'});
   await control.waitForFunction(()=>document.querySelector('#connection')?.textContent.includes('updated'));
   assert(await control.locator('#batch-source').inputValue()==='native-math-common','Unfinalized upload selected as current scored series');
   assert(await control.locator('#batch-chart circle').count()===1,'Unfinalized upload created a batch point');
   fixture={...fixture,epochs:[mathWindow,pendingUpload,fakeNew]};
   await control.setViewportSize({width:1200,height:900});await control.waitForTimeout(200);
   // Reload refreshes from the changed local fixture, without waiting15seconds.
   await control.reload({waitUntil:'domcontentloaded'});
   await control.waitForFunction(()=>document.querySelector('#batch-source')?.value==='separated-hopper-math');
   const labels=await control.locator('#batch-source option').evaluateAll(rows=>rows.map(x=>x.textContent));
   assert(labels.some(x=>x.includes('Qwen2.5-Math-7B'))&&labels.some(x=>x.includes('SmolLM2-1.7B')),'Model series labels missing');
   assert(await control.locator('#batch-chart circle').count()===1,'Latest scored MATH series mismatch');
   await control.selectOption('#batch-source','native-math-common');
   await control.setViewportSize({width:390,height:900});await control.waitForTimeout(200);
   assert(await control.locator('#batch-source').inputValue()==='native-math-common','Manual historical series selection lost');
   await control.locator('[data-scope="test"]').click();
   assert(await control.locator('#batch-source').inputValue()==='native-math-common','Manual series selection reset on render');
   controls.push({intercepted_response:'synthetic-source-selection-fixtures',unfinalized_upload_excluded:true,newest_finalized_math_default:true,historical_series_retained:true,manual_selection_preserved:true});
   await control.close();
  }
  assert(errors.length===0,'Browser errors: '+errors.join('; '));
  const out={passed:true,checked_at:Date.now()/1000,site,actual_page_checked:true,public_response_sha256:crypto.createHash('sha256').update(raw).digest('hex'),structure,default_view:defaultView,cohort_checks:checks,batch_checks:batchChecks,viewports,public_guide:true,local_controls:controls,page_errors:errors,chain_transactions:false,quality_improvement_claimed:false};
  fs.writeFileSync(path.join(output,'two-chart-browser-check.json'),JSON.stringify(out,null,2)+'\n');
  console.log(JSON.stringify({passed:true,cohorts:checks.length,batch_series:batchChecks.length,viewports:viewports.map(x=>x.width),local_controls:controls.length}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e.message);process.exit(1)});
