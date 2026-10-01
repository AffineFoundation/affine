// Read-only public pilot smoke check; requires Playwright and an installed Chrome.
const fs=require('fs');
const crypto=require('crypto');
const {chromium}=require('playwright');
const path=require('path');
const site=process.argv[2]||'https://affine.io';
const output=process.argv[3]||'state/dashboard';
fs.mkdirSync(output,{recursive:true});
const key=e=>JSON.stringify([e.dataset_id,e.harness,e.environment_version,e.model,e.model_runtime_revision]);
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.AFFINE_CHROME_BIN||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 try{
  const pending=page.waitForResponse(r=>r.url().endsWith('/network-data.json')&&r.status()===200);
  await page.goto(site,{waitUntil:'domcontentloaded',timeout:30000});
  const response=await pending;const raw=await response.body();const snapshot=JSON.parse(raw);
  await page.waitForFunction(()=>document.querySelector('#grid')?.children.length===256&&document.querySelector('#connection')?.textContent==='SN120 / PILOT');
  const environments=await page.locator('#environment option').evaluateAll(rows=>rows.map(x=>x.value));
  if(environments.length<16)throw Error('Missing environment coverage');
  const checks=[];
  for(const env of environments){
   await page.selectOption('#environment',env);await page.selectOption('#metric','reward');
   const cohorts=await page.locator('#evaluation option').evaluateAll(rows=>rows.map(x=>x.value));
   for(const cohort of cohorts){
    await page.selectOption('#evaluation',cohort);
    const expected=snapshot.evaluations.filter(x=>x.env_id===env&&key(x)===cohort&&(x.status===undefined||x.status==='complete')).sort((a,b)=>a.timestamp-b.timestamp);
    const actual=await page.evaluate(()=>({title:document.querySelector('#chart-title').textContent,points:document.querySelectorAll('#chart circle').length,grid:document.querySelector('#grid').children.length,note:document.querySelector('#metric-note').textContent,overflow:document.documentElement.scrollWidth>innerWidth}));
    if(!expected.length||actual.points!==expected.length||actual.grid!==256||actual.overflow||!actual.title.includes(env))throw Error('Cohort chart mismatch: '+env);
    const hits=page.locator('#chart .chart-hit');
    await hits.last().focus();const tip=await page.locator('#chart-tip').textContent();
    if(!tip.includes('reward '+expected.at(-1).mean_reward.toFixed(3)))throw Error('Reward tooltip mismatch: '+env);
    checks.push({env_id:env,series_key:cohort,points:actual.points,last_reward:expected.at(-1).mean_reward});
   }
  }
  const viewports=[];
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:1000});await page.selectOption('#environment','affine_math');
   await page.waitForTimeout(200);
   const row=await page.evaluate(()=>({grid:document.querySelector('#grid').children.length,points:document.querySelectorAll('#chart circle').length,overflow:document.documentElement.scrollWidth>innerWidth}));
   if(row.grid!==256||!row.points||row.overflow)throw Error('Viewport mismatch');
   viewports.push({width,...row});
   await page.screenshot({path:path.join(output,'current-all-cohorts-'+width+'.png'),fullPage:true});
  }
  await page.selectOption('#metric','batches');
  const title=await page.locator('#chart-title').textContent();
  if(title!=='Batches / epoch'||!await page.locator('#environment').isDisabled()||!await page.locator('#evaluation').isDisabled())throw Error('Batch chart scope mismatch');
  const batchPoints=await page.locator('#chart circle').count();
  const completed=snapshot.epochs.filter(x=>x.mode==='test'&&x.finalized);
  if(batchPoints!==completed.length)throw Error('Completed batch timeline mismatch');
  if(errors.length)throw Error('Browser errors');
  const out={passed:true,checked_at:Date.now()/1000,actual_live_page:true,public_response_sha256:crypto.createHash('sha256').update(raw).digest('hex'),environments:environments.length,cohort_checks:checks,viewports,batch_points:batchPoints,page_errors:errors,chain_transactions:false,quality_improvement_claimed:false};
  fs.writeFileSync(path.join(output,'current-all-public-cohorts-browser-check.json'),JSON.stringify(out,null,2)+'\n');
  console.log(JSON.stringify({passed:true,environments:environments.length,cohorts:checks.length,viewports:viewports.map(x=>x.width),batch_points:batchPoints}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e.message);process.exit(1)});
