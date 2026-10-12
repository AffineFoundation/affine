const assert = require('node:assert/strict');
const test = require('node:test');
const api = require('./dashboard/public/performance-series.js');
const epochs = [1,2,3,4].map(id => ({id:'epoch-'+id}));
function row(n, split='heldout', extra={}) {
  return {epoch_id:'epoch-'+n,env_id:'affine_math',timestamp:n,status:'complete',
    evaluation_split:split,generation_origin:'fresh-operator-generation',
    comparability_passed:true,
    count:128,requested_count:128,completed_count:128,successes:80,
    error_count:0,mean_reward:80/128,dataset_id:'fixed128',seed:'fixed-seeds',
    fixed_initial_trained:{count:55,solved:33,wrong:20,unresolved:2,errors:0,
      mask_sha256:'a'.repeat(64),checkpoint:'b'.repeat(64),optimizer_step:67},
    training_run_id:'current-run',runtime_profile_sha256:'same-runtime',...extra};
}
test('same-epoch TRAIN never replaces heldout and absent TRAIN is empty',()=>{
  const data={evaluations:[row(1)],training_evaluations:[row(1,'train',{successes:90,mean_reward:90/128})]};
  assert.equal(api.history(data,epochs,'heldout')[0].successes,80);
  assert.equal(api.history(data,epochs,'train')[0].successes,90);
  assert.deepEqual(api.history({evaluations:data.evaluations},epochs,'train'),[]);
});
test('partial/error points are gaps, never zero and never connected through',()=>{
  const data={training_evaluations:[row(1,'train'),row(2,'train',{status:'partial',error_count:1,mean_reward:null}),row(3,'train')]};
  const values=api.history(data,epochs,'train');assert.equal(values.length,2);
  assert.equal(values[1].measurement_gap,true);
  assert.deepEqual(api.segments(values,[1,3]),[[1],[3]]);
});
test('cohort runtime run and output-cap changes split lines',()=>{
  for(const field of ['dataset_id','runtime_profile_sha256','training_run_id','output_token_budget']){
    const values=[row(1,'train'),row(2,'train',{[field]:'different'})];
    assert.deepEqual(api.segments(values,[1,2]),[[1],[2]]);
  }
});
test('comparable observations connect and no foreign epochs leak in',()=>{
  const values=api.history({training_evaluations:[row(1,'train'),row(2,'train'),row(99,'train')]},epochs,'train');
  assert.deepEqual(api.segments(values,[1,2]),[[1,2]]);
});
test('submitted balanced labels and malformed denominators cannot become TRAIN measurements',()=>{
  for(const extra of [{generation_origin:'submitted-miner-labels'},{comparability_passed:false},{completed_count:127},{mean_reward:null},{error_count:1},{mean_reward:.5}])
    assert.equal(api.measured(row(1,'train',extra)),false);
});

test('same-epoch original-base and current TRAIN points are both retained in checkpoint order',()=>{
  const baseline=row(1,'train',{checkpoint:'base',phase:'baseline',timestamp:9,training_steps:0});
  const current=row(1,'train',{checkpoint:'current',phase:'latest',timestamp:8,training_steps:67});
  const values=api.history({training_evaluations:[current,baseline]},epochs,'train');
  assert.deepEqual(values.map(r=>r.checkpoint),['base','current']);
  assert.deepEqual(api.segments(values,[1,2]),[[1,2]]);
  assert.equal(api.history({evaluations:[row(1),row(1,'heldout',{timestamp:9})]},epochs,'heldout').length,1);
});

test('TRAIN fit uses fixed55 while stored full128 values remain intact',()=>{
  const original=row(1,'train');
  const rows=api.history({training_evaluations:[original]},epochs,'train');
  assert.equal(rows[0].count,128);assert.equal(rows[0].successes,80);
  assert.deepEqual(api.metric(rows[0]),{count:55,successes:33,mean_reward:33/55});
  assert.equal(original.count,128);assert.equal(original.mean_reward,80/128);
  assert.deepEqual(api.metric(row(1)),{count:128,successes:80,mean_reward:80/128});
});
test('missing invalid or errored fixed subset is unavailable without full-pool fallback',()=>{
  const base=row(1,'train');
  const invalid=[null,{}, {...base.fixed_initial_trained,count:0},
    {...base.fixed_initial_trained,mask_sha256:'not-a-hash'},
    {...base.fixed_initial_trained,solved:34},
    {...base.fixed_initial_trained,wrong:19,errors:1},
    {...base.fixed_initial_trained,optimizer_step:-1}];
  for(const fixed of invalid){
    const bad={...base,fixed_initial_trained:fixed};
    assert.equal(api.metric(bad),null);assert.equal(api.measured(bad),false);
    const values=api.history({training_evaluations:[base,{...bad,epoch_id:'epoch-2'},row(3,'train')]},epochs,'train');
    assert.deepEqual(api.segments(values,[1,3]),[[1],[3]]);
  }
});
test('fixed mask checkpoint step and denominator are separate longitudinal series',()=>{
  const base=row(1,'train');
  for(const change of [{mask_sha256:'c'.repeat(64)}, {checkpoint:'c'.repeat(64)},
    {optimizer_step:68}, {count:54,wrong:19}]){
    const next=row(2,'train',{fixed_initial_trained:{...base.fixed_initial_trained,...change}});
    assert.equal(api.measured(next),true);
    const rows=api.history({training_evaluations:[base,next]},epochs,'train');
    assert.deepEqual(api.segments(rows,[1,2]),[[1],[2]]);
  }
});
test('changing current exposure groups cannot change the fixed metric or curve key',()=>{
  const before=row(1,'train',{actual_trained_task_count:55});
  const after=row(2,'train',{actual_trained_task_count:90});
  assert.deepEqual(api.metric(before),api.metric(after));
  assert.equal(api.cohortKey(before),api.cohortKey(after));
});
test('all128 completion and errors remain mandatory even if fixed55 is clean',()=>{
  for(const change of [{error_count:1},{completed_count:127},{comparability_passed:false},{status:'partial'}]){
    const value=row(1,'train',change);
    assert.notEqual(api.metric(value),null);
    assert.equal(api.history({training_evaluations:[value]},epochs,'train').length,0);
  }
});
