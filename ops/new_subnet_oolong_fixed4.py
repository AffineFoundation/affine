"""Exact pinned Oolong tasks; candidate policy reads public context metadata only."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from subnet.environments import build_spec, _taskset, _source_hash, EnvironmentSpec

REVISION = 'f0d59eaf0febf130664cfceb710436c8e3216b2b'
INDICES = [215, 214, 216, 217]

def main():
    spec = build_spec('affine_oolong', {'taskset': {'context_len': 16384, 'split': 'test'}},
                      num_samples=4, max_turns=2, max_output_tokens=512)
    _taskset(spec)
    from affine_oolong_v1.taskset import SYSTEM, task_name
    from oolong_synth_v1.taskset import OolongSynthTask, OolongSynthData, OolongSynthTaskConfig, INSTRUCTIONS, WORKDIR
    cache = Path.home()/'.cache/huggingface/hub/datasets--oolongbench--oolong-synth/snapshots'/REVISION/'data'
    selected = {}; offset = 0; origins = []
    for path in sorted(cache.glob('test-*.parquet')):
        count = pq.ParquetFile(path).metadata.num_rows
        wanted = [i for i in INDICES if offset <= i < offset + count]
        if wanted:
            rows = pq.read_table(path).to_pylist()
            for i in wanted:
                selected[i] = rows[i-offset]
            origins.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'first_global_row': offset})
        offset += count
        if len(selected) == len(INDICES): break
    if len(selected) != 4: raise RuntimeError('missing original rows')
    tasks = []; public = []
    for ordinal, index in enumerate(INDICES):
        row = selected[index]
        if row['context_len'] != 16384: raise RuntimeError('wrong original context bucket')
        kind = row.get('answer_type', '')
        if kind not in ('ANSWER_TYPE.NUMERIC', 'ANSWER_TYPE.DATE'): kind = ''
        task = OolongSynthTask(OolongSynthData(idx=index, name=task_name(16384,index,'test'),
            system_prompt=SYSTEM, prompt=row['question']+'\n\n'+INSTRUCTIONS,
            question=row['question'], answer=row['answer'], context=row['context_window_text'],
            answer_type=kind, workdir=WORKDIR), OolongSynthTaskConfig())
        tasks.append({'task_class': type(task).__name__, 'data': task.data.model_dump(mode='json'), 'task_config': task.config.model_dump(mode='json')})
        public.append({'ordinal': ordinal, 'original_index': index, 'name': task.data.name,
                       'question': row['question'], 'context_sha256': hashlib.sha256(row['context_window_text'].encode()).hexdigest(),
                       'context_bytes':len(row['context_window_text'].encode())})
    dest = Path('state/original-task-snapshots/oolong-date-fixed4.tasks.json')
    if dest.exists(): raise RuntimeError('refuse to replace frozen task snapshot')
    dest.write_text(json.dumps(tasks,sort_keys=True,separators=(',',':'))+'\n'); dest.chmod(0o400)
    spec.config['task_snapshot'] = str(dest)
    spec = EnvironmentSpec.from_dict({**spec.to_dict(), 'source_hash': _source_hash(spec)})
    dest.with_name('oolong-date-fixed4.spec.json').write_text(json.dumps(spec.to_dict(),indent=2)+'\n')
    def wire(command):
        return json.dumps({'tool_call': {'name': 'bash', 'arguments': {'command': command}}},separators=(',',':'))
    script = "import re,collections; text=open('/workspace/context.txt').read(); c=collections.Counter(re.findall(r'^Date: (.*?) \\|\\| User:',text,re.M)); n=sum(v==2 for v in c.values()); open('/workspace/answer.txt','w').write(str(n%s))"
    candidates = [wire("python -c " + __import__('shlex').quote(script % delta)) for delta in ('','+1')]
    harness = {'version':'plain-transcript-v1','policy':'candidates','max_output_tokens':512,'temperature':4.0,'top_p':1.0,
       'candidates':[wire('head -n 5 /workspace/context.txt'), wire('head -n 6 /workspace/context.txt')],
       'turn_overrides':{'1':{'policy':'candidates','candidates':candidates}}}
    out = Path('state/multi-environment/oolong-date-harness.json'); out.write_text(json.dumps(harness,indent=2)+'\n')
    report = {'source':'affine_oolong','revision':REVISION,'parquets':origins,'source_hash':spec.source_hash,
              'snapshot_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'tasks':public,
              'training_ordinals':[0], 'heldout_ordinals':[2,3], 'policy':'public date-frequency Counter vs Counter+1; no gold access',
              'expected_negative_reward':0.75, 'expected_positive_reward':1.0}
    Path('state/multi-environment/oolong-date-fixed4-public.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__': main()
