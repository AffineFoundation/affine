"""Export a public, strictly allowlisted snapshot of operator coverage evidence."""
import argparse
import hashlib
import json
import re
from pathlib import Path

FLAGS=('import','local_reset','original_reward','native_tool_execution','remote_proof','remote_replay','training')
TEXT=('source','module','taskset_class','category','status')

def export(raw):
    rows=json.loads(raw)
    if not isinstance(rows,list) or not rows:raise ValueError('nonempty source matrix required')
    public=[];seen=set()
    for row in rows:
        result={}
        for key in TEXT:
            value=row.get(key)
            if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',value):raise ValueError('matrix identifier '+key)
            result[key]=value
        if result['source'] in seen:raise ValueError('duplicate source')
        seen.add(result['source'])
        for key in FLAGS:
            value=row.get(key)
            if type(value) is not bool:raise ValueError('explicit boolean required '+key)
            result[key]=value
        # Raw exceptions, credentials, artifact URLs and nested probe payloads
        # are intentionally outside the public schema. Status is the blocker code.
        public.append(result)
    public.sort(key=lambda r:r['source'])
    return dict(schema=1,source_matrix_sha256=hashlib.sha256(raw).hexdigest(),
        source_count=len(public),counts={key:sum(row[key] for row in public) for key in FLAGS},
        scope='Recorded source-level operator evidence; each row is a milestone, not full production support or proof of performance improvement.',
        status_is_blocker_or_milestone=True,independently_reexecuted_by_exporter=False,
        rows=public)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,default=Path('state/multi-environment/environment-execution-matrix.json'))
    p.add_argument('--output',type=Path,default=Path('docs/environment-coverage.json'))
    args=p.parse_args();value=export(args.matrix.read_bytes())
    args.output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(sources=value['source_count'],counts=value['counts'])))

if __name__=='__main__':main()
