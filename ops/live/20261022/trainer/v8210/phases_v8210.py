"""Read-only: per-job trainer phase timings on the pod (UTC). usage: python3 phases_v8210.py <job-id-prefix>..."""
import json, sys, time, os, glob
from pathlib import Path
W = Path('/workspace/affine-mimo-trainer-prod'); H = W/'trainer-state/affine-mimo9b-grpo-20261013/history'
t = lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if x else None
for prefix in sys.argv[1:]:
    jobs = sorted(glob.glob(str(W/'workspace/backend/jobs'/(prefix + '*'))))
    for job in jobs:
        ev = Path(job)/'training-events.jsonl'
        if not ev.exists(): print(json.dumps(dict(job=Path(job).name[:22], events=None))); continue
        rows = [json.loads(l) for l in ev.open() if l.strip()]
        loss = [r['observed_at'] for r in rows if r.get('stage') == 'trajectory-loss']
        ret = [r for r in rows if r.get('stage') == 'optimizer-retention']
        out = dict(job=Path(job).name[:22], first_event=t(rows[0]['observed_at']) if rows else None,
                   gpu_training_done=t(max(loss)) if loss else None, retention=t(ret[-1]['observed_at']) if ret else None)
        # match the history generation by the retained checkpoint of this job's own output
        gen = None
        for d in H.iterdir():
            tx = d/'TRANSACTION.json'
            if tx.exists():
                v = json.loads(tx.read_bytes())
                if ret and v.get('output_checkpoint') == ret[-1].get('retained_checkpoint'): gen = d
        if gen is None and loss:
            cands = [d for d in H.iterdir() if (d/'STAGED.json').exists() and (d/'STAGED.json').stat().st_mtime > max(loss)]
            gen = min(cands, key=lambda d: (d/'STAGED.json').stat().st_mtime) if cands else None
        if gen is not None:
            m = sorted(p.stat().st_mtime for p in (gen/'model').iterdir()) if (gen/'model').is_dir() else []
            o = sorted(p.stat().st_mtime for p in (gen/'optimizer').iterdir()) if (gen/'optimizer').is_dir() else []
            out.update(generation=gen.name[:15], model_files_first=t(m[0]) if m else None, model_files_last=t(m[-1]) if m else None,
                       optimizer_files=len(o), optimizer_first=t(o[0]) if o else None, optimizer_last=t(o[-1]) if o else None,
                       staged_json=t((gen/'STAGED.json').stat().st_mtime) if (gen/'STAGED.json').exists() else None,
                       transaction_json=t((gen/'TRANSACTION.json').stat().st_mtime))
        print(json.dumps(out))
