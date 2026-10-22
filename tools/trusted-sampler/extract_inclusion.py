#!/usr/bin/env python3
"""Read-only: per training epoch, which miners (UIDs) had groups in the trained batch.

Reads state/<run>/epochs/<e>/{TRAINING-JOB,AUDIT-POPULATION,FINALIZED}.json. The trained
document keys embed the miner public key; public key -> UID comes from the epoch's signed
registrations document. Output (aggregate counts per UID only; no hotkeys, no group data):
  [{epoch, optimizer_step, input_checkpoint, output_checkpoint, max_tasks, trained_total,
    trained_by_uid{uid:n}, submitted_by_uid{uid:n}, eligible_total}]
"""
import json, re, sys, glob
from pathlib import Path

state = Path(sys.argv[1]); out = Path(sys.argv[2]); lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rows = []
for d in sorted(glob.glob(str(state / 'epochs/*')), key=lambda p: int(Path(p).name)):
    e = int(Path(d).name)
    if e < lo:
        continue
    d = Path(d)
    if not ((d / 'TRAINING-JOB.json').exists() and (d / 'FINALIZED.json').exists() and (d / 'AUDIT-POPULATION.json').exists()):
        continue
    job = json.loads((d / 'TRAINING-JOB.json').read_bytes())['payload']
    fin = json.loads((d / 'FINALIZED.json').read_bytes()); fin = fin.get('payload', fin)
    pop = json.loads((d / 'AUDIT-POPULATION.json').read_bytes()); pop = pop.get('payload', pop)
    regs = pop['registrations_document']['payload']['registrations']
    uid_of = {v['public_key']: v['uid'] for v in regs.values()}
    tr = job['training']; docs = tr['documents']
    by = {}
    for k, v in docs.items():
        m = re.search(r'complete-group-snapshots/([0-9a-f]{64})/', v['url'])
        u = uid_of.get(m.group(1)) if m else None
        by[str(u)] = by.get(str(u), 0) + 1
    sub = {}
    for r in pop['records']:
        u = str(uid_of.get(r['miner'])); sub[u] = sub.get(u, 0) + 1
    res = fin['result']
    rows.append(dict(epoch=e, optimizer_step=res.get('optimizer_step'), status=res.get('status'),
                     input_checkpoint=fin.get('input_checkpoint'),
                     output_checkpoint=(res.get('checkpoint') or {}).get('id'),
                     max_tasks=tr['config']['settings'].get('max_tasks'), trained_total=len(docs),
                     trained_by_uid=by, submitted_by_uid=sub, eligible_total=len(pop.get('eligible_evidence_ids', [])),
                     registered_uids=len(uid_of)))
    print(e, rows[-1]['optimizer_step'], len(docs), len(by), rows[-1]['max_tasks'], file=sys.stderr)
out.write_text(json.dumps(rows))
