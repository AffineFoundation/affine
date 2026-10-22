#!/usr/bin/env python3
"""Read-only: per epoch, every admitted group -> (uid, commitment, task index, trained?, outcomes if trained).
Sources: AUDIT-POPULATION (records + registrations), coordinator/admission.sqlite (claims: document sha -> index),
TRAINING-JOB (trained document set), roles.sqlite3 train report (per-rollout outcomes). Output stays on our hosts."""
import json, re, sqlite3, sys
from pathlib import Path
state = Path(sys.argv[1]); out = {}
roles = sqlite3.connect(f"file:{state/'roles.sqlite3'}?mode=ro", uri=True)
for e in map(int, sys.argv[2].split(',')):
    d = state / 'epochs' / str(e)
    pop = json.loads((d / 'AUDIT-POPULATION.json').read_bytes()); pop = pop.get('payload', pop)
    uid_of = {v['public_key']: v['uid'] for v in pop['registrations_document']['payload']['registrations'].values()}
    job = json.loads((d / 'TRAINING-JOB.json').read_bytes()); job = job.get('payload', job)
    trained_docs = set(job['training']['documents'])
    adm = sqlite3.connect(f"file:{d/'coordinator/admission.sqlite'}?mode=ro", uri=True)
    doc_of = {}
    for miner, slot, claim in adm.execute('select miner, slot, claim from groups'):
        c = json.loads(claim); doc_of[(miner, c['batch_sha256'], int(c['slot']), c['index'])] = c['document_sha256']
    rep = json.loads(roles.execute('select report from jobs where id=?', (job['job_id'],)).fetchone()[0])['training']
    outc = {}
    for i, ds in enumerate(rep['input_documents_sha256']):
        tr = rep['trajectories'][8 * i:8 * i + 8]
        assert all(t['task_id'] == rep['admitted_groups'][i]['task_id'] for t in tr)
        outc[ds] = [t['outcome'] for t in tr]
    groups = []; miss = 0
    for r in pop['records']:
        cands = [v for (m, b, s, ix), v in doc_of.items() if m == r['miner'] and b == r['batch_sha256'] and ix == r['index']]
        ds = cands[0] if cands else None; miss += ds is None
        groups.append(dict(uid=uid_of.get(r['miner']), commitment=r['commitment_sha256'][:16], index=r['index'],
                           trained=ds in trained_docs, outcomes=outc.get(ds)))
    out[e] = dict(base=job['training'].get('input', {}).get('checkpoint') or rep['input_checkpoint'], output=rep['output_checkpoint'],
                  groups=groups)
    print(e, out[e]['base'][:8], len(groups), 'trained', sum(g['trained'] for g in groups), 'with outcomes', sum(g['outcomes'] is not None for g in groups), 'nodoc', miss, file=sys.stderr)
Path(sys.argv[3]).write_text(json.dumps(out))
