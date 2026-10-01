"""Public-file forensic control for the original standard UUIDCTF corpus."""
import base64
import csv
import gzip
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path


def solve_public_corpus(corpus):
    corpus=Path(corpus)
    ticket=(corpus/'support/tickets/ticket_091.md').read_text()
    case=re.search(r'^Case: (.+)$',ticket,re.M).group(1)
    customer=re.search(r'^Customer: (.+)$',ticket,re.M).group(1)
    with (corpus/'crm/accounts.csv').open() as stream:
        accounts=list(csv.DictReader(stream))
    matched=[row for row in accounts if row['customer']==customer]
    if len(matched)!=1:
        raise ValueError('unique public customer account required')
    tenant=matched[0]['tenant_id']
    policy=(corpus/'ops/policy/recovery-token.yaml').read_text()
    if re.search(r'^quorum_size: (\d+)$',policy,re.M).group(1)!='5':
        raise ValueError('original five-shard protocol required')
    domain=json.loads(re.search(r'^domain_separator: (.+)$',policy,re.M).group(1))
    migration=(corpus/'ops/migrations/billing-v3.yaml').read_text()
    def timestamp(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
    start=timestamp(re.search(r'^window_start_utc: (.+)$',migration,re.M).group(1))
    end=timestamp(re.search(r'^window_end_utc: (.+)$',migration,re.M).group(1))
    records=[]
    def accept(row,kind,key,path,decoder):
        if (row.get('case')==case and row.get('tenant_id')==tenant and row.get(key)==kind
                and start<=timestamp(row['observed_at'])<=end):
            records.append((timestamp(row['observed_at']),str(decoder(row)),str(path)))
    for path in sorted((corpus/'logs/audit').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            accept(json.loads(line),'quorum_piece','event',path,lambda r:uuid.UUID(r['material_uuid']))
    path=corpus/'exports/materialized/recovery_material.csv'
    with path.open() as stream:
        for row in csv.DictReader(stream):
            accept(row,'recovery_material','record_type',path,lambda r:uuid.UUID(r['value']))
    path=corpus/'backups/redrive/recovery-redrive.jsonl.gz'
    with gzip.open(path,'rt') as stream:
        for line in stream:
            accept(json.loads(line),'recovery_material','kind',path,
                   lambda r:uuid.UUID(bytes=base64.b64decode(r['material'],validate=True)))
    path=corpus/'notes/escalations/mirror-register.md'
    fields=dict(re.findall(r'^([a-z_]+) = (.+)$',path.read_text(),re.M))
    accept(dict(case=fields['target_case'],tenant_id=fields['tenant'],observed_at=fields['observed_at'],
                purpose=fields['purpose'],material=fields['mirrored_material']),
           'recovery_material','purpose',path,lambda r:uuid.UUID(r['material'][::-1]))
    path=corpus/'warehouse/parts/part-0007.jsonl'
    for line in path.read_text().splitlines():
        accept(json.loads(line),'recovery_material','kind',path,lambda r:uuid.UUID(
            bytes=int(r['uuid_high64']).to_bytes(8,'big')+int(r['uuid_low64']).to_bytes(8,'big')))
    if len(records)!=5 or len({row[1] for row in records})!=5 or len({row[0] for row in records})!=5:
        raise ValueError('unique five-shard public evidence required')
    records.sort()
    source_uuids=[row[1] for row in records]
    digest=hashlib.sha256(domain.encode()+b''.join(uuid.UUID(value).bytes for value in source_uuids)).digest()
    return dict(result_uuid=str(uuid.UUID(bytes=digest[:16])),source_uuids=source_uuids,
                evidence_paths=[row[2] for row in records])


def public_solver_command():
    """Embed this reviewed stdlib solver; it reads no host task/answer metadata."""
    import inspect
    source=inspect.getsource(solve_public_corpus)
    program='import base64,csv,gzip,hashlib,json,re,uuid\nfrom datetime import datetime\nfrom pathlib import Path\n'+source
    program+="\nanswer=solve_public_corpus('/workspace/corpus')\nPath('/workspace/answer.json').write_text(json.dumps(answer))\nprint(json.dumps(answer))\n"
    return "python - <<'AFFINE_PUBLIC_UUID_SOLVER'\n"+program+"AFFINE_PUBLIC_UUID_SOLVER"
