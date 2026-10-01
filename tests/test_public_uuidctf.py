import base64
import csv
import gzip
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from subnet.public_uuidctf import solve_public_corpus,public_solver_command


class PublicForensicsTests(unittest.TestCase):
    def fixture(self,root):
        ids=[uuid.UUID(int=n) for n in range(1,6)]
        def write(name,text):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path
        write('support/tickets/ticket_091.md','Case: INC-1000\nCustomer: Customer A\n')
        write('crm/accounts.csv','customer,tenant_id\nDecoy,tnt_wrong\nCustomer A,tnt_real\n')
        write('ops/policy/recovery-token.yaml','quorum_size: 5\ndomain_separator: "ctf-shard-v1\\n"\n')
        write('ops/migrations/billing-v3.yaml','window_start_utc: 2026-01-01T00:00:00Z\nwindow_end_utc: 2026-01-01T01:00:00Z\n')
        rows=[dict(case='INC-1000',tenant_id='tnt_real',observed_at=f'2026-01-01T00:0{n+1}:00Z') for n in range(5)]
        first=dict(rows[0],event='quorum_piece',material_uuid=str(ids[0]))
        write('logs/audit/ledger.jsonl',json.dumps(dict(first,tenant_id='tnt_wrong'))+'\n'+json.dumps(first))
        path=write('exports/materialized/recovery_material.csv','')
        with path.open('w',newline='') as stream:
            row=dict(rows[1],record_type='recovery_material',value=ids[1].hex)
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        path=root/'backups/redrive/recovery-redrive.jsonl.gz';path.parent.mkdir(parents=True)
        with gzip.open(path,'wt') as stream:
            stream.write(json.dumps(dict(rows[2],kind='recovery_material',material=base64.b64encode(ids[2].bytes).decode()))+'\n')
        write('notes/escalations/mirror-register.md','target_case = INC-1000\ntenant = tnt_real\nobserved_at = 2026-01-01T00:04:00Z\npurpose = recovery_material\nmirrored_material = '+str(ids[3])[::-1]+'\n')
        write('warehouse/parts/part-0007.jsonl',json.dumps(dict(rows[4],kind='recovery_material',uuid_high64=0,uuid_low64=5)))
        return ids

    def test_five_public_encodings_customer_selection_and_protocol_reducer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ids=self.fixture(root);result=solve_public_corpus(root)
            self.assertEqual(result['source_uuids'],[str(value) for value in ids])
            expected=uuid.UUID(bytes=hashlib.sha256(b'ctf-shard-v1\n'+b''.join(value.bytes for value in ids)).digest()[:16])
            self.assertEqual(result['result_uuid'],str(expected))
            self.assertEqual(len(result['evidence_paths']),5)

    def test_out_of_window_record_does_not_satisfy_quorum(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);path=root/'warehouse/parts/part-0007.jsonl'
            row=json.loads(path.read_text());row['observed_at']='2026-01-02T00:00:00Z';path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'five-shard'):solve_public_corpus(root)

    def test_command_uses_public_container_corpus_not_operator_metadata(self):
        command=public_solver_command()
        self.assertIn("solve_public_corpus('/workspace/corpus')",command)
        self.assertNotIn('original-tasks',command)
        self.assertNotIn('self.data',command)


if __name__=='__main__':unittest.main()
