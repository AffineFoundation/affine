import json
import unittest
from ops.export_environment_coverage import export,FLAGS

class PublicCoverageExport(unittest.TestCase):
    def row(self):
        return dict(source='original_environment',module='original_v1',taskset_class='OriginalTaskset',category='native_tools',status='controlled_negative_only',**{k:False for k in FLAGS})
    def test_secrets_and_probe_metadata_are_never_published(self):
        row=self.row();row.update(import_retry={'url':'PRIVATE_CAPABILITY'},blocker='SECRET_TOKEN',password='PRIVATE_PASSWORD')
        result=export(json.dumps([row]).encode());body=json.dumps(result)
        for secret in ('PRIVATE_CAPABILITY','SECRET_TOKEN','PRIVATE_PASSWORD'):self.assertNotIn(secret,body)
        self.assertEqual(result['counts']['training'],0)
    def test_truthy_strings_and_missing_evidence_do_not_become_verified(self):
        for value in ('true',1,None):
            row=self.row();row['training']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'explicit boolean'):export(json.dumps([row]).encode())
    def test_unsafe_identifier_or_duplicate_source_rejected(self):
        row=self.row();row['status']='https://secret.example/?token=private'
        with self.assertRaisesRegex(ValueError,'identifier'):export(json.dumps([row]).encode())
        with self.assertRaisesRegex(ValueError,'duplicate'):export(json.dumps([self.row(),self.row()]).encode())
    def test_snapshot_changes_when_authoritative_bytes_change(self):
        first=self.row();second=self.row();second['remote_proof']=True
        a=export(json.dumps([first]).encode());b=export(json.dumps([second]).encode())
        self.assertNotEqual(a['source_matrix_sha256'],b['source_matrix_sha256'])
        self.assertEqual(b['counts']['remote_proof'],1)

if __name__=='__main__':unittest.main()
