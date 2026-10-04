import hashlib
import io
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from ops.retain_verifier_downloads import verified_archive,run_cycle


class ArchiveReadback(unittest.TestCase):
    def check(self, data=b'approved', size=8, reported_size=8, expected=None):
        body=io.BytesIO(data)
        client=SimpleNamespace(get_object=lambda **kw:dict(Body=body,ContentLength=reported_size,ETag='opaque'))
        bucket=SimpleNamespace(client=client,name='private')
        plan=dict(archive_key='public/immutable.zip',size=size,sha256=expected or hashlib.sha256(b'approved').hexdigest())
        return body,bucket,plan

    def test_complete_original_archive_read_before_deletion_grant(self):
        body,bucket,plan=self.check()
        result=verified_archive(bucket,plan)
        self.assertTrue(result['archive_verified']);self.assertTrue(body.closed)
        self.assertEqual(result['archive_read_bytes'],8)

    def test_archive_changes_never_create_deletion_grant(self):
        for args in (dict(reported_size=9),dict(data=b'altered!'),dict(data=b'short'),dict(data=b'too-long!')):
            body,bucket,plan=self.check(**args)
            with self.assertRaises(ValueError):verified_archive(bucket,plan)
            self.assertTrue(body.closed)

    def test_multiple_stream_chunks_are_all_hashed(self):
        data=b'x'*(2*1024*1024+9)
        body,bucket,plan=self.check(data=data,size=len(data),reported_size=len(data),expected=hashlib.sha256(data).hexdigest())
        result=verified_archive(bucket,plan)
        self.assertEqual(result['archive_read_bytes'],len(data));self.assertTrue(body.closed)

    def test_signed_unsharded_checkpoint_uses_full_readback_and_bounded_model_size(self):
        import base64
        from nacl.signing import SigningKey
        from subnet.storage import canonical
        from ops.retain_checkpoint_caches import archived_files
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        files={'config.json':'a'*64,'model.safetensors':'b'*64}
        cp=hashlib.sha256(canonical(files)).hexdigest()
        descriptor={'id':cp,'files':files}
        envelope={'payload':descriptor,'signer':authority,
                  'signature':base64.b64encode(key.sign(canonical(descriptor)).signature).decode()}
        sizes={'config.json':2,'model.safetensors':15_231_272_152}
        client=SimpleNamespace(head_object=lambda **kw:dict(ContentLength=sizes[kw['Key'].rsplit('/',1)[1]]))
        bucket=SimpleNamespace(client=client,name='private',get=lambda _:canonical(envelope))
        with patch('ops.retain_checkpoint_caches.verified_archive',side_effect=lambda _,p:dict(p,archive_verified=True)) as check:
            result=archived_files(bucket,cp,authority)
            self.assertEqual(result['model.safetensors']['size'],15_231_272_152)
            self.assertEqual(check.call_count,2)
            self.assertEqual({c.args[1]['sha256']for c in check.call_args_list},set(files.values()))
        for name,invalid in [('model.safetensors',32*1024**3+1),('config.json',5*1024**3+1),
                             ('model.safetensors',0),('model.safetensors',True)]:
            old=sizes[name];sizes[name]=invalid
            with self.subTest(name=name,size=invalid),patch('ops.retain_checkpoint_caches.verified_archive',side_effect=lambda _,p:p),self.assertRaisesRegex(ValueError,'bounded archived checkpoint size'):
                archived_files(bucket,cp,authority)
            sizes[name]=old


class RetentionCycleControls(unittest.TestCase):
    def fixture(self, root):
        import sqlite3
        state=root/'state';(state/'roles').mkdir(parents=True)
        (state/'controller.json').write_text(json.dumps({'active':None}))
        db=sqlite3.connect(state/'roles/verifier-queue.sqlite3');db.execute('create table jobs(status text,role text)');db.close()
        config=root/'config.json';config.write_text(json.dumps(dict(state=str(state),bucket={},remote=dict(roles={'verify':[dict(worker_identity='approved',workspace='/root/work')]}))));config.chmod(0o600)
        writer=root/'writer.json';writer.write_text('{}')
        return config,writer

    def test_completed_epoch_boundary_and_empty_queue_are_safe_noops(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,writer=self.fixture(root)
            with patch('ops.retain_verifier_downloads.signed',return_value={'verifier_identities':['approved']}),patch('ops.retain_verifier_downloads.approved_source_members',return_value={}),patch('ops.retain_verifier_downloads.Bucket') as bucket,patch('ops.retain_verifier_downloads.subprocess.run') as remote:
                result=run_cycle(config,writer,'a'*64,root/'receipts')
                self.assertEqual(result['removed_files'],0);self.assertEqual(result['failures'],0)
                remote.assert_not_called()

    def test_roster_mismatch_stops_before_any_storage_or_remote_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,writer=self.fixture(root)
            with patch('ops.retain_verifier_downloads.signed',return_value={'verifier_identities':['another']}),patch('ops.retain_verifier_downloads.approved_source_members',return_value={}),patch('ops.retain_verifier_downloads.Bucket') as bucket,patch('ops.retain_verifier_downloads.subprocess.run') as remote:
                with self.assertRaisesRegex(ValueError,'roster'):run_cycle(config,writer,'a'*64,root/'receipts')
                bucket.assert_not_called();remote.assert_not_called()

    def test_world_readable_config_is_refused_before_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,writer=self.fixture(root);config.chmod(0o644)
            with patch('ops.retain_verifier_downloads.signed') as signed:
                with self.assertRaisesRegex(ValueError,'private'):run_cycle(config,writer,'a'*64,root/'receipts')
                signed.assert_not_called()

if __name__=='__main__':unittest.main()
