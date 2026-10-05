from test_trainer_cache_lifecycle import TrainerRetention
from subnet.trainer_legacy_cache_catalog import retire_catalog,VERSION
from subnet.backend_jobs import file_map
import json,hashlib

class LegacyCatalog(TrainerRetention):
    def catalog(self):
        (self.oldpath/'config.json').write_bytes(b'{}')
        files=dict(self.cp['files'],**{'config.json':hashlib.sha256(b'{}').hexdigest()})
        cp={'id':file_map(files),'files':files}
        path=self.root/'checkpoints'/cp['id'];self.oldpath.rename(path);self.oldpath=path
        upload=self.authority.sign(dict(role='upload',manifest=self.authority.sign({'checkpoint':cp})))
        receipt={'checkpoint':cp['id'],'operator_independent_hashes':True,'objects':{n:{'sha256':sha}for n,sha in cp['files'].items()}}
        return dict(version=VERSION,durable_trainer_ack=self.authority.sign(self.value),legacy_checkpoints=[dict(path=str(path),checkpoint=cp,original_upload_job=upload,publication_receipt=receipt)])
    def test_explicit_durable_legacy_catalog_retires_only_old_model(self):
        value=self.catalog();result=retire_catalog(self.authority.sign(value),self.authority.id,self.root)
        self.assertEqual(result['status'],'complete');self.assertFalse(self.oldpath.exists());self.assertTrue(self.newpath.exists())
    def test_current_or_non_durable_catalog_cannot_delete(self):
        value=self.catalog();value['legacy_checkpoints'][0]['publication_receipt']['operator_independent_hashes']=False
        with self.assertRaises(ValueError):retire_catalog(self.authority.sign(value),self.authority.id,self.root)
        self.assertTrue(self.oldpath.exists());self.assertTrue(self.newpath.exists())
