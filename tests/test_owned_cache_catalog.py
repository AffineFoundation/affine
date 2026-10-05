import base64
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from nacl.signing import SigningKey
from ops.adopt_owned_verifier_cache_catalog import apply
from subnet.storage import canonical

class CatalogTests(unittest.TestCase):
    def test_authenticated_quiescent_catalog_recovers_disk_without_manual_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'checkpoints'/'old';old.mkdir(parents=True);(old/'model.safetensors').write_bytes(b'weights')
            report=root/'jobs'/'historic';report.mkdir(parents=True);(report/'report.json').write_text('preserve')
            key=SigningKey.generate();payload=dict(revision='owned-verifier-cache-catalog-v1',quiescent_readers_confirmed=True,created_at=10,expires_at=20,
                roots=[dict(root=str(root),keep=[],checkpoints=[dict(id='old',files={'model.safetensors':hashlib.sha256(b'weights').hexdigest()},durability_ack={'r2':'previously authenticated map'})])])
            def envelope():return dict(payload=payload,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
            with self.assertRaises(ValueError):apply(envelope(),key.verify_key.encode().hex(),now=21)
            payload['quiescent_readers_confirmed']=False
            with self.assertRaises(ValueError):apply(envelope(),key.verify_key.encode().hex(),now=15)
            payload['quiescent_readers_confirmed']=True
            result=apply(envelope(),key.verify_key.encode().hex(),now=15)
            self.assertEqual(result[0]['removed'],['old']);self.assertTrue((report/'report.json').exists());self.assertFalse(old.exists())
            with self.assertRaises(Exception):apply(envelope(),SigningKey.generate().verify_key.encode().hex(),now=15)
