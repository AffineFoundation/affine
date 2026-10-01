"""Independent authorities sharing model bytes retain historical signatures."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from nacl.signing import VerifyKey
import base64
from subnet.controller import Controller
from subnet.storage import canonical


class MemoryBucket:
    def __init__(self):self.objects={}
    def upload(self,key,path):self.objects[key]=Path(path).read_bytes()
    def get(self,key):return self.objects[key]
    def json(self,key,value):self.objects[key]=canonical(value)


class DescriptorIsolation(unittest.TestCase):
    def test_shared_content_never_replaces_other_authority(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);model=root/'model';model.mkdir()
            (model/'config.json').write_text('{"model_type":"gpt2"}')
            bucket=MemoryBucket();gateway=SimpleNamespace(url='https://example.invalid')
            first=Controller(bucket,gateway,root/'a');second=Controller(bucket,gateway,root/'b')
            a=first.publish_checkpoint(model);legacy=f"public/checkpoints/{a['id']}/checkpoint.json"
            original=bucket.get(legacy)
            b=second.publish_checkpoint(model)
            self.assertEqual(a['id'],b['id'])
            self.assertNotEqual(a['descriptor_key'],b['descriptor_key'])
            self.assertEqual(bucket.get(legacy),original)
            for owner,checkpoint in ((first,a),(second,b)):
                envelope=json.loads(bucket.get(checkpoint['descriptor_key']))
                self.assertEqual(envelope['signer'],owner.authority.id)
                VerifyKey(bytes.fromhex(owner.authority.id)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature']))
                self.assertEqual(envelope['payload'],{'id':checkpoint['id'],'files':checkpoint['files']})
            own_original=bucket.get(a['descriptor_key'])
            first.gateway=SimpleNamespace(url='https://new-transport.invalid')
            relocated=first.publish_checkpoint(model)
            self.assertNotEqual(relocated['base_url'],a['base_url'])
            self.assertEqual(bucket.get(a['descriptor_key']),own_original)
            self.assertEqual(bucket.get(legacy),original)


if __name__=='__main__':unittest.main()
