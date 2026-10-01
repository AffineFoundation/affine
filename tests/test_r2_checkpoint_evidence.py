import hashlib
import io
from types import SimpleNamespace
import unittest

from ops.check_r2_checkpoint import check
from subnet.storage import canonical


class CheckpointStreamTests(unittest.TestCase):
    def fixture(self, data=b'approved weights', length=None):
        files={'model.safetensors':hashlib.sha256(b'approved weights').hexdigest()}
        checkpoint={'files':files,'id':hashlib.sha256(canonical(files)).hexdigest()}
        body=io.BytesIO(data);calls=[]
        def get(**kwargs):
            calls.append(kwargs)
            return {'Body':body,'ContentLength':len(data) if length is None else length}
        bucket=SimpleNamespace(name='isolated-mock',client=SimpleNamespace(get_object=get))
        return bucket,checkpoint,body,calls

    def test_stream_hash_checks_complete_published_bytes_and_closes(self):
        bucket,checkpoint,body,calls=self.fixture()
        result=check(bucket,checkpoint,chunk_size=3)
        self.assertEqual(result['files']['model.safetensors']['bytes'],16)
        self.assertTrue(body.closed)
        self.assertEqual(calls[0]['Key'],f"public/checkpoints/{checkpoint['id']}/model.safetensors")
        self.assertFalse(result['model_weights_written_to_operator_disk'])

    def test_corrupted_or_truncated_body_is_not_verified(self):
        for data,length,message in ((b'corrupt weights!',None,'bytes changed'),
                                    (b'approved',16,'truncated')):
            with self.subTest(message=message):
                bucket,checkpoint,body,calls=self.fixture(data,length)
                with self.assertRaisesRegex(ValueError,message):check(bucket,checkpoint,chunk_size=3)
                self.assertTrue(body.closed)

    def test_unapproved_filemap_is_rejected_before_network(self):
        bucket,checkpoint,body,calls=self.fixture();checkpoint['files']['model.safetensors']='0'*64
        with self.assertRaisesRegex(ValueError,'identity'):check(bucket,checkpoint)
        self.assertFalse(calls)


if __name__=='__main__':unittest.main()
