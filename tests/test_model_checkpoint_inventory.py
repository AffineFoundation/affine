"""Full checkpoint integrity remains required with bounded parallel readers."""
import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from subnet.model import model_files,file_hash

class CheckpointInventoryTests(unittest.TestCase):
    def test_large_sharded_checkpoint_matches_complete_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expected={}
            for i in range(5):
                data=(bytes([i])*1024+bytes(range(256)))*2049
                name=f'model-{i:05d}.safetensors';(root/name).write_bytes(data)
                expected[name]=hashlib.sha256(data).hexdigest()
            (root/'config.json').write_bytes(b'{"architecture":"test"}')
            expected['config.json']=hashlib.sha256((root/'config.json').read_bytes()).hexdigest()
            self.assertEqual(model_files(root),expected)

    def test_weight_mutation_is_detected_on_next_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'model.safetensors';p.write_bytes(b'original weights')
            before=model_files(root);p.write_bytes(b'modified weights')
            self.assertNotEqual(model_files(root),before)
            self.assertEqual(model_files(root)[p.name],hashlib.sha256(p.read_bytes()).hexdigest())

    def test_unexpected_model_relevant_file_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_bytes(b'{}')
            before=model_files(root);(root/'unexpected.pt').write_bytes(b'unapproved weights')
            self.assertEqual(set(model_files(root))-set(before),{'unexpected.pt'})

    def test_metadata_assets_and_directories_preserve_existing_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expected={}
            for ext in ['json','safetensors','txt','model','jinja','bin','pt','tiktoken']:
                p=root/('asset.'+ext);p.write_bytes(ext.encode());expected[p.name]=hashlib.sha256(ext.encode()).hexdigest()
            (root/'report.log').write_bytes(b'not a model file');(root/'nested.safetensors').mkdir()
            self.assertEqual(model_files(root),expected)

    def test_disappearing_weight_fails_entire_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_bytes(b'{}');p=root/'model.safetensors';p.write_bytes(b'weights')
            def read(path):
                if path.name==p.name:path.unlink()
                return file_hash(path)
            with patch('subnet.model.file_hash',side_effect=read):
                with self.assertRaises(FileNotFoundError):model_files(root)

    def test_empty_inventory_and_single_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.assertEqual(model_files(root),{})
            p=root/'config.json';p.write_bytes(b'{}')
            self.assertEqual(model_files(root),{'config.json':hashlib.sha256(b'{}').hexdigest()})

if __name__=='__main__':unittest.main()
