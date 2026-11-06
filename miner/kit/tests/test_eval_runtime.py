"""CPU-only published probe provenance and tamper tests."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('verify_eval_runtime',ROOT/'kit/verify_eval_runtime.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)


class EvalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        pins=json.loads((ROOT/'validator/probe/runtime/PINS.json').read_text())
        for rel in list(pins['files'])+['validator/probe/runtime/PINS.json','data/gate-pool-v2.jsonl']:
            target=self.root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)

    def test_published_runtime_and_every_eligible_task_index(self):
        result=V.verify(self.root)
        self.assertEqual(result['eligible_tasks'],21574);self.assertEqual(result['indexed_rows'],21578)
        self.assertFalse(result['gpu_replay_verified']);self.assertFalse(result['parallel_eval_live'])

    def test_changed_serve_file_rejected(self):
        p=self.root/'validator/probe/serve.sh';p.write_text(p.read_text().replace('bfloat16','float16'))
        with self.assertRaisesRegex(ValueError,'sha256 mismatch'):V.verify(self.root)

    def test_changed_snapshot_rejected(self):
        p=self.root/'validator/probe/runtime/snap-affine_gatepool-v2.json.gz';p.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'sha256 mismatch'):V.verify(self.root)

    def test_shifted_eligible_mapping_rejected(self):
        p=self.root/'data/gate-pool-v2.jsonl';rows=p.read_text().splitlines();row=json.loads(rows[0]);row['id']='math:1';rows[0]=json.dumps(row);p.write_text('\n'.join(rows))
        with self.assertRaisesRegex(ValueError,'probe index mismatch'):V.verify(self.root)

    def test_optional_package_version_mismatch(self):
        with patch.object(V.importlib.metadata,'version',return_value='wrong-version'):
            with self.assertRaisesRegex(ValueError,'package version mismatch'):V.verify(self.root,True)


if __name__=='__main__':unittest.main()
