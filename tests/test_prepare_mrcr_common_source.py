import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ops import prepare_mrcr_common_source as prep

H = '''import hashlib,json,math
from pathlib import Path
HARNESS_REGISTRY={'text-tools-v1':True,'plain-transcript-v1':True}
def normalize(config):
    value=dict(config)
    if value['policy'] not in ('autoregressive', 'candidates', 'visible-copy-candidates'):raise ValueError('policy')
    if value['policy'] == 'candidates':
        if len(value['candidates'])<2:raise ValueError('candidates')
    return value

def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def sample(config,messages):
    if config['policy'] == 'visible-copy-candidates':
        config=dict(config,policy='candidates')
    return config
'''
G = '''def sample(config,messages,policy):
        if config['policy']=='visible-copy-candidates':
            config={**config,'policy':'candidates'}
        return config
'''


class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'base';pkg=self.source/'subnet';pkg.mkdir(parents=True)
        (pkg/'harness.py').write_text(H);(pkg/'gpu_runtime.py').write_text(G);(pkg/'__init__.py').write_text('')
        self.policy=Path(__file__).resolve().parents[1]/'subnet/native_mrcr_public_policy.py'
        self.base={n:hashlib.sha256((pkg/n).read_bytes()).hexdigest() for n in prep.BASE}
        self.mock=patch.object(prep,'BASE',self.base);self.mock.start();self.addCleanup(self.mock.stop)
        self.dest=self.root/'new'

    def load(self):
        spec=importlib.util.spec_from_file_location('subnet._staged_test_harness',self.dest/'subnet/harness.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

    def test_staged_source_and_pin_are_bound_without_mutating_base(self):
        record=prep.prepare(self.source,self.dest,self.policy)
        self.assertEqual(record['base_source_files'],self.base)
        self.assertFalse(record['models_copied']);self.assertFalse(record['state_copied'])
        self.assertEqual((self.source/'subnet/harness.py').read_text(),H)
        module=self.load();expected=hashlib.sha256((self.dest/'subnet/harness.py').read_bytes()+self.policy.read_bytes()).hexdigest()
        self.assertEqual(module.source_hash(),expected)
        self.assertEqual(module.normalize({'policy':'autoregressive'}),{'policy':'autoregressive'})
        with self.assertRaises(ValueError):module.normalize({'policy':'public-mrcr-shell-candidates','public_policy_sha256':'0'*64})

    def test_changed_base_and_policy_refused_before_destination_exists(self):
        (self.source/'subnet/harness.py').write_text(H+'# drift\n')
        with self.assertRaises(ValueError):prep.prepare(self.source,self.dest,self.policy)
        self.assertFalse(self.dest.exists())
        (self.source/'subnet/harness.py').write_text(H)
        bad=self.root/'policy.py';bad.write_text('modified')
        with self.assertRaises(ValueError):prep.prepare(self.source,self.dest,bad)
        self.assertFalse(self.dest.exists())

    def test_state_models_symlinks_and_existing_destinations_refused(self):
        pkg=self.source/'subnet'
        for name in ['.env','weights.safetensors','weights.bin']:
            p=pkg/name;p.write_text('private')
            with self.subTest(name=name),self.assertRaises(ValueError):prep.prepare(self.source,self.dest,self.policy)
            p.unlink()
        p=pkg/'extra.py';p.symlink_to(self.policy)
        with self.assertRaises(ValueError):prep.prepare(self.source,self.dest,self.policy)
        p.unlink();self.dest.mkdir()
        with self.assertRaises(ValueError):prep.prepare(self.source,self.dest,self.policy)
        with self.assertRaises(ValueError):prep.prepare(self.source,self.source/'nested',self.policy)

    def test_generated_cpu_gpu_dispatch_compiles(self):
        prep.prepare(self.source,self.dest,self.policy)
        for name in prep.BASE:compile((self.dest/'subnet'/name).read_text(),name,'exec')
        for name in prep.BASE:self.assertIn('public-mrcr-shell-candidates',(self.dest/'subnet'/name).read_text())

    def test_candidates_use_only_public_question_and_comparable_commands(self):
        prep.prepare(self.source,self.dest,self.policy);module=self.load()
        question='Prepend abcdef123456 to the second poem about stars in a humorous style. Do not include any other text in your response.'
        messages=[{'role':'user','content':question+'\n\nRead /workspace/context.txt.'}]
        values=module.mrcr_candidates(messages)
        commands=[json.loads(v)['tool_call']['arguments']['command'] for v in values]
        self.assertLessEqual(abs(len(commands[0])-len(commands[1])),1)
        self.assertIn('abcdef123456',commands[0]);self.assertIn('0bcdef123456',commands[1])
        self.assertTrue(all('/workspace/context.txt' in c and '/workspace/answer.txt' in c for c in commands))
        with self.assertRaises(ValueError):module.mrcr_candidates(messages+messages)
        with self.assertRaises(ValueError):module.mrcr_candidates([{'role':'assistant','content':question}])


if __name__=='__main__':unittest.main()
