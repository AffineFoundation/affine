import base64
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError
from ops.training_retention import canonical,digest,hash_file,remove_training_replica


class TrainingRetention(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.workspace=Path(self.tmp.name);self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        self.root=self.workspace/'jobs'/'test-train';self.root.mkdir(parents=True)
        self.gpu=patch('ops.training_retention.gpu_processes',return_value=[]);self.gpu.start();self.addCleanup(self.gpu.stop)
        self.proc=patch('ops.training_retention.processes',return_value=[Path('/proc',str(os.getpid()))]);self.proc.start();self.addCleanup(self.proc.stop)
        self.contents={'config.json':b'{}','model.safetensors':b'model'}
        self.checkpoint=digest({n:hashlib.sha256(v).hexdigest() for n,v in self.contents.items()})
        self.manifest={'epoch':'epoch','checkpoint':{'id':'a'*64}}
        self.job={'job_id':'test-train','role':'train','steps':3,'manifest':self.sign(self.manifest),
                  'submissions':[{'sha256':hashlib.sha256(b'zip').hexdigest()}]}
        self.report={'job_id':'test-train','role':'train','success':True,'operator':self.authority,
                     'job_sha256':digest(self.job),'epoch':'epoch','checkpoint':'a'*64,'new_checkpoint':{'id':self.checkpoint}}
        self.terminal={'job_id':'test-train','phase':'complete','exit_code':0,'runner_pid':999999998,
                       'child_pid':999999999,'runner_pid_ticks':'0','child_pid_ticks':'0'}
        self.write_evidence()

    def sign(self,value):
        return {'payload':value,'signer':self.authority,'signature':base64.b64encode(self.key.sign(canonical(value)).signature).decode()}

    def write_evidence(self):
        self.paths={'job':self.workspace/'test-train.json','report':self.root/'report.json',
                    'terminal':self.workspace/'runner-status/test-train.json'}
        for name,value in [('job',self.sign(self.job)),('report',self.report),('terminal',self.terminal)]:
            self.paths[name].parent.mkdir(parents=True,exist_ok=True);self.paths[name].write_bytes(canonical(value))

    def plan(self,kind='submission',step=1):
        if kind=='submission':
            contents={'submission-0.zip':b'zip'};root=self.root
        else:contents=self.contents;root=self.root/('checkpoint-step-'+str(step));root.mkdir()
        for n,v in contents.items():(root/n).write_bytes(v)
        return {'workspace':str(self.workspace),'job_id':'test-train','authority':self.authority,
                **{n+'_sha256':hash_file(p) for n,p in self.paths.items()},'archive_verified':True,'archive_authenticated':True,
                'kind':kind,'directory':str(root),'submission_index':0,'step':step,'checkpoint':self.checkpoint,
                'protected_checkpoints':['b'*64],'files':{n:{'sha256':hashlib.sha256(v).hexdigest(),'size':len(v)} for n,v in contents.items()}}

    def test_completed_archived_download_removed_and_report_preserved(self):
        p=self.plan();self.assertEqual(remove_training_replica(p)['bytes'],3)
        self.assertTrue(all(path.is_file() for path in self.paths.values()))
        self.assertTrue(remove_training_replica(p)['already_absent'])

    def test_report_before_actual_terminal_wait_is_insufficient(self):
        self.terminal.update(phase='running');self.write_evidence();p=self.plan()
        with self.assertRaisesRegex(ValueError,'completed'):remove_training_replica(p)
        self.assertTrue((self.root/'submission-0.zip').exists())

    def test_archives_hashes_paths_and_positions_cannot_be_substituted(self):
        p=self.plan()
        for change in [{'archive_verified':False},{'archive_authenticated':False},{'job_sha256':'0'*64},
                       {'submission_index':1},{'directory':str(self.workspace)}]:
            with self.assertRaises(ValueError):remove_training_replica(dict(p,**change))
        (self.root/'submission-0.zip').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'bytes'):remove_training_replica(p)

    def test_actual_original_process_must_have_exited(self):
        self.terminal.update(child_pid=os.getpid(),child_pid_ticks=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()[19]);self.write_evidence();p=self.plan()
        with self.assertRaisesRegex(ValueError,'still alive'):remove_training_replica(p)

    def test_rehashed_evidence_does_not_authorize_a_forged_signature(self):
        p=self.plan();envelope=json.loads(self.paths['job'].read_text())
        signature=bytearray(base64.b64decode(envelope['signature']));signature[0]^=1
        envelope['signature']=base64.b64encode(signature).decode()
        self.paths['job'].write_bytes(canonical(envelope));p['job_sha256']=hash_file(self.paths['job'])
        with self.assertRaises(BadSignatureError):remove_training_replica(p)
        self.assertTrue((self.root/'submission-0.zip').exists())

    def test_open_hardlinked_and_gpu_used_downloads_refused(self):
        p=self.plan();path=self.root/'submission-0.zip'
        with path.open('rb'):
            with self.assertRaisesRegex(ValueError,'still open'):remove_training_replica(p)
        os.link(path,self.workspace/'other')
        with self.assertRaisesRegex(ValueError,'bytes'):remove_training_replica(p)
        (self.workspace/'other').unlink()
        with patch('ops.training_retention.gpu_processes',return_value=['123']):
            with self.assertRaisesRegex(ValueError,'idle GPU'):remove_training_replica(p)

    def test_archived_intermediate_export_removed_current_export_protected(self):
        p=self.plan('checkpoint-export');self.assertTrue(remove_training_replica(p)['removed'])
        self.assertTrue(self.paths['report'].exists())
        p=self.plan('checkpoint-export',step=3)
        with self.assertRaisesRegex(ValueError,'unprotected'):remove_training_replica(dict(p,protected_checkpoints=[self.checkpoint]))
        self.assertTrue(Path(p['directory']).exists())

    def test_final_export_requires_original_report_binding_and_exact_membership(self):
        self.report['new_checkpoint']['id']='f'*64;self.write_evidence();p=self.plan('checkpoint-export',step=3)
        with self.assertRaisesRegex(ValueError,'final checkpoint'):remove_training_replica(p)
        p['step']=1
        with self.assertRaises(ValueError):remove_training_replica(p)


if __name__=='__main__':unittest.main()
