"""Actual native fp32 finish/save collision regression; CPU, never a GPU qualification."""
import ast,hashlib,json,os,tempfile,threading,types,unittest
from pathlib import Path
import torch
from safetensors.torch import load_file
from subnet.storage import Identity
from subnet.mimo_release_policy import sign
from .durable import digest,read
from .test_protocol import Replay
from .trainer_generation import configure

class TrainerGenerationTests(unittest.TestCase):
    def test_actual_native_finish_preserves_both_same_bf16_residual_generations(self):
        source=os.environ.get('PE_NATIVE_TRAINER_SOURCE')
        if not source:self.skipTest('native source required')
        with tempfile.TemporaryDirectory() as td:
            r=Replay(Path(td)/'native-capture');row=r.publisher.registry[r.genesis]
            output,unused_master=r.weights('candidate',1.0001)
            names={'_sha_tensor','_digest_map','_fp32_save','_fp32_finish','_fp32_gc','_fp32_store_entries'}
            tree=ast.parse(Path(source).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
            self.assertEqual({n.name for n in nodes},names)
            native=types.ModuleType('native_capture_fixture')
            native.__dict__.update(Path=Path,json=json,hashlib=hashlib,FP32_STORE=str(r.publisher.masters.root),FP32_KEEP=2,FP32_VERSION='v833f-fp32-head-master-v1',FP32_SHARD_BYTES=128<<20,FP32_SAVE_DEFER_MAX=0.01,_PUBLISHED=threading.Event())
            exec(compile(ast.Module(body=nodes,type_ignores=[]),source,'exec'),native.__dict__)
            root=Identity();job=digest('native-test-job');generation=digest('native-output-generation')
            cert=sign(root,dict(version='pe-isolated-trainer-admission-v1',payable=False,stage_root=str(r.stage.root),replay_authority=r.operator.id,original_job_sha256=job,allowed_node_ids=['fixture-node']))
            request=sign(r.operator,dict(version='pe-native-train-request-v1',payable=False,training_output_tokens=32768,input=row,output_generation_id=generation,command_id=job,original_job_sha256=job,node_id='fixture-node'))
            configure(native,r.stage.root,request,cert,root.id,output)
            w=load_file(str(output/'model.safetensors'))['w']
            opt=types.SimpleNamespace(parameters=[('w',w)],masters={'w':torch.zeros_like(w,dtype=torch.float32)},global_step=1)
            info=native._fp32_prepare(opt,None,lambda x:None)
            opt.masters['w'][0]=1.0001
            info=native._fp32_finish(opt,info,lambda x:None)
            with opt._v833f_lock:pass
            entry=info['output_store_entry'];meta=r.publisher.masters.verify(entry,output)
            self.assertEqual(meta['bf16_sha256'],read(r.publisher.masters.root/row['master_entry']/'meta.json')['bf16_sha256'])
            self.assertNotEqual(meta['master_digest'],row['master_digest'])
            self.assertNotEqual(entry,row['master_entry'])
            self.assertEqual(meta['generation_id'],generation)
            self.assertTrue((r.publisher.masters.root/row['master_entry']).is_dir())
            self.assertIn('training:'+job,read(r.publisher.masters.root/entry/'PE-PINS.json')['owners'])

if __name__=='__main__':unittest.main()
