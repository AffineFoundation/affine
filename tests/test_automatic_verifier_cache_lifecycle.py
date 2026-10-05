import hashlib
import unittest
from subnet.storage import canonical
from ops.automatic_verifier_cache_lifecycle import select_candidate,verified_launch_map,PROBE,save,private_output,capacity_admission,select_cluster,journalled_operation,future_worker_arguments

class AutomaticCacheTests(unittest.TestCase):
    def setUp(self):
        self.files={'config.json':hashlib.sha256(b'{}').hexdigest(),'model.safetensors':hashlib.sha256(b'weights').hexdigest()}
        self.cp=hashlib.sha256(canonical(self.files)).hexdigest()
        self.other='a'*64
    def observation(self,cp=None,free=10):
        cp=cp or self.cp
        return dict(gpu_busy=False,scientific=[],free_bytes=free,worker_mappings=[],candidates=[dict(checkpoint=cp,directory='/cache/'+cp,single_link=True,files={n:dict(ordinary=True,links=1)for n in self.files})])
    def test_current_pending_and_worker_cli_mapping_are_protected(self):
        obs=self.observation()
        self.assertIsNone(select_candidate({'verify2':obs},[self.cp,self.other]))
        obs['worker_mappings']=[dict(checkpoint=self.cp,directory='/cache/'+self.cp)]
        self.assertIsNone(select_candidate({'verify2':obs},[]))
    def test_busy_cpu_gpu_uncertain_roles_and_aliases_defer(self):
        obs=self.observation()
        self.assertIsNone(select_candidate({'verify2':obs},[],['verify2']))
        obs['scientific']=[dict(pid=123,ticks='42')]
        self.assertIsNone(select_candidate({'verify2':obs},[]));obs['scientific']=[]
        obs['gpu_busy']=True;self.assertIsNone(select_candidate({'verify2':obs},[]))
        obs['gpu_busy']=False;obs['candidates'][0]['single_link']=False
        self.assertIsNone(select_candidate({'verify2':obs},[]))
    def test_one_candidate_prioritizes_actual_free_capacity(self):
        obs={'verify1':self.observation(free=20),'verify2':self.observation(free=10)}
        self.assertEqual(select_candidate(obs,[])[0],'verify2')
    def test_future_map_requires_hash_callback_and_preserves_live(self):
        calls=[]
        def verify(path,cp,files):calls.append((path,cp,files));return True
        result=verified_launch_map({self.cp:self.files},self.observation(),verify)
        self.assertEqual(len(calls),1);self.assertEqual(result['checkpoint_caches'],{self.cp:'/cache/'+self.cp})
        self.assertTrue(result['future_launch_only']);self.assertFalse(result['live_worker_changed'])
        self.assertEqual(result['cli_arguments'],['--checkpoint-cache',self.cp+'=/cache/'+self.cp])
    def test_incomplete_or_bad_cache_is_missing_not_blind_mapping(self):
        obs=self.observation();obs['candidates'][0]['files'].pop('config.json')
        result=verified_launch_map({self.cp:self.files},obs,lambda *_:self.fail('incomplete cache cannot be hashed/admitted'))
        self.assertEqual(result['missing_checkpoints'],[self.cp])
        result=verified_launch_map({self.cp:self.files},self.observation(),lambda *_:False)
        self.assertEqual(result['checkpoint_caches'],{})
    def test_signed_filemap_identity_and_unsafe_names_refuse(self):
        with self.assertRaises(ValueError):verified_launch_map({self.other:self.files},self.observation(),lambda *_:True)
        unsafe=dict(self.files);unsafe['../secret']='f'*64;cp=hashlib.sha256(canonical(unsafe)).hexdigest()
        with self.assertRaises(ValueError):verified_launch_map({cp:unsafe},self.observation(),lambda *_:True)
    def test_capacity_counts_missing_checkpoint_plus_signed_artifact_and_reserve(self):
        maps=dict(missing_checkpoints=[self.cp]);reads={self.cp:{'model':dict(size=15000,archive_verified=True)}}
        r=capacity_admission(maps,reads,free_bytes=14000,artifact_bytes=2000,reserve_bytes=2000)
        self.assertFalse(r['admitted']);self.assertEqual(r['required_free_bytes'],19000)
        maps['missing_checkpoints']=[]
        self.assertTrue(capacity_admission(maps,reads,free_bytes=14000,artifact_bytes=2000,reserve_bytes=2000)['admitted'])
        with self.assertRaises(ValueError):capacity_admission(maps,reads,free_bytes=True,artifact_bytes=0,reserve_bytes=0)
    def test_fsynced_private_journal_refuses_symlinks_and_overwrite(self):
        import tempfile,os
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);p=root/'record.json'
            with patch('os.fsync',wraps=os.fsync)as fsync:
                save(p,{'started':True});self.assertGreaterEqual(fsync.call_count,2)
            self.assertEqual(p.stat().st_mode&0o777,0o600)
            with self.assertRaises(FileExistsError):save(p,{})
            linked=root/'linked';linked.symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError):save(linked/'else.json',{})
            with self.assertRaises(ValueError):private_output(linked/'new')
            other=root/'other';other.symlink_to(p)
            with self.assertRaises(FileExistsError):save(other,{})
    def test_cluster_selected_only_when_unreferenced_exact_two_known_links(self):
        left=self.observation()['candidates'][0];left['single_link']=False
        right=dict(left,directory='/other/'+self.cp)
        for row in(left,right):row['files']={n:dict(ordinary=True,links=2,device=1,inode=i,size=20)for i,n in enumerate(self.files)}
        obs=self.observation();obs['candidates']=[left,right]
        catalog=[dict(role='verify2',checkpoint=self.cp,directories=[left['directory'],right['directory']])]
        self.assertIsNotNone(select_cluster(catalog,{'verify2':obs},[]))
        self.assertIsNone(select_cluster(catalog,{'verify2':obs},[self.cp]))
        obs['worker_mappings']=[dict(checkpoint=self.cp)];self.assertIsNone(select_cluster(catalog,{'verify2':obs},[]))
        obs['worker_mappings']=[];right['files']=dict(right['files']);right['files']['config.json']=dict(right['files']['config.json'],links=3)
        self.assertIsNone(select_cluster(catalog,{'verify2':obs},[]))
    def test_transport_uncertainty_quarantines_role_and_prohibits_repeat(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory()as d:
            out=Path(d);cycle=out/'cycle';cycle.mkdir(mode=0o700)
            def uncertain():raise TimeoutError('fixture remote observation timeout')
            with self.assertRaises(TimeoutError):journalled_operation(out,cycle,'verify2',{'checkpoint':self.cp},uncertain)
            self.assertTrue((out/'verify2-pending-operation.private.json').exists())
            self.assertTrue(json.loads((cycle/'uncertain-operation.private.json').read_text())['automatic_repeat_forbidden'])
            with self.assertRaises(FileExistsError):journalled_operation(out,cycle,'verify2',{},lambda:self.fail('must never repeat uncertain remote operation'))
            self.assertIsNone(select_candidate({'verify2':self.observation()},[],['verify2']))
    def test_future_worker_args_drop_stale_maps_preserve_authority_seed_and_workspace(self):
        argv=['python','-B','-m','subnet.distributed_worker','--authority','operator','--seed-file','/private/seed','--workspace','/worker','--checkpoint-cache',self.other+'=/old']
        mapping=dict(future_launch_only=True,missing_checkpoints=[],capacity={'admitted':True},checkpoint_caches={self.cp:'/verified'})
        actual=future_worker_arguments(argv,mapping)
        self.assertEqual(actual[:10],argv[:10]);self.assertEqual(actual[-2:],['--checkpoint-cache',self.cp+'=/verified'])
        self.assertNotIn(self.other+'=/old',actual)
        mapping['missing_checkpoints']=[self.cp]
        with self.assertRaises(ValueError):future_worker_arguments(argv,mapping)
    def test_scanner_actual_module_names_and_native_cpu_gate(self):
        # Execute the actual embedded scanner with a simulated Linux proc tree.
        import tempfile,json,os,contextlib,io
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);proc=root/'proc';proc.mkdir();cache=root/'checkpoints';cache.mkdir()
            def process(pid,argv):
                p=proc/str(pid);p.mkdir();(p/'stat').write_text(str(pid)+' (x) S '+'0 '*18+'42 0');(p/'cmdline').write_bytes('\0'.join(argv).encode()+b'\0')
            process(1,['python','-B','-m','subnet.distributed_worker','--checkpoint-cache',self.cp+'=/cache/'+self.cp])
            process(2,['python','-B','-m','subnet.backend_jobs','job.json'])
            process(3,['python','-B','/root/affine-original-qwen-i3math-generation-v1/generate_only_node.py'])
            original=Path.iterdir
            def iterdir(p):return original(proc if str(p)=='/proc'else p)
            capture=io.StringIO()
            with patch.object(Path,'iterdir',iterdir),patch('subprocess.check_output',return_value=''),contextlib.redirect_stdout(capture):
                # Path('/proc',pid) must resolve to synthetic entries as well.
                code=PROBE.replace("Path('/proc').iterdir()",'Path('+repr(str(proc))+').iterdir()')
                exec(code,dict(ROOTS=[str(cache)]))
            result=json.loads(capture.getvalue())
            self.assertEqual(result['worker_mappings'][0]['checkpoint'],self.cp)
            self.assertEqual({r['pid']for r in result['scientific']},{2,3})
if __name__=='__main__':unittest.main()
