import base64,copy,unittest
from nacl.signing import SigningKey
from subnet import verified_replay_pool as r
from subnet.replay_training import admitted,verified_pairs,merge_pairs

class ReplayTrainingTests(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        files={'config.json':'a'*64,'model.safetensors':'b'*64,'tokenizer.json':'c'*64};cp={'files':files,'id':r.digest(files)}
        self.manifest={'payable':False,'model_id':'approved','model_runtime_revision':'approved-v1','backend_profile':{'device':'cuda'},'numerical_policy':{'atol':1e-5},'tokenizer_binding':{'tokenizer.json':'c'*64},'checkpoint':cp,'environments':[{'env_id':'e','spec':{'id':'e','num_samples':4},'harness':{'policy':'approved'},'indices':[0,1]}],'heldout_indices':{'e':[2,3]}}
        current=self.sign(self.manifest);entry={'version':r.VERSION,'current_manifest_sha256':r.digest(current),'current_checkpoint':cp,'family':'e','environment_id':'e','environment_index':0,'task_hash':'d'*64,'target_sha256':'e'*64}
        for label in ('positive','negative'):entry[label]={'classification':label,'env_id':'e','index':0,'task_hash':'d'*64}
        pool={'version':r.POOL_VERSION,'current_manifest_sha256':r.digest(current),'policy':{'max_pairs':9,'max_reuse':8,'max_zip_bytes':250000000,'reference_policy':r.REFERENCE_POLICY},'entries':[entry]};pool['pool_sha256']=r.digest(pool)
        self.inputs={'manifest':current,'pool':self.sign(pool),'reuse_counts':{}}
    def sign(self,v):return {'payload':v,'signer':self.authority,'signature':base64.b64encode(self.key.sign(r.canonical(v)).signature).decode()}
    def test_eligible_current_pool(self):self.assertEqual(len(admitted(self.manifest,self.inputs,self.authority)[1]['selected']),1)
    def test_changed_live_checkpoint(self):
        live=copy.deepcopy(self.manifest);live['checkpoint']['id']='f'*64
        with self.assertRaises(ValueError):admitted(live,self.inputs,self.authority)
    def test_missing_live_heldout_declaration(self):
        live=copy.deepcopy(self.manifest);live.pop('heldout_indices')
        with self.assertRaises(ValueError):admitted(live,self.inputs,self.authority)
    def test_changed_live_harness(self):
        live=copy.deepcopy(self.manifest);live['environments'][0]['harness']['policy']='unapproved'
        with self.assertRaises(ValueError):admitted(live,self.inputs,self.authority)
    def test_reuse_limit_no_extra_update(self):
        inputs={**self.inputs,'reuse_counts':{'e'*64:8}}
        self.assertEqual(admitted(self.manifest,inputs,self.authority)[1]['selected'],[])
    def traced_inputs(self):
        pool=copy.deepcopy(self.inputs['pool']['payload'])
        for label in ('positive','negative'):pool['entries'][0][label]['turns']=[{'prompt':[1,2],'output':[3],'proofs':['historical']}]
        pool['pool_sha256']=r.digest({k:v for k,v in pool.items() if k!='pool_sha256'})
        return {**self.inputs,'pool':self.sign(pool)}
    def test_current_proofs_recomputed_without_mutating_history(self):
        class Runtime:
            calls=0
            def configure(self,*args):pass
            def compute(self,prompt,output):self.calls+=1;return 'current-activations','current-probabilities'
            def build_proofs(self,acts,**kwargs):return ['fresh']
            def verify(self,rollout,arrays):return rollout['turns'][0]['proofs']==['fresh'] and arrays==['current-probabilities']
        runtime=Runtime();inputs=self.traced_inputs();pairs,report=verified_pairs(runtime,self.manifest,inputs,self.authority)
        self.assertEqual(runtime.calls,2);self.assertEqual(pairs[0][1]['turns'][0]['proofs'],['historical'])
        self.assertFalse(report['optimizer_performed']);self.assertTrue(report['checks'][0]['fresh_current_numerical_native_verification'])
    def test_failed_native_verification_never_returns_training_pair(self):
        class Runtime:
            def configure(self,*args):pass
            def compute(self,*args):return None,None
            def build_proofs(self,*args,**kwargs):return ['fresh']
            def verify(self,*args):return False
        with self.assertRaises(ValueError):verified_pairs(Runtime(),self.manifest,self.traced_inputs(),self.authority)
    def pair(self,env,index):
        return ({'env_id':env},{'index':index,'task_hash':'a'*64,'classification':'positive'},{'index':index,'task_hash':'a'*64,'classification':'negative'})
    def test_current_family_supersedes_historical_duplicate(self):
        fresh=self.pair('e',1);pairs,targets=merge_pairs([fresh],[self.pair('e',0),self.pair('f',0)],{})
        self.assertEqual(pairs[0],fresh);self.assertEqual(len(pairs),2);self.assertEqual(len(targets),1)
    def test_multiple_current_pairs_need_weighted_objective(self):
        with self.assertRaises(ValueError):merge_pairs([self.pair('e',0),self.pair('e',1)],[],{})
    def test_least_used_historical_target_rotates_task(self):
        a=self.pair('e',0);b=self.pair('e',1)
        _,used=merge_pairs([], [a],{});target=next(iter(used));pairs,_=merge_pairs([], [a,b],{target:3})
        self.assertEqual(pairs,[b])
if __name__=='__main__':unittest.main()
