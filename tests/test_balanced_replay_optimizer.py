import base64,copy,unittest
from nacl.signing import SigningKey
from ops.probe_balanced_replay_optimizer import approve
from subnet import verified_replay_pool as r

class ReplayOptimizerAdmission(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        files={'config.json':'a'*64,'model.safetensors':'b'*64};cp={'files':files,'id':r.digest(files)}
        self.manifest={'checkpoint':cp,'environments':[{'env_id':'e','spec':{'id':'e','num_samples':4},'harness':None,'indices':[0,1]}],'heldout_indices':{'e':[2,3]}}
        self.m=self.sign(self.manifest)
        entry={'version':r.VERSION,'current_manifest_sha256':r.digest(self.m),'current_checkpoint':cp,'family':'e','environment_id':'e','environment_index':0,'task_hash':'c'*64,'target_sha256':'d'*64}
        for label in ('positive','negative'):entry[label]={'classification':label,'env_id':'e','index':0,'task_hash':'c'*64}
        body={'version':r.POOL_VERSION,'current_manifest_sha256':r.digest(self.m),'policy':{'max_pairs':9,'max_reuse':8,'max_zip_bytes':250000000,'reference_policy':r.REFERENCE_POLICY},'entries':[entry]}
        self.pool={**body,'pool_sha256':r.digest(body)}
        self.plan={'revision':'balanced-current-reference-qualification-v1','payable':False,'chain_transactions':False,'current_manifest_sha256':r.digest(self.m),'pool_sha256':self.pool['pool_sha256'],'target_sha256':['d'*64],'steps':1}
    def sign(self,v):return {'payload':v,'signer':self.authority,'signature':base64.b64encode(self.key.sign(r.canonical(v)).signature).decode()}
    def test_valid_selection(self):self.assertEqual(len(approve(self.sign(self.plan),self.m,self.sign(self.pool),self.authority)[2]),1)
    def test_unapproved_model(self):
        pool=copy.deepcopy(self.pool);pool['entries'][0]['current_checkpoint']['id']='f'*64
        pool['pool_sha256']=r.digest({k:v for k,v in pool.items() if k!='pool_sha256'});plan={**self.plan,'pool_sha256':pool['pool_sha256']}
        with self.assertRaises(ValueError):approve(self.sign(plan),self.m,self.sign(pool),self.authority)
    def test_each_family_update_required(self):
        with self.assertRaises(ValueError):approve(self.sign({**self.plan,'steps':0}),self.m,self.sign(self.pool),self.authority)
    def test_positive_negative_swap(self):
        pool=copy.deepcopy(self.pool);pool['entries'][0]['negative']['classification']='positive';pool['pool_sha256']=r.digest({k:v for k,v in pool.items() if k!='pool_sha256'})
        with self.assertRaises(ValueError):approve(self.sign({**self.plan,'pool_sha256':pool['pool_sha256']}),self.m,self.sign(pool),self.authority)
if __name__=='__main__':unittest.main()
