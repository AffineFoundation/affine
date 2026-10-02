"""Reserved task shards remain available to evaluation and impossible to mine."""
import copy
import base64
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.service import definitions
from subnet.gpu_service import contract,initial_manifest,heldout,owned_mining_job_fields
from subnet.protocol import entries
from subnet.backend_jobs import mining_definitions,initial_configuration
from subnet.cli import selected_tasks
from subnet.storage import Identity,canonical
from nacl.signing import VerifyKey
from subnet.controller import Controller
from subnet.harness import source_hash
from subnet.task_assets import bindings
from subnet.math_corpus_assets import hydrate
from subnet.environments import build_spec
from test_math_corpus_assets import fixture

class EvaluationOnlyRows(unittest.TestCase):
    def setUp(self):
        self.harness=dict(version='text-tools-long-v2',policy='autoregressive',max_output_tokens=1024,temperature=.8,top_p=1.)
        self.train=dict(spec=dict(id='training',num_samples=4,max_output_tokens=1024,version='fixed'),indices=[0,1],harness=self.harness)
        self.reserved=dict(spec=dict(id='reserved',num_samples=4,max_output_tokens=1024,version='fixed'),indices=[],harness=self.harness,evaluation_only=True)
        self.config=dict(environments=[self.train,self.reserved],source_bundle={},heldout=[dict(env_id='reserved',indices=[0,1],seed=42,harness=self.harness)],model_runtime_revision='cuda-bf16-eager-sm90-v1',artifact_policy='full-vocabulary-long-v1',epoch_prefix='nonpayable-reserved')
        self.parse=patch('subnet.service.EnvironmentSpec.from_dict',side_effect=lambda raw:SimpleNamespace(**raw,to_dict=lambda:raw))
        self.parse.start();self.addCleanup(self.parse.stop)
    def manifest(self,config=None):return initial_manifest(config or self.config,dict(id='weights',files={}))
    def test_legacy_nonempty_defaults_do_not_gain_a_role_flag(self):
        row=copy.deepcopy(self.train);row.pop('indices')
        value=definitions(dict(environments=[row]))[0]
        self.assertEqual(value['indices'],[0,1,2,3]);self.assertNotIn('evaluation_only',value)
    def test_empty_mining_rows_stay_invalid_without_explicit_role(self):
        for flag in (None,False):
            row=copy.deepcopy(self.reserved)
            if flag is None:row.pop('evaluation_only')
            else:row['evaluation_only']=flag
            with self.assertRaisesRegex(ValueError,'challenge indices'):definitions(dict(environments=[row]))
    def test_role_cannot_hide_nonempty_or_implicit_default_mining_indices(self):
        for indices in ([0],None,()):
            row=copy.deepcopy(self.reserved)
            if indices is None:row.pop('indices')
            else:row['indices']=indices
            with self.assertRaisesRegex(ValueError,'explicit empty'):definitions(dict(environments=[row]))
        for flag in (1,'true',None):
            row=dict(self.reserved,evaluation_only=flag)
            with self.assertRaisesRegex(ValueError,'boolean'):definitions(dict(environments=[row]))
    def test_signed_initial_contract_keeps_reserved_registry_and_fixed_heldout(self):
        manifest=self.manifest();rows=entries(manifest)
        self.assertEqual(rows[1]['indices'],[]);self.assertTrue(rows[1]['evaluation_only'])
        self.assertEqual(manifest['sample_harness_registry']['reserved']['indices'],[])
        self.assertEqual(heldout(self.config,manifest)[0]['indices'],[0,1])
        self.assertEqual(rows[0]['indices'],[0,1])
    def test_reserved_row_does_not_bypass_output_budget(self):
        row=dict(self.reserved,harness=dict(self.harness,max_output_tokens=2048))
        with self.assertRaisesRegex(ValueError,'harness exceeds environment budget'):
            definitions(dict(environments=[self.train,row]))
    def test_reserved_rows_cannot_be_selected_as_a_training_group(self):
        with self.assertRaisesRegex(ValueError,'evaluation-only environment in training group'):
            contract(dict(self.config,training_groups=[['reserved']]),0)
        with self.assertRaisesRegex(ValueError,'GPU training group'):
            contract(dict(self.config,environments=[self.reserved]),0)
    def test_owned_miner_subset_and_runtime_selection_exclude_reserved(self):
        manifest=self.manifest()
        with self.assertRaisesRegex(ValueError,'outside authorized training'):
            mining_definitions(manifest,{'mining_subset':{'reserved':[0]}})
        with self.assertRaisesRegex(ValueError,'outside authorized training'):
            owned_mining_job_fields({'owned_mining_subset':{'reserved':[0]}},manifest,0)
        self.assertEqual(selected_tasks(manifest),[('training',0),('training',1)])
        self.assertEqual(selected_tasks(manifest,'reserved'),[])
        with self.assertRaisesRegex(ValueError,'authorized unique training subset'):
            selected_tasks(manifest,'reserved',[0])
        row,harness=initial_configuration(manifest,{'role':'mine'})
        self.assertEqual(row['env_id'],'training')
        row,harness=initial_configuration(manifest,{'role':'evaluate','heldout':[{'env_id':'reserved','harness':self.harness}]})
        self.assertEqual(row['env_id'],'reserved');self.assertEqual(harness,self.harness)
    def test_public_manifest_cannot_reintroduce_reserved_mining_or_full_registry_indices(self):
        for where in ('environment','registry'):
            manifest=self.manifest()
            if where=='environment':manifest['environments'][1]['indices']=[0]
            else:manifest['sample_harness_registry']['reserved']['indices']=[0]
            with self.assertRaisesRegex(ValueError,'evaluation-only'):entries(manifest)
        manifest=self.manifest();manifest['environments'][1]['evaluation_only']=1
        with self.assertRaisesRegex(ValueError,'evaluation-only'):entries(manifest)
    def test_existing_inactive_mining_groups_remain_valid(self):
        other=copy.deepcopy(self.train);other['spec']['id']='other'
        config=dict(self.config,environments=[self.train,other,self.reserved],training_groups=[['training'],['other']])
        manifest=initial_manifest(config,dict(id='weights',files={}))
        self.assertEqual([r['indices'] for r in entries(manifest)],[[0,1],[],[]])
        self.assertEqual(manifest['sample_harness_registry']['other']['indices'],[0,1])
    def test_resource_registry_still_requires_reserved_asset(self):
        body,binding,spec=fixture();spec.id='math_corpus_deepmath103k_heldout_000';spec.config['math_corpus_asset']['fold']='heldout'
        resource=dict(read_url='https://a.r2.cloudflarestorage.com/x?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=a',compressed_sha256=binding['compressed_sha256'],compressed_size=binding['compressed_size'])
        manifest=dict(environments=[dict(spec=vars(spec),indices=[],evaluation_only=True)],task_assets={binding['sha256']:resource})
        self.assertEqual(len(bindings(manifest)),1)
        manifest['task_assets']={}
        with self.assertRaisesRegex(ValueError,'exact signed task asset registry'):bindings(manifest)

    def test_real_hydrated_corpus_shards_have_disjoint_mining_and_evaluation_contracts(self):
        self.parse.stop()
        body,training_binding,_=fixture()
        raw=json.loads(gzip.decompress(body));raw[0]['data'].update(problem='2+2?',prompt='2+2?',answer='4')
        raw=json.dumps(raw).encode();reserved_body=gzip.compress(raw,mtime=0);sha=hashlib.sha256(raw).hexdigest()
        reserved_binding=dict(training_binding,fold='heldout',sha256=sha,size=len(raw),compressed_sha256=hashlib.sha256(reserved_body).hexdigest(),compressed_size=len(reserved_body),path='assets/math-corpora/'+sha+'.tasks.json')
        url='https://a.r2.cloudflarestorage.com/x?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=a'
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{'AFFINE_MATH_CORPUS_ASSET_ROOT':folder}):
            configs=[];resources={}
            for binding,data,fold in [(training_binding,body,'train'),(reserved_binding,reserved_body,'heldout')]:
                hydrate(folder,binding,url,lambda *args:data)
                spec=build_spec('math_corpus_deepmath103k_'+fold+'_000',config={'task_snapshot':binding['path'],'math_corpus_asset':binding},num_samples=1,max_turns=1,max_output_tokens=1024)
                row=dict(spec=spec.to_dict(),indices=[0] if fold=='train' else [],harness=self.harness)
                if fold=='heldout':row['evaluation_only']=True
                configs.append(row);resources[binding['sha256']]=dict(read_url=url,compressed_sha256=binding['compressed_sha256'],compressed_size=binding['compressed_size'])
            config=dict(self.config,environments=configs,task_assets=resources,heldout=[dict(env_id=configs[1]['spec']['id'],indices=[0],seed=42,harness=self.harness)])
            manifest=initial_manifest(config,dict(id='weights',files={}))
            self.assertEqual(len(bindings(manifest)),2)
            self.assertEqual(selected_tasks(manifest),[(configs[0]['spec']['id'],0)])
            self.assertEqual(heldout(config,manifest)[0]['env_id'],configs[1]['spec']['id'])
            self.assertEqual(entries(manifest)[1]['indices'],[])
    def test_controller_persists_role_in_actual_manifest_without_external_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            controller=Controller.__new__(Controller);controller.state=Path(folder);controller.bucket=SimpleNamespace(json=Mock());controller.authority=Identity()
            gateway=SimpleNamespace(epochs={})
            def opening(epoch,miners,deadline,**kwargs):gateway.epochs[epoch]={'start':1};return {}
            gateway.open=opening;controller.gateway=gateway
            with patch('subnet.controller.EnvironmentSpec.from_dict',side_effect=lambda raw:SimpleNamespace(**raw,to_dict=lambda:raw)):
                chosen=contract(self.config,0);chosen.pop('heldout_indices')
                manifest=controller.open('nonpayable-fixture',dict(id='weights',files={}),[],**chosen)
            self.assertTrue(entries(manifest)[1]['evaluation_only'])
            self.assertTrue((Path(folder)/'nonpayable-fixture-manifest.json').exists())
            self.assertTrue(controller.bucket.json.call_args_list[0].args[1]['payload']['environments'][1]['evaluation_only'])
            signed=controller.bucket.json.call_args_list[0].args[1]
            VerifyKey(bytes.fromhex(signed['signer'])).verify(canonical(signed['payload']),base64.b64decode(signed['signature']))
            self.assertEqual(signed['payload']['sample_harness_registry']['reserved']['indices'],[])
            self.assertEqual(heldout(self.config,signed['payload'])[0]['indices'],[0,1])

    def test_controller_refuses_bad_reserved_row_before_opening_gateway(self):
        controller=Controller.__new__(Controller);controller.gateway=SimpleNamespace(open=Mock())
        for value in (1,'true',True):
            row=dict(self.reserved,evaluation_only=value,indices=[0])
            with self.assertRaisesRegex(ValueError,'evaluation-only'):
                controller.open('nonpayable-fixture',{},[],environments=[row])
        controller.gateway.open.assert_not_called()

if __name__=='__main__':unittest.main()
