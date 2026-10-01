import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ops.check_service_evidence import check
from subnet.storage import Identity, canonical


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        state=root/'service';state.mkdir();evaluations=root/'eval';evaluations.mkdir()
        authority=Identity();(state/'authority.seed').write_text(authority.key.encode().hex())
        epoch='nonpayable-service-fixture';miner='a'*64
        checkpoint_dir=state/'checkpoints'/epoch;checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir/'model.safetensors').write_bytes(b'changed model bytes')
        files={'model.safetensors':hashlib.sha256(b'changed model bytes').hexdigest()}
        checkpoint=hashlib.sha256(canonical(files)).hexdigest()
        manifest=dict(epoch=epoch,payable=False,start=100,deadline=200,
                      checkpoint={'id':'old','files':{'model.safetensors':'oldhash'}},
                      evaluation={'suites':[{'env_id':'original'}]})
        receipt=dict(received_at=150,sha256=hashlib.sha256(b'frozen').hexdigest(),frozen_key='frozen')
        scores=dict(epoch_id=epoch,payable=False,checkpoint='old',finalized_at=201,
                    receipts={miner:receipt},weights={miner:1},total=1)
        challenge=dict(seed='seed',receipts=scores['receipts'],generated_after_freeze_at=200)
        audit=dict(epoch=epoch,submission_sha256=receipt['sha256'],audit_seed='seed',
                   outcomes=[{'valid':True,'fully_audited':True}],accepted=[{'env_id':'original'}])
        training=dict(weights_changed=True,steps=1,checkpoint=checkpoint)
        (state/f'{epoch}-manifest.json').write_bytes(canonical(manifest))
        (state/f'{epoch}-training-metrics.json').write_bytes(canonical(training))
        for phase,cp in [('before','old'),('after',checkpoint)]:
            record=dict(status='complete',dataset_id='fixed',task_hashes=['heldout'],
                        runtime_profile={'threads':4},checkpoint=cp,mean_reward=0)
            (evaluations/f'{epoch}-{phase}-original.json').write_bytes(canonical(record))
        objects={'frozen':b'frozen'}
        def sign(key,value):
            objects[key]=canonical(dict(payload=value,signer=authority.id,
                signature=base64.b64encode(authority.key.sign(canonical(value)).signature).decode()))
        for name,value in [('manifest',manifest),('scores',scores),('audit-challenge',challenge),('training',training)]:
            sign(f'public/{epoch}/{name}.json',value)
        sign(f'public/{epoch}/audits/{miner}.json',audit)
        sign(f'public/checkpoints/{checkpoint}/authorities/{authority.id}/checkpoint.json',
             {'id':checkpoint,'files':files})
        class ReadOnlyBucket:
            def get(self,key):return objects[key]
        return state,evaluations,ReadOnlyBucket(),objects,sign,epoch,scores

    def test_honest_evidence_and_changed_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            state,evals,bucket,objects,_,_,_=self.fixture(Path(temp))
            result=check(state,bucket,evals)
            self.assertEqual(len(result['epochs']),1)
            self.assertEqual(result['payout_filter_result'],{})
            objects['frozen']=b'replacement'
            with self.assertRaisesRegex(ValueError,'frozen artifact changed'):check(state,bucket,evals)

    def test_validly_signed_early_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            state,evals,bucket,_,sign,epoch,scores=self.fixture(Path(temp))
            scores['finalized_at']=199
            sign(f'public/{epoch}/scores.json',scores)
            with self.assertRaisesRegex(ValueError,'froze before deadline'):check(state,bucket,evals)

    def test_training_in_progress_remains_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            state,evals,bucket,_,_,epoch,_=self.fixture(Path(temp))
            (evals/f'{epoch}-after-original.json').unlink()
            result=check(state,bucket,evals)
            self.assertEqual(result['epochs'],[])
            self.assertEqual(result['pending_epochs'],[epoch])

    def test_wrong_namespaced_signer_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as temp:
            state,evals,bucket,objects,_,_,_=self.fixture(Path(temp))
            key=next(k for k in objects if '/authorities/' in k)
            original=json.loads(objects[key]);payload=original['payload']
            objects[f'public/checkpoints/{payload["id"]}/checkpoint.json']=objects[key]
            attacker=Identity()
            objects[key]=canonical(dict(payload=payload,signer=attacker.id,
                signature=base64.b64encode(attacker.key.sign(canonical(payload)).signature).decode()))
            with self.assertRaisesRegex(ValueError,'wrong checkpoint descriptor authority'):
                check(state,bucket,evals)

    def test_direct_manifest_cannot_hide_tunnel_file_route(self):
        with tempfile.TemporaryDirectory() as temp:
            state,evals,bucket,_,sign,epoch,_=self.fixture(Path(temp))
            path=state/f'{epoch}-manifest.json';manifest=json.loads(path.read_text())
            manifest['transport_policy']='direct-r2-v1'
            manifest['checkpoint']['read_urls']={'model.safetensors':'https://example.trycloudflare.com/model'}
            path.write_bytes(canonical(manifest));sign(f'public/{epoch}/manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'not direct R2'):check(state,bucket,evals)

    def test_empty_epoch_can_retry_but_cannot_change_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            state,evals,bucket,_,sign,epoch,scores=self.fixture(Path(temp))
            (state/f'{epoch}-empty-closed.json').write_text('{}')
            scores.update(total=0,points={},receipts={},weights={})
            sign(f'public/{epoch}/scores.json',scores)
            sign(f'public/{epoch}/training.json',{'status':'paused_no_verified_pairs'})
            manifest=json.loads((state/f'{epoch}-manifest.json').read_text())
            following=dict(manifest,epoch='nonpayable-service-next',start=201,deadline=301)
            path=state/f'{following["epoch"]}-manifest.json';path.write_bytes(canonical(following))
            sign(f'public/{following["epoch"]}/manifest.json',following)
            result=check(state,bucket,evals)
            self.assertEqual(result['epochs'],[])
            self.assertEqual(result['empty_epochs'][0]['next_epoch'],following['epoch'])
            following['checkpoint']={'id':'untrained-change','files':{}}
            path.write_bytes(canonical(following));sign(f'public/{following["epoch"]}/manifest.json',following)
            with self.assertRaisesRegex(ValueError,'changed checkpoint without training'):check(state,bucket,evals)


if __name__=='__main__':unittest.main()
