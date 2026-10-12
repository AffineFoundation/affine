import base64
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from nacl.signing import SigningKey
import train_contract as c
from train_archive import validate

KEY = SigningKey(bytes(range(32)))  # Synthetic fixture only, never ROOT.
AUTH = KEY.verify_key.encode().hex()


def sign(value):
    return dict(payload=value, signer=AUTH,
                signature=base64.b64encode(KEY.sign(c.canonical(value)).signature).decode())


def fixture():
    manifest = dict(epoch='synthetic-epoch-1', heldout_indices=[900],
                    model_runtime_revision='cuda-fp32-eager-sm90-v1', backend_profile={'dtype':'float32'},
                    numerical_policy={'logprob_atol':1e-5},
                    environments=[dict(env_id='affine_math', indices=list(range(128)), spec={'id':'affine_math'})])
    tasks = [dict(env_id='affine_math', index=i, normalized_question_sha256=c.digest(['q',i]),
                  registry_row_sha256=c.digest(['r',i]), seed=c.seed_for(i,c.digest(['q',i]))) for i in range(128)]
    cohort = dict(version=c.COHORT_VERSION, task_count=128, tasks=tasks, tasks_sha256=c.digest(tasks),
                  selection_uses_outcomes=False, confirmation=False,
                  approved_manifest_payload_sha256=c.digest(manifest))
    files={'config.json':c.digest('config')}
    plan = dict(version=c.PLAN_VERSION, task_count=128, production_mutation=False, optimizer_allocation=False,
                training_cohort=sign(cohort), approved_mining_manifest=sign(manifest), manifest=sign(manifest),
                cohort_sha256=c.digest(cohort), checkpoint={'id':c.digest(files),'files':files},
                suites=[dict(env_id='affine_math',harness=c.HARNESS,indices=list(range(n,n+32)),
                             seeds=[t['seed'] for t in tasks[n:n+32]]) for n in range(0,128,32)],
                scientific_files={'subnet/cached_sampling.py':c.digest('sample')}, source_files={'subnet/example.py':c.digest('source')},
                runtime_versions={'torch':'2.14.0','transformers':'5.14.1'}, run_id='synthetic-train128',
                training_run_id='synthetic-training-run', epoch='synthetic-epoch-1',phase='baseline',optimizer_step=0,
                reference_kind='original-upstream-comparison',actual_run_genesis_checkpoint=c.digest('actual-genesis'))
    plan['profile']=c.scientific_profile(plan);plan['profile_sha256']=c.digest(plan['profile'])
    plan['seeds_sha256']=c.digest([(t['index'],t['seed']) for t in tasks])
    rows=[dict(env_id='affine_math',index=t['index'],seed=t['seed'],checkpoint=plan['checkpoint']['id'],
               cohort_sha256=plan['cohort_sha256'],task_hash=c.digest(['native',t['index']]),
               profile_sha256=plan['profile_sha256'],generation_origin='fresh-policy-generation',
               classification='positive' if t['index']<96 else 'negative', reward=1 if t['index']<96 else 0,
               native_graded=True,verified=False,proof_verification_performed=False,
               cap_hit=False,terminal_eos=True,budget_exhausted=False,native_results=[],
               raw_turns=[dict(prompt_tokens=[10,t['index']],output_tokens=[22,151645],seed=t['seed'])]) for t in tasks]
    return plan,rows


class ContractTests(unittest.TestCase):
    def test_fixed_training_plan_and_full_denominator(self):
        p,r=fixture();c.validate_plan(p,AUTH);s=c.summarize(p,r)
        self.assertEqual((s['requested'],s['solved'],s['wrong'],s['completed']),(128,96,32,128))
        self.assertEqual(s['exposure_groups']['unknown']['count'],128)
        self.assertNotEqual(s['checkpoint'],s['actual_run_genesis_checkpoint'])

    def test_membership_and_heldout_are_separate_guard(self):
        p,r=fixture();m=p['approved_mining_manifest']['payload'];m['heldout_indices']=[1]
        cohort=p['training_cohort']['payload'];cohort['approved_manifest_payload_sha256']=c.digest(m)
        with self.assertRaisesRegex(ValueError,'heldout'):c.validate_cohort(cohort,m)

    def test_cap1024_cannot_relabel_as_profile2048(self):
        p,r=fixture();p['suites'][0]=copy.deepcopy(p['suites'][0]);p['suites'][0]['harness']['max_output_tokens']=1024
        with self.assertRaises(ValueError):c.validate_plan(p,AUTH)

    def test_duplicate_missing_and_seed_mismatch_rejected(self):
        for alter in ('duplicate','missing','seed'):
            p,r=fixture()
            if alter=='duplicate':r[-1]=r[0]
            elif alter=='missing':r.pop()
            else:r[0]['seed']+=1
            with self.assertRaises(ValueError):c.summarize(p,r)

    def test_errors_not_dropped_and_native_unresolved_counted(self):
        p,r=fixture();r[0].update(classification='error',reward=0,native_graded=False,raw_turns=[],terminal_eos=False)
        r[1].update(classification='unresolved',reward=0)
        s=c.summarize(p,r)
        self.assertEqual((s['requested'],s['completed'],s['errors'],s['unresolved'],s['solved']),(128,127,1,1,94))

    def test_actual_prompt_mismatch_disables_comparison(self):
        p,r=fixture();other=copy.deepcopy(r);other[7]['raw_turns'][0]['prompt_tokens'].append(777)
        self.assertFalse(c.paired_comparability(p,r,p,other)['comparable'])

    def test_actual_raw_seed_mismatch_rejected(self):
        p,r=fixture();r[0]['raw_turns'][0]['seed']+=1
        with self.assertRaises(ValueError):c.summarize(p,r)

    def test_exposure_solved_partition_and_fixed_mask(self):
        p,r=fixture();p['exposure_task_counts']={str(i):int(i<80) for i in range(128)}
        p['fixed_initial_trained_mask']=dict(indices=list(range(70)),checkpoint=c.digest('initial'),optimizer_step=67)
        e=dict(scope='current-training-run-through-checkpoint',trained_tasks=80,untrained_tasks=48,unknown_tasks=0,task_exposures=80)
        s=c.summarize(p,r,exposure=e)
        self.assertEqual(s['exposure_groups']['trained']['solved'],80)
        self.assertEqual(s['exposure_groups']['untrained']['solved'],16)
        self.assertEqual(s['fixed_initial_trained']['count'],70)

    def test_four_update_coalescing(self):
        self.assertFalse(c.choose_latest(67,64));self.assertTrue(c.choose_latest(68,64));self.assertTrue(c.choose_latest(75,64))
        self.assertFalse(c.choose_latest(75,75))

    def test_profile_mismatch_never_paired(self):
        p,r=fixture();other=copy.deepcopy(p);other['profile_sha256']=c.digest('different')
        with self.assertRaises(ValueError):c.paired_comparability(p,r,other,r)

    def test_native_neutral_is_unresolved_only_for_original_binary_guard(self):
        native=[dict(classification='neutral',done=True,reward=0)]
        self.assertEqual(c.failed_rollout_classification(native,ValueError('binary native math outcome')),'unresolved')
        self.assertEqual(c.failed_rollout_classification(native,RuntimeError('grading failed')),'error')
        self.assertEqual(c.failed_rollout_classification([],ValueError('binary native math outcome')),'error')
        native[0]['done']=False
        self.assertEqual(c.failed_rollout_classification(native,ValueError('binary native math outcome')),'error')

    def test_pairing_retains_different_checkpoint_exposure_groups(self):
        p,r=fixture();other=copy.deepcopy(p)
        for plan,trained in ((p,0),(other,80)):
            plan['exposure_task_counts']={str(i):int(i<trained) for i in range(128)}
            plan['exposure']=dict(scope='current-training-run-through-checkpoint',trained_tasks=trained,
                                  untrained_tasks=128-trained,unknown_tasks=0,task_exposures=trained)
        self.assertTrue(c.paired_comparability(p,r,other,r)['comparable'])


if __name__=='__main__':unittest.main()
