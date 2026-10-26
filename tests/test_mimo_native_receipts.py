from copy import deepcopy
import unittest

from subnet.complete_group_sampling import digest, canonical
from subnet.complete_group_native_receipts import compact_native_grades, expand_native_grades, FULL, COMPACT


def fixture(prompt_tokens=3, output_tokens=5):
    prompt = [248319] * prompt_tokens
    output = [248319] * (output_tokens-1) + [248046]
    batch = dict(miner='a'*64, task_hash='b'*64,
        rollouts=[dict(seed=i, task_hash='b'*64,
            turns=[dict(prompt=prompt, output=output)]) for i in range(8)])
    grade = dict(version=FULL, batch_sha256=digest(batch), manifest_sha256='c'*64,
        miner='a'*64, task_hash='b'*64, prompt=prompt, sampling_verified=False,
        results=[dict(nonce=i, classification='positive' if i%2 else 'negative', reward=float(i%2),
            stop_reason='eos', output_sha256=digest(output), native_classification='positive' if i%2 else 'negative',
            done=True, observations=[], prompt=prompt, output=output, task_hash='b'*64) for i in range(8)])
    return batch, grade


class CompactNativeReceiptsTests(unittest.TestCase):
    def test_roundtrip_retains_trusted_labels_and_original_tokens(self):
        batch, grade = fixture()
        compact = compact_native_grades(grade)
        self.assertEqual(compact['version'], COMPACT)
        self.assertNotIn('output', compact['results'][0])
        self.assertNotIn('prompt', compact['results'][0])
        self.assertEqual(expand_native_grades(compact, batch), grade)
        self.assertFalse(compact['sampling_verified'])

    def test_compact_receipt_cannot_rehydrate_substituted_tokens_or_nonce(self):
        batch, grade = fixture()
        for mutate in (lambda b:b['rollouts'][0]['turns'][0]['output'].__setitem__(0, 42),
                       lambda b:b['rollouts'][0].__setitem__('seed', 1),
                       lambda b:b.__setitem__('miner', 'd'*64)):
            bad=deepcopy(batch); mutate(bad)
            with self.assertRaises(ValueError):expand_native_grades(compact_native_grades(grade), bad)
        compact=compact_native_grades(grade)
        compact['results'][0]['output_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'token hash'):
            expand_native_grades(compact,batch)

    def test_malformed_native_nonce_or_output_digest_rejected_before_compaction(self):
        _, grade=fixture()
        grade['results'][0]['nonce']=True
        with self.assertRaisesRegex(ValueError,'nonce'):compact_native_grades(grade)
        _, grade=fixture()
        grade['results'][0]['output_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'native prompt/task/output'):compact_native_grades(grade)

    def test_full_current_population_fits_dispatch_bound_at_maximal_token_ids(self):
        # Current 12K-context/8K-output math policy leaves <=4096 prompt tokens.
        # Every 256 UIDs may submit nine groups. Selected and excluded entries
        # together retain that entire population, but not the 151M output tokens.
        _, grade=fixture(prompt_tokens=4096, output_tokens=8192)
        compact=compact_native_grades(grade)
        row=dict(miner='a'*64,env_id='math',index=1,document=dict(slot=0,sha256='c'*64,
            key='private/epoch/complete-group-snapshots/'+'a'*64+'/training/0.json'),
            batch_sha256=grade['batch_sha256'],native_grades=compact,
            assurance='natively-graded-sampling-unaudited')
        # Exact JSON framing: repeated object lists add one comma per entry.
        worst_size=(len(canonical(row))+1)*256*9+4096
        self.assertLess(worst_size,128*1024**2)
        self.assertGreater(len(canonical(grade))*256*9,256*1024**2)


if __name__=='__main__':unittest.main()
