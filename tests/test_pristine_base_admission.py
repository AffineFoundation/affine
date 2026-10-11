"""Prospective admission rejects validly signed but non-upstream checkpoints.

Signatures here use an isolated unit-test key, never a production authority.
Local readback controls use tiny synthetic files rather than loading a model.
"""
import base64
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey

from ops import pristine_base_admission as admission
from subnet.storage import canonical, sha


# Public filemap retained by the original6493-upstream-provenance-v1 report.
TRAINED_FILES = {
    'chat_template.jinja': 'c564d12c9977b8942423753b0c409a11f50142508ced170e9fe1c2f5b803cf70',
    'config.json': '8027e74d1df10c4d112e7eec34db259aedefac4fe92dfa926b2a1fb8fcad6603',
    'generation_config.json': '9a464692406690ab0d54da69ee1b1a88ef80ea5f5cfad9d84180f2bc7f53259c',
    'model-00001-of-00004.safetensors': '716feb52e9a6f23150aa76d2024242532f6f92c2e9f4fbfe23e03567eb001702',
    'model-00002-of-00004.safetensors': '1986c28f642577310bd77995b1342ae246109f69aa0f1058e894dfc0c9a8f98f',
    'model-00003-of-00004.safetensors': 'fa833a2debb441befa1e2c8bdab6d9ff8a041f766aea639a4eb64fb1b96f2a37',
    'model-00004-of-00004.safetensors': 'e1bbb7141acbdb68e3f80ee83c6cb38f236ff589e5e3e8a999e0cc81bec42dc4',
    'model.safetensors.index.json': '46392f54c0347692b06f127397c6359a4958b15da1e095bad9cf211fa07023d2',
    'tokenizer.json': '3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8',
    'tokenizer_config.json': '1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95',
}


class PristineAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey(bytes(range(32)))
        self.authority = self.key.verify_key.encode().hex()

    def envelope(self, payload):
        return dict(payload=payload, signer=self.authority,
                    signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())

    def fixture(self, files=None):
        files = dict(admission.PRISTINE_FILES if files is None else files)
        checkpoint = dict(id=sha(canonical(files)), files=files)
        config = dict(model_id=admission.MODEL_ID, model_revision=admission.MODEL_REVISION,
                      initial_checkpoint=copy.deepcopy(checkpoint))
        return config, self.envelope(checkpoint)

    def check(self, config, descriptor, **kwargs):
        return admission.validate_pristine_base(config, descriptor, self.authority, **kwargs)

    def test_exact_upstream_descriptor_accepts_without_mutating_inputs(self):
        config, descriptor = self.fixture()
        config['initial_checkpoint']['base_url'] = 'https://example.invalid/cache'
        before = copy.deepcopy((config, descriptor))
        result = self.check(config, descriptor)
        self.assertEqual(result['checkpoint_id'], admission.CHECKPOINT_ID)
        self.assertTrue(result['pristine_descriptor_verified'])
        self.assertFalse(result['local_bytes_verified'])
        self.assertFalse(result['execute_allowed'])
        self.assertEqual((config, descriptor), before)

    def test_actual_trained_checkpoint_with_same_model_name_is_rejected(self):
        config, descriptor = self.fixture(TRAINED_FILES)
        self.assertEqual(config['initial_checkpoint']['id'],
                         '6493a901bd009f0800d5eed97d19aee78947cc586afeb27abfb2c72032ad1924')
        with self.assertRaisesRegex(ValueError, 'independently pinned pristine'):
            self.check(config, descriptor)

    def test_validly_signed_self_consistent_mutations_are_not_pristine(self):
        for change in ('missing', 'extra', 'swapped', 'trained_shard', 'tokenizer'):
            with self.subTest(change=change):
                files = dict(admission.PRISTINE_FILES)
                first, second = 'model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors'
                if change == 'missing':
                    del files[first]
                elif change == 'extra':
                    files['adapter_model.safetensors'] = '0' * 64
                elif change == 'swapped':
                    files[first], files[second] = files[second], files[first]
                else:
                    files[first if change == 'trained_shard' else 'tokenizer.json'] = '0' * 64
                config, descriptor = self.fixture(files)
                with self.assertRaisesRegex(ValueError, 'independently pinned pristine'):
                    self.check(config, descriptor)

    def test_claiming_pristine_id_cannot_hide_trained_filemap(self):
        config, descriptor = self.fixture(TRAINED_FILES)
        config['initial_checkpoint']['id'] = admission.CHECKPOINT_ID
        descriptor = self.envelope(copy.deepcopy(config['initial_checkpoint']))
        with self.assertRaisesRegex(ValueError, 'filemap identity mismatch'):
            self.check(config, descriptor)

    def test_config_cannot_substitute_filemap_behind_pristine_descriptor(self):
        for field, value in (('id', '0' * 64), ('files', TRAINED_FILES)):
            with self.subTest(field=field):
                config, descriptor = self.fixture()
                config['initial_checkpoint'][field] = value
                with self.assertRaisesRegex(ValueError, 'configured initial checkpoint'):
                    self.check(config, descriptor)

    def test_missing_config_filemap_is_rejected(self):
        config, descriptor = self.fixture()
        del config['initial_checkpoint']['files']
        with self.assertRaisesRegex(ValueError, 'configured initial checkpoint'):
            self.check(config, descriptor)

    def test_wrong_model_or_revision_is_rejected(self):
        for field in ('model_id', 'model_revision'):
            with self.subTest(field=field):
                config, descriptor = self.fixture()
                config[field] = 'unapproved'
                with self.assertRaisesRegex(ValueError, 'model name and upstream revision'):
                    self.check(config, descriptor)

    def test_wrong_authority_and_corrupt_signature_fail(self):
        config, descriptor = self.fixture()
        with self.assertRaisesRegex(ValueError, 'authority mismatch'):
            admission.validate_pristine_base(config, descriptor, '0' * 64)
        descriptor['signature'] = base64.b64encode(bytes(64)).decode()
        with self.assertRaises(BadSignatureError):
            self.check(config, descriptor)

    def test_cli_emits_only_unsigned_non_execution_result(self):
        config, descriptor = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_bytes(canonical(config))
            (root / 'checkpoint.json').write_bytes(canonical(descriptor))
            output = io.StringIO()
            with redirect_stdout(output):
                admission.main(['--config', str(root / 'config.json'), '--descriptor',
                                str(root / 'checkpoint.json'), '--authority', self.authority])
            result = json.loads(output.getvalue())
            self.assertFalse(result['execute_allowed'])
            self.assertFalse(result['local_bytes_verified'])
            self.assertNotIn('signature', result)
            self.assertEqual({p.name for p in root.iterdir()}, {'config.json', 'checkpoint.json'})

    def test_local_bytes_missing_extra_swapped_and_symlink_controls(self):
        data = {'config.json': b'{}', 'a.safetensors': b'first', 'b.safetensors': b'second'}
        files = {name: sha(value) for name, value in data.items()}
        # Tiny fixture trust anchor applies only inside this local-I/O control.
        with patch.object(admission, 'PRISTINE_FILES', files), \
                patch.object(admission, 'CHECKPOINT_ID', sha(canonical(files))):
            for change in ('none', 'missing', 'extra', 'swapped', 'symlink', 'wrong_path'):
                with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    for name, value in data.items():
                        (root / name).write_bytes(value)
                    config, descriptor = self.fixture(files)
                    config['initial_checkpoint_path'] = str(root)
                    if change == 'missing':
                        (root / 'a.safetensors').unlink()
                    elif change == 'extra':
                        (root / 'unapproved.log').write_bytes(b'extra')
                    elif change == 'swapped':
                        (root / 'a.safetensors').write_bytes(data['b.safetensors'])
                        (root / 'b.safetensors').write_bytes(data['a.safetensors'])
                    elif change == 'symlink':
                        (root / 'a.safetensors').unlink()
                        (root / 'a.safetensors').symlink_to(root / 'b.safetensors')
                    elif change == 'wrong_path':
                        config['initial_checkpoint_path'] = str(root / 'another-cache')
                    if change == 'none':
                        self.assertTrue(self.check(config, descriptor, checkpoint_path=root)['local_bytes_verified'])
                    else:
                        with self.assertRaises(ValueError):
                            self.check(config, descriptor, checkpoint_path=root)


if __name__ == '__main__':
    unittest.main()
