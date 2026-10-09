import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ops/trainer_lifecycle"))
import trainer_reset_entry as entry


class EntryTests(unittest.TestCase):
    def test_absent_sibling_retains_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reset.ROOT-SIGNED.private.json';seen=[]
            helper=types.SimpleNamespace(_read=lambda p:{'reset':True},
                install_for_retirement=lambda *a,**k:seen.append(k))
            with patch.object(entry,'helper',return_value=helper):entry.install_in_child(path,'authority',directory,role='retirement')
            self.assertEqual(seen,[dict(recovery_envelope=None)])

    def test_exact_sibling_is_passed_for_authenticated_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reset.ROOT-SIGNED.private.json'
            sibling=path.with_name('retry.ROOT-SIGNED.private.json');sibling.touch();seen=[]
            def read(p):seen.append(p);return {'path':str(p)}
            helper=types.SimpleNamespace(_read=read,install_for_retirement=lambda *a,**k:k)
            with patch.object(entry,'helper',return_value=helper):result=entry.install_in_child(path,'authority',directory,role='retirement')
            self.assertEqual(result['recovery_envelope'],{'path':str(sibling)})
            self.assertEqual(seen,[path,sibling])

    def test_wrong_sibling_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reset.ROOT-SIGNED.private.json';path.with_name('retry.ROOT-SIGNED.private.json').touch()
            def validate(*a,**kw):raise ValueError('invalid authenticated retry')
            helper=types.SimpleNamespace(_read=lambda p:{'invalid':True},install_for_retirement=validate)
            with patch.object(entry,'helper',return_value=helper),self.assertRaisesRegex(ValueError,'invalid authenticated retry'):
                entry.install_in_child(path,'authority',directory,role='retirement')


if __name__=='__main__':unittest.main()
