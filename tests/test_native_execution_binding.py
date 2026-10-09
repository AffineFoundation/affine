import copy
import types
import unittest
from ops.native_execution_binding import ALLOWED, ADDED, VERSION, assert_loader, validate_declaration


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.before = {name: 'a'*64 for name in ALLOWED}
        self.before['subnet/native_math_prompt.py'] = 'c'*64
        self.after = dict(self.before, **{name: 'b'*64 for name in ALLOWED})
        self.after.update({name:'b'*64 for name in ADDED})
        self.old = dict(execution_root='/old', source_files={'subnet/native_math_prompt.py': 'c'*64}, limits={'workers': 16}, checkpoint='bound')
        self.new = dict(self.old, execution_root='/new')
        self.repair = dict(version=VERSION, previous_authorization={}, previous_execution_root='/old', previous_files=self.before, changed_files={name:'b'*64 for name in ALLOWED|ADDED})
    def run_check(self):
        validate_declaration(self.repair, self.old, self.new, '/new', self.after)
    def test_exact(self): self.run_check()
    def test_stale_root(self):
        self.new['execution_root']='/old'
        with self.assertRaises(ValueError):self.run_check()
    def test_wrong_previous(self):
        self.repair['previous_execution_root']='/other'
        with self.assertRaises(ValueError):self.run_check()
    def test_limits_change(self):
        self.new=copy.deepcopy(self.new);self.new['limits']['workers']=4
        with self.assertRaises(ValueError):self.run_check()
    def test_checkpoint_change(self):
        self.new['checkpoint']='different'
        with self.assertRaises(ValueError):self.run_check()
    def test_scientific_mutation(self):
        self.after['subnet/native_math_prompt.py']='d'*64
        with self.assertRaises(ValueError):self.run_check()
    def test_extra_file(self):
        self.after['subnet/extra.py']='b'*64
        with self.assertRaises(ValueError):self.run_check()
    def test_missing_file(self):
        del self.after['subnet/native_math_prompt.py']
        with self.assertRaises(ValueError):self.run_check()
    def test_wrong_reviewed_hash(self):
        self.repair['changed_files']=dict(self.repair['changed_files']);self.repair['changed_files']['subnet/backend_jobs.py']='e'*64
        with self.assertRaises(ValueError):self.run_check()
    def test_unreviewed_change(self):
        self.before['subnet/unrelated.py']='a'*64;self.after['subnet/unrelated.py']='b'*64
        with self.assertRaises(ValueError):self.run_check()
    def test_loader_exact(self):
        assert_loader(types.SimpleNamespace(__file__='/new/subnet/native_math_prompt.py'),self.new)
    def test_loader_stale(self):
        with self.assertRaises(ValueError):assert_loader(types.SimpleNamespace(__file__='/old/subnet/native_math_prompt.py'),self.new)


if __name__=='__main__':unittest.main()
