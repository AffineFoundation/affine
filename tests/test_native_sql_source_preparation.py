import hashlib,tempfile,unittest
from pathlib import Path
import shutil,subprocess
from ops.prepare_native_sql_diversity_source import BASE,prepare,transformed
ROOT=Path(__file__).resolve().parents[1]
class SourcePreparation(unittest.TestCase):
    def fixture(self,base):
        source=base/'source';(source/'subnet').mkdir(parents=True)
        for n in BASE:shutil.copy2(ROOT/'subnet'/n,source/'subnet'/n)
        (source/'state/private').mkdir(parents=True);(source/'state/private/seed').write_text('SECRET_SHOULD_NOT_COPY')
        subprocess.run(['git','init','-q',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'add','subnet'],check=True)
        subprocess.run(['git','-C',str(source),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','source'],check=True)
        return source
    def test_exact_reviewed_feature_bytes_without_changing_source(self):
        with tempfile.TemporaryDirectory() as td:
            source=self.fixture(Path(td));before={n:(source/'subnet'/n).read_bytes() for n in BASE};out=Path(td)/'prepared';r=prepare(source,out)
            for n,v in before.items():self.assertEqual((source/'subnet'/n).read_bytes(),v)
            self.assertFalse((out/'state').exists());self.assertFalse(r['model_or_native_execution']);self.assertIn('public-sql-candidates',(out/'subnet/harness.py').read_text());self.assertIn('native_sql_controlled',(out/'subnet/environments.py').read_text())
    def test_altered_base_rejected_before_any_destination_write(self):
        with tempfile.TemporaryDirectory() as td:
            source=self.fixture(Path(td));(source/'subnet/harness.py').write_text('MUTATED');out=Path(td)/'prepared'
            with self.assertRaises(ValueError):prepare(source,out)
            self.assertFalse(out.exists())
    def test_existing_destination_and_relative_operator_path_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            source=self.fixture(Path(td));out=Path(td)/'prepared';out.mkdir()
            with self.assertRaises(ValueError):prepare(source,out)
            with self.assertRaises(ValueError):transformed(source,'miner-uploaded-relative-path')
    def test_symlink_parent_aliasing_live_tree_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=self.fixture(base);alias=base/'alias';alias.symlink_to(source,target_is_directory=True)
            before=(source/'subnet/environments.py').read_bytes()
            with self.assertRaises(ValueError):prepare(source,alias/'prepared')
            self.assertFalse((source/'prepared').exists());self.assertEqual((source/'subnet/environments.py').read_bytes(),before)
    def test_unqualified_symlink_to_private_asset_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=self.fixture(base);out=base/'prepared'
            (source/'subnet/private-link.py').symlink_to(source/'state/private/seed')
            with self.assertRaises(ValueError):prepare(source,out)
            self.assertFalse(out.exists())
    def test_vendor_symlink_directory_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=self.fixture(base);out=base/'prepared';(source/'prototype').mkdir()
            (source/'prototype/vendor').symlink_to(source/'state/private',target_is_directory=True)
            with self.assertRaises(ValueError):prepare(source,out)
            self.assertFalse(out.exists())
    def test_vendor_parent_alias_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=self.fixture(base);external=base/'external';(external/'vendor').mkdir(parents=True)
            (source/'prototype').symlink_to(external,target_is_directory=True);out=base/'prepared'
            with self.assertRaises(ValueError):prepare(source,out)
            self.assertFalse(out.exists())
    def test_untracked_regular_private_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);source=self.fixture(base);(source/'subnet/private.py').write_text('PRIVATE_SECRET');out=base/'prepared'
            with self.assertRaises(ValueError):prepare(source,out)
            self.assertFalse(out.exists())
    def test_custom_operator_path_is_source_bound(self):
        with tempfile.TemporaryDirectory() as td:
            source=self.fixture(Path(td));default=transformed(source,'/root/native-sql-common-v1/operator/private-tasks.json');other=transformed(source,'/operator/private/tasks.json')
            self.assertNotEqual(default['environments.py'],other['environments.py']);self.assertEqual(default['harness.py'],other['harness.py'])
if __name__=='__main__':unittest.main()
