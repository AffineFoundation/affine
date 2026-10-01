import hashlib
import io
import tarfile
import unittest
from ops.archive_harness_identity import identity

class ArchiveHarnessIdentity(unittest.TestCase):
    def archive(self,source,companion=None,duplicate=False):
        files={'subnet/harness.py':source.encode()}
        if companion is not None:files['subnet/native_mrcr_public_policy.py']=companion
        output=io.BytesIO()
        with tarfile.open(fileobj=output,mode='w:gz') as archive:
            for name,value in files.items():
                row=tarfile.TarInfo(name);row.size=len(value);archive.addfile(row,io.BytesIO(value))
            if duplicate:
                name='subnet/harness.py';value=files[name];row=tarfile.TarInfo('./'+name);row.size=len(value);archive.addfile(row,io.BytesIO(value))
        return output.getvalue(),{name:hashlib.sha256(value).hexdigest() for name,value in files.items()},files
    def source(self,expression='Path(__file__).read_bytes()'):
        return 'def source_hash():\n    return hashlib.sha256('+expression+').hexdigest()\n'
    def test_original_single_module_identity(self):
        body,pins,files=self.archive(self.source())
        self.assertEqual(identity(body,pins),pins['subnet/harness.py'])
    def test_companion_identity_from_actual_pinned_bytes(self):
        body,pins,files=self.archive(self.source('Path(__file__).read_bytes()+(Path(__file__).parent / "native_mrcr_public_policy.py").read_bytes()'),b'public policy')
        self.assertEqual(identity(body,pins),hashlib.sha256(b''.join(files.values())).hexdigest())
    def test_wrong_missing_or_unpinned_module_refused(self):
        body,pins,_=self.archive(self.source(),b'public policy')
        for bad in [{},{**pins,'subnet/harness.py':'0'*64}]:
            with self.assertRaises(ValueError):identity(body,bad)
        self.assertEqual(identity(body,{'subnet/harness.py':pins['subnet/harness.py']}),pins['subnet/harness.py'])
        used=self.source('Path(__file__).read_bytes()+(Path(__file__).parent / "native_mrcr_public_policy.py").read_bytes()')
        body,pins,_=self.archive(used,b'public policy')
        with self.assertRaises(ValueError):identity(body,{'subnet/harness.py':pins['subnet/harness.py']})
        body,pins,_=self.archive(self.source('Path(__file__).read_bytes()+(Path(__file__).parent / "sample_harness.py").read_bytes()'))
        with self.assertRaises(ValueError):identity(body,pins)
    def test_duplicate_alias_module_refused(self):
        body,pins,_=self.archive(self.source(),duplicate=True)
        with self.assertRaises(ValueError):identity(body,pins)
    def test_unreviewed_hash_or_execution_shape_refused(self):
        sources=[self.source('Path("/tmp/other.py").read_bytes()'),self.source('Path(__file__).read_bytes()+Path("../../secret").read_bytes()'),self.source('b"pretend"'),self.source('Path(__file__).read_bytes()+Path(__file__).read_bytes()'),self.source().replace('sha256','sha1'),self.source()+'def source_hash():\n    return "fake"\n',self.source().replace('    return','    side_effect()\n    return')]
        for source in sources:
            with self.subTest(source=source),self.assertRaises(ValueError):identity(*self.archive(source)[:2])

if __name__=='__main__':unittest.main()
