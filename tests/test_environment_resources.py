import json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from subnet.environment_resources import (collect_dependencies,verify_dependencies,export_dependencies,collect_task_resources,export_task_resources,materialize,bind_task_resources,digest,canonical_task_identity)

class ResourceTests(unittest.TestCase):
 def test_actual_installed_package_bundle(self):
  manifest=collect_dependencies(['packaging']);self.assertTrue(verify_dependencies(manifest))
  with tempfile.TemporaryDirectory() as directory:
   archive=Path(directory)/'deps.zip';receipt=export_dependencies(manifest,archive)
   root=materialize(manifest,archive,Path(directory)/'isolated',archive_sha256=receipt['archive_sha256'])
   for row in manifest['packages']:
    for path,sha in row['files'].items():self.assertEqual(digest((root/path).read_bytes()),sha)
   # The isolated transport remains visible to Python distribution identity checks.
   from importlib.metadata import distributions
   found=list(distributions(path=[str(root)]));self.assertEqual(found[0].version,manifest['packages'][0]['version'])

 def test_changed_provider_source_and_version_fail_closed(self):
  with tempfile.TemporaryDirectory() as directory:
   file=Path(directory)/'provider.py';file.write_text('VALUE = 1\n')
   class Provider:
    metadata={'Name':'provider'};version='1';files=['provider.py']
    def locate_file(self,n):return Path(directory)/n
    def read_text(self,n):return None
   fake=Provider()
   from importlib.util import spec_from_file_location
   with patch('subnet.environment_resources.metadata.distribution',return_value=fake),patch('subnet.environment_resources.util.find_spec',return_value=spec_from_file_location('provider',file)):
    manifest=collect_dependencies(['provider']);self.assertTrue(verify_dependencies(manifest))
    file.write_text('VALUE = 2\n')
    with self.assertRaisesRegex(ValueError,'source mismatch'):verify_dependencies(manifest)
    file.write_text('VALUE = 1\n');fake.version='2'
    with self.assertRaisesRegex(ValueError,'version mismatch'):verify_dependencies(manifest)

 def test_unlisted_module_and_shadowed_import_are_rejected(self):
  import sys
  with tempfile.TemporaryDirectory() as directory:
   base=Path(directory)/'original';package=base/'proof_provider_fixture';package.mkdir(parents=True)
   (package/'__init__.py').write_text('VALUE = 1\n')
   class Provider:
    metadata={'Name':'proof-provider-fixture'};version='1';files=['proof_provider_fixture/__init__.py']
    def locate_file(self,n):return base/n
    def read_text(self,n):return None
   sys.path.insert(0,str(base))
   try:
    with patch('subnet.environment_resources.metadata.distribution',return_value=Provider()):
     manifest=collect_dependencies(['proof-provider-fixture']);self.assertTrue(verify_dependencies(manifest))
     # This file is NOT in RECORD. It still changes the enforced source map.
     injected=package/'unlisted.py';injected.write_text('MALICIOUS = True\n')
     with self.assertRaisesRegex(ValueError,'source mismatch'):verify_dependencies(manifest)
     injected.unlink()
     shadow=Path(directory)/'shadow';(shadow/'proof_provider_fixture').mkdir(parents=True)
     (shadow/'proof_provider_fixture'/'__init__.py').write_text('VALUE = 999\n')
     sys.path.insert(0,str(shadow))
     try:
      with self.assertRaisesRegex(ValueError,'import resolution mismatch'):verify_dependencies(manifest)
     finally:sys.path.pop(0)
   finally:sys.path.remove(str(base))

 def test_original_resource_roundtrip_private_audience_and_mapping(self):
  with tempfile.TemporaryDirectory() as directory:
   source=Path(directory)/'original';(source/'tests').mkdir(parents=True);(source/'environment').mkdir()
   (source/'tests'/'test.sh').write_text('test -f /app/answer\n');(source/'environment'/'Dockerfile').write_text('FROM ubuntu:22.04\n')
   manifest=collect_task_resources(source,origin={'git':'original-repo','commit':'a'*40});self.assertEqual(manifest['audience'],'verifier')
   archive=Path(directory)/'resource.zip';receipt=export_task_resources(manifest,source,archive)
   root=materialize(manifest,archive,Path(directory)/'portable',archive_sha256=receipt['archive_sha256'])
   mapped=bind_task_resources({'task_dir':'/old/absolute/path','name':'original-index-0'},manifest,root)
   self.assertEqual(mapped['task_dir'],str(root.resolve()));self.assertEqual(mapped['name'],'original-index-0')
   data={'task_dir':'/left/host/task','name':'original-index-0'}
   a=canonical_task_identity(data,{},manifest,environment_version='next-v2')
   b=canonical_task_identity(dict(data,task_dir='/right/host/task'),{},manifest,environment_version='next-v2')
   self.assertEqual(a,b)
   self.assertNotEqual(a,canonical_task_identity(dict(data,name='different-original-index'),{},manifest,environment_version='next-v2'))
   self.assertEqual((root/'tests'/'test.sh').read_text(),'test -f /app/answer\n')
   with self.assertRaisesRegex(ValueError,'overwrite'):materialize(manifest,archive,root,archive_sha256=receipt['archive_sha256'])

 def test_archive_tamper_and_traversal_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   source=Path(directory)/'original';source.mkdir();(source/'file').write_text('original')
   manifest=collect_task_resources(source,origin={'commit':'a'*40});archive=Path(directory)/'bundle.zip';receipt=export_task_resources(manifest,source,archive)
   with zipfile.ZipFile(archive,'a') as z:z.writestr('files/../../escape','evil')
   with self.assertRaisesRegex(ValueError,'archive hash'):materialize(manifest,archive,Path(directory)/'out',archive_sha256=receipt['archive_sha256'])
   with self.assertRaisesRegex(ValueError,'unexpected'):materialize(manifest,archive,Path(directory)/'out',archive_sha256=digest(archive.read_bytes()))
   self.assertFalse((Path(directory)/'escape').exists())

if __name__=='__main__':unittest.main()
