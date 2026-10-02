import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_math_corpus_assets import fixture
from subnet.task_assets import bindings,hydrate_manifest

class TaskAssets(unittest.TestCase):
    def manifest(self):
        body,binding,spec=fixture()
        url='https://a.r2.cloudflarestorage.com/x?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=a'
        manifest={'environments':[{'spec':vars(spec)}], 'task_assets':{binding['sha256']:{
            'read_url':url,'compressed_sha256':binding['compressed_sha256'],'compressed_size':binding['compressed_size']}}}
        return body,binding,manifest
    def test_exact_admitted_registry_is_required_before_fetch(self):
        body,binding,manifest=self.manifest()
        for mutation in ('missing','extra','digest','size','url'):
            changed=copy.deepcopy(manifest);row=changed['task_assets'][binding['sha256']]
            if mutation=='missing':changed['task_assets']={}
            elif mutation=='extra':changed['task_assets']['f'*64]=row
            elif mutation=='digest':row['compressed_sha256']='0'*64
            elif mutation=='size':row['compressed_size']=True
            else:row['read_url']='https://untrusted.invalid/data'
            with tempfile.TemporaryDirectory() as root,self.assertRaises(ValueError):
                hydrate_manifest(root,changed,lambda *a:self.fail('unauthorized fetch'))
    def test_separate_asset_cache_preserves_source_tree(self):
        body,binding,manifest=self.manifest()
        with tempfile.TemporaryDirectory() as cache,tempfile.TemporaryDirectory() as source:
            admitted=Path(source)/'reviewed.py';admitted.write_text('pass\n')
            result=hydrate_manifest(cache,manifest,lambda *a:body)
            self.assertEqual(result[binding['sha256']],Path(cache)/binding['path'])
            self.assertEqual(list(Path(source).iterdir()),[admitted])
            self.assertEqual(bindings(manifest)[binding['sha256']],binding)
    def test_bootstrap_uses_admitted_code_and_stdin_for_capabilities(self):
        body,binding,manifest=self.manifest()
        from subnet.source_bootstrap import hydrate_task_assets
        with tempfile.TemporaryDirectory() as source,tempfile.TemporaryDirectory() as cache:
            with patch('subprocess.run') as run,patch.dict('os.environ',{}):
                run.return_value.returncode=0
                hydrate_task_assets(source,manifest,cache)
                args,kwargs=run.call_args
                self.assertNotIn(manifest['task_assets'][binding['sha256']]['read_url'],' '.join(args[0]))
                self.assertIn(b'X-Amz-Signature',kwargs['input'])
                import os
                self.assertEqual(os.environ['AFFINE_MATH_CORPUS_ASSET_ROOT'],str(Path(cache)/'task-assets'))
