import copy,json,time,tempfile,unittest,hashlib,os
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
import test_backend_jobs as fixture
from subnet import source_bootstrap as bootstrap,backend_jobs
CAPTURE=dict(version='bounded-parallel-token-capture-v2',workers=8,max_document_bytes=2000000,max_inflight_bytes=16000000,completion_order='first-completed',journal_version='fsynced-per-epoch-capture-v1',state_checkpoint_documents=16)
FROZEN_ARCHIVE=Path(os.environ.get('AFFINE_F213_TEST_ARCHIVE','/home/const/subnet120-rewrite/state/root-audits/ordinary-v3-bounded-confirmation-source-preparation-20261006-v1/source-candidate.REVIEW-ONLY.tar.gz'))
class FrozenMetadata(unittest.TestCase):
 def test_signed_operator_capture_metadata_does_not_change_backend_numerical_authorization(self):
  f=fixture.BackendJobAuthorization();f.setUp();m=copy.deepcopy(f.manifest);m['learner_capture_policy']=CAPTURE;m['source_bundle']={'sha256':'f21373d7ccb167bcd868f5ed03ce9ac7ef567d894e8ecc9e61f5d7f8645b67b8'};f.job['manifest']=f.sign(m)
  job,actual=backend_jobs.validate(f.sign(f.job),f.authority,now=50)
  self.assertEqual(actual['learner_capture_policy'],CAPTURE);self.assertEqual(actual['numerical_policy'],f.manifest['numerical_policy']);self.assertEqual(job['source_files'],f.job['source_files'])
  envelope=f.sign(f.job);envelope['payload']['manifest']['payload']['learner_capture_policy']['workers']=16
  with self.assertRaises(Exception):backend_jobs.validate(envelope,f.authority,now=50)
 @unittest.skipUnless(FROZEN_ARCHIVE.is_file(),'operator integration requires the immutable f213 archive; set AFFINE_F213_TEST_ARCHIVE')
 def test_frozen_bootstrap_admits_actual_f213_and_forwards_original_source_pin(self):
  f=fixture.BackendJobAuthorization();f.setUp();url='https://account.r2.cloudflarestorage.com/bucket/current?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=test'
  archive=FROZEN_ARCHIVE
  body=archive.read_bytes();sha=hashlib.sha256(body).hexdigest();self.assertEqual(sha,'f21373d7ccb167bcd868f5ed03ce9ac7ef567d894e8ecc9e61f5d7f8645b67b8')
  m=dict(epoch='prospective-local-capture-control',deadline=time.time()+120,transport_policy='direct-r2-v1',learner_capture_policy=CAPTURE,source_bundle=dict(sha256=sha,size=len(body),url=url))
  pointer=dict(epoch=m['epoch'],manifest_url=url,transport_policy='direct-r2-v1')
  calls=iter([json.dumps(f.sign(pointer)).encode(),json.dumps(f.sign(m)).encode()]);actual=bootstrap.manifest(url,f.authority,lambda u,n:next(calls));self.assertEqual(actual['learner_capture_policy'],CAPTURE)
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);cap=root/'dummy-cap';cap.write_text('{}');executed=[]
   with patch.object(bootstrap,'manifest',return_value=actual),patch.object(bootstrap,'download',return_value=body),patch.object(bootstrap,'execute',side_effect=lambda source,args:executed.append((source,args))):
    bootstrap.main(['--authority',f.authority,'--current-url',url,'--cap-file',str(cap),'--source-cache',str(root/'cache'),'--state',str(root/'state'),'--once'])
   self.assertEqual(len(executed),1);source,args=executed[0];admitted=bootstrap.admitted_files(body,actual['source_bundle']);bootstrap.verify_cache(source,admitted);self.assertEqual(len(admitted),2121);self.assertEqual(args[args.index('--source-bundle-sha256')+1],sha);self.assertIn('--cap-file',args)
