import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import requests
from subnet.backend_jobs import ArtifactRejected,get_object,prefetched_training_submissions
from subnet.cache_lifecycle import CacheLifecycle

URL='https://bucket.example.r2.cloudflarestorage.com/{}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=private-capability'
class Response:
    status_code=200
    def __init__(self,data):self.data=data;self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def iter_content(self,size):yield self.data
class Sessions:
    def __init__(self,bodies,barrier=False):
        self.bodies=bodies;self.sessions=[];self.lock=threading.Lock();self.active=0;self.peak=0;self.barrier=threading.Barrier(4)if barrier else None;self.first=set();self.error=None
    def factory(self):
        outer=self
        class Session:
            def __init__(self):self.owner=threading.get_ident();self.calls=0;self.closed=False
            def get(self,url,**kwargs):
                assert threading.get_ident()==self.owner,'sessions must never cross threads'
                assert kwargs==dict(stream=True,timeout=180,allow_redirects=False)
                self.calls+=1
                index=int(url.split('/')[-1].split('?')[0])
                with outer.lock:outer.active+=1;outer.peak=max(outer.active,outer.peak);first=len(outer.first)<4;outer.first.add(self.owner)
                try:
                    if outer.barrier and first:outer.barrier.wait(timeout=3)
                    time.sleep((8-index%8)*.001)
                    if outer.error and index==outer.error[0]:raise outer.error[1]
                    return Response(outer.bodies[index])
                finally:
                    with outer.lock:outer.active-=1
            def close(self):self.closed=True
        session=Session()
        with self.lock:self.sessions.append(session)
        return session
class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);self.out=self.root/'jobs'/'train';self.out.mkdir(parents=True)
        self.bodies=[('data-'+str(i)).encode()for i in range(12)]
        self.objects=[dict(url=URL.format(i),sha256=hashlib.sha256(data).hexdigest(),size=len(data))for i,data in enumerate(self.bodies)]
    def test_parallel_reads_are_bounded_ordered_and_sessions_are_reused_closed(self):
        sessions=Sessions(self.bodies,barrier=True);timings={}
        rows=list(prefetched_training_submissions(self.objects,self.out,timings,session_factory=sessions.factory))
        self.assertEqual([i for i,obj,path in rows],list(range(12)))
        self.assertEqual([path.read_bytes()for i,obj,path in rows],self.bodies)
        self.assertEqual(sessions.peak,4);self.assertLessEqual(len(sessions.sessions),4)
        self.assertGreater(max(s.calls for s in sessions.sessions),1);self.assertTrue(all(s.closed for s in sessions.sessions))
        self.assertEqual(timings['submission_download_and_authentication']['calls'],12)
    def test_lifecycle_writes_stay_serial_and_all_receipts_survive(self):
        sessions=Sessions(self.bodies);owner=threading.get_ident();calls=[];original=CacheLifecycle.record_download
        def record(cache,*args):
            self.assertEqual(threading.get_ident(),owner);calls.append(args[0]);return original(cache,*args)
        with patch.dict(os.environ,{'AFFINE_CACHE_LIFECYCLE_ROOT':str(self.root)}),patch.object(CacheLifecycle,'record_download',record):
            list(prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory))
        self.assertEqual(len(calls),12)
        self.assertEqual(len(CacheLifecycle(self.root).retire_downloads('train')),12)
    def test_size_or_digest_error_rejects_and_leaves_no_partial_file(self):
        self.objects[0]['size']=2
        sessions=Sessions(self.bodies)
        with self.assertRaises(ArtifactRejected):list(prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory))
        self.assertFalse((self.out/'submission-0.json').exists());self.assertFalse(list(self.out.glob('*.partial')))
        self.assertTrue(all(s.closed for s in sessions.sessions))
    def test_digest_error_and_nonbounded_size_reject(self):
        self.objects[0]['sha256']='0'*64;sessions=Sessions(self.bodies)
        with self.assertRaisesRegex(ArtifactRejected,'digest'):list(prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory))
        self.objects[0]['size']=2_000_001
        with self.assertRaisesRegex(ArtifactRejected,'bounded'):list(prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory))
    def test_transport_error_does_not_leak_presigned_capability(self):
        sessions=Sessions(self.bodies);sessions.error=(0,requests.ConnectionError(self.objects[0]['url']))
        with self.assertRaises(ValueError)as exc:list(prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory))
        self.assertNotIn('private-capability',str(exc.exception));self.assertTrue(all(s.closed for s in sessions.sessions))
    def test_admission_failure_closes_sessions_and_cancels_ahead_window(self):
        sessions=Sessions(self.bodies);generator=prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory)
        next(generator);generator.close()
        self.assertTrue(all(s.closed for s in sessions.sessions));self.assertLessEqual(sum(s.calls for s in sessions.sessions),4)
    def test_read_ahead_inputs_are_owned_after_admission_failure(self):
        sessions=Sessions(self.bodies)
        with patch.dict(os.environ,{'AFFINE_CACHE_LIFECYCLE_ROOT':str(self.root)}):
            generator=prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory)
            next(generator);generator.close()
        files=list(self.out.glob('submission-*.json'))
        self.assertGreaterEqual(len(files),1);self.assertLessEqual(len(files),4)
        self.assertEqual(len(CacheLifecycle(self.root).retire_downloads('train')),len(files))
    def test_serial_and_concurrent_transport_produce_identical_verified_bytes(self):
        sessions=Sessions(self.bodies);serial=self.root/'serial';serial.mkdir()
        for i,obj in enumerate(self.objects):
            with patch('requests.get',return_value=Response(self.bodies[i])):get_object(obj['url'],obj['sha256'],serial/str(i),obj['size'])
        rows=list(prefetched_training_submissions(self.objects,self.out,{},session_factory=sessions.factory))
        self.assertEqual([path.read_bytes()for i,obj,path in rows],[(serial/str(i)).read_bytes()for i in range(12)])
    def test_invalid_concurrency_and_redirect_status_cannot_succeed(self):
        with self.assertRaises(ValueError):list(prefetched_training_submissions(self.objects,self.out,{},workers=5))
        response=Response(self.bodies[0]);response.status_code=302
        with patch('requests.get',return_value=response),self.assertRaisesRegex(ValueError,'302'):get_object(self.objects[0]['url'],self.objects[0]['sha256'],self.out/'submission-0.json',100)
        self.assertTrue(response.closed);self.assertFalse(list(self.out.glob('*.partial')))
if __name__=='__main__':unittest.main()
