import io,threading,time,unittest
from subnet.committed_training_inputs import bounded_document_reads,MAX_BYTES
from subnet.storage import Bucket
class Reads(unittest.TestCase):
 def test_four_inflight_reads_keep_original_order(self):
  class Client:
   def __init__(self):self.lock=threading.Lock();self.active=0;self.peak=0;self.start=threading.Barrier(4)
   def get_bounded(self,key,*,limit):
    with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
    if int(key)<4:self.start.wait(timeout=2)
    time.sleep(.002*(5-int(key)%4))
    with self.lock:self.active-=1
    return key.encode()
  client=Client();entries=[(None,None,None,dict(frozen_key=str(i),size=len(str(i))))for i in range(12)]
  found=list(bounded_document_reads(client,entries));self.assertEqual([data.decode()for _,data,_ in found],[str(i)for i in range(12)]);self.assertEqual(client.peak,4)
 def test_infrastructure_exception_preserved(self):
  class Client:
   def get_bounded(self,key,*,limit):raise ConnectionError('transport')
  with self.assertRaisesRegex(ConnectionError,'transport'):list(bounded_document_reads(Client(),[(None,None,None,dict(frozen_key='x',size=1))]))
 def test_bounded_get_closes_body_and_caps_read(self):
  class Body(io.BytesIO):
   def __init__(self,data):super().__init__(data);self.requested=[]
   def read(self,n=-1):self.requested.append(n);return super().read(n)
  class Client:
   def __init__(self,body,length):self.body=body;self.length=length
   def get_object(self,**kwargs):return dict(Body=self.body,ContentLength=self.length)
  b=Bucket.__new__(Bucket);b.name='test';body=Body(b'abc');b.client=Client(body,3)
  self.assertEqual(b.get_bounded('x',limit=3),b'abc');self.assertEqual(body.requested,[4]);self.assertTrue(body.closed)
  body=Body(b'x'*10);b.client=Client(body,10)
  with self.assertRaisesRegex(ValueError,'byte size'):b.get_bounded('x',limit=3)
  self.assertEqual(body.requested,[]);self.assertTrue(body.closed)
 def test_protocol_bound_rejects_unbounded_request_without_get(self):
  b=Bucket.__new__(Bucket)
  for limit in (0,True,MAX_BYTES+1):
   with self.assertRaisesRegex(ValueError,'size'):b.get_bounded('x',limit=limit)
if __name__=='__main__':unittest.main()
