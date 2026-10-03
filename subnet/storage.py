"""R2 persistence and deadline-enforced upload capability gateway."""
import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import boto3
from nacl.public import SealedBox
from nacl.signing import SigningKey, VerifyKey


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()

class SubmissionPolicyError(ValueError):
    """A completed submission violates operator size policy, not a read fault."""
    pass


class Identity:
    def __init__(self, seed=None):
        self.key = SigningKey(seed) if seed else SigningKey.generate()
        self.id = self.key.verify_key.encode().hex()

    def decrypt(self, envelope):
        return json.loads(SealedBox(self.key.to_curve25519_private_key()).decrypt(base64.b64decode(envelope)))


def encrypt(identity, value):
    box = SealedBox(VerifyKey(bytes.fromhex(identity)).to_curve25519_public_key())
    return base64.b64encode(box.encrypt(canonical(value))).decode()


class Bucket:
    def __init__(self, config):
        values = {}
        for line in Path(config['credentials_file']).read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('\"\'')
        self.name = config['bucket']
        self.client = boto3.client('s3', endpoint_url=config['endpoint'], region_name='auto',
                                   aws_access_key_id=values['R2_ACCESS_KEY_ID'],
                                   aws_secret_access_key=values['R2_SECRET_ACCESS_KEY'])

    def put(self, key, data, content_type='application/octet-stream'):
        self.client.put_object(Bucket=self.name, Key=key, Body=data, ContentType=content_type)

    def get(self, key):
        return self.client.get_object(Bucket=self.name, Key=key)['Body'].read()

    def json(self, key, value):
        self.put(key, canonical(value), 'application/json')

    def upload(self, key, path):
        self.client.upload_file(str(path), self.name, key)

    def copy(self, source, destination, *, expected_etag=None):
        """Copy an operator-owned frozen object within R2, without re-uploading."""
        condition = {} if expected_etag is None else {'CopySourceIfMatch': expected_etag}
        self.client.copy_object(Bucket=self.name, Key=destination,
                                CopySource={'Bucket': self.name, 'Key': source}, **condition)

    def verified_copy(self, source, destination, digest, size):
        """Stream-check immutable frozen bytes, then copy that exact object.

        The ETag condition only closes the read/copy race. SHA256 of the full
        body remains the integrity check; neither HEAD nor ETag replaces it.
        """
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest) or
                type(size) is not int or size <= 0):
            raise ValueError('frozen receipt integrity fields')
        response = self.client.get_object(Bucket=self.name, Key=source)
        body = response['Body']
        try:
            if response['ContentLength'] != size:
                raise ValueError('receipt size changed')
            hashed = hashlib.sha256(); count = 0
            while True:
                part = body.read(1024 * 1024)
                if not part: break
                count += len(part)
                if count > size: raise ValueError('receipt size changed')
                hashed.update(part)
            if count != size or hashed.hexdigest() != digest:
                raise ValueError('receipt hash changed')
            etag = response['ETag']
            if not isinstance(etag, str) or not etag:
                raise ValueError('frozen object ETag missing')
        finally:
            body.close()
        self.copy(source, destination, expected_etag=etag)

    def download(self, key, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.name, key, str(path))

    def presign(self,key,operation='get_object',expires=604800):
        if not 1<=expires<=604800:raise ValueError('R2 URL expiration')
        params=dict(Bucket=self.name,Key=key)
        if operation=='put_object':params['ContentType']='application/octet-stream'
        return self.client.generate_presigned_url(operation,Params=params,ExpiresIn=expires)

    def snapshot(self,key,limit=100_000_000):
        """Body and completion metadata belong to this single atomic GET."""
        from botocore.exceptions import ClientError
        try:response=self.client.get_object(Bucket=self.name,Key=key)
        except ClientError as exc:
            if str(exc.response.get('Error',{}).get('Code')) in ('NoSuchKey','404','NotFound'):return None
            raise
        body=response['Body']
        try:
            size=response['ContentLength']
            if not 0<size<=limit:raise SubmissionPolicyError('R2 upload size')
            data=body.read(limit+1)
            if len(data)!=size:raise ValueError('R2 snapshot size mismatch')
            return dict(data=data,size=size,etag=response['ETag'],completed_at=response['LastModified'].timestamp())
        finally:body.close()


class Gateway:
    """The bucket stays private; this service exposes only frozen public artifacts.

    Upload URLs are HMAC-signed, single-object capabilities encrypted to Ed25519
    identities. All PUTs and closure share a lock, including the R2 write itself.
    This makes finalization atomic with respect to an in-flight upload.
    """
    def __init__(self, bucket, host='127.0.0.1', port=0, state_path=None, public_url=None,direct_r2=False,publication_workers=4):
        if type(publication_workers) is not int or not 1<=publication_workers<=8:
            raise ValueError('bounded frozen publication workers')
        self.publication_workers=publication_workers
        self.bucket = bucket
        self.direct_r2=direct_r2
        self.state_path = Path(state_path) if state_path else None
        saved = json.loads(self.state_path.read_text()) if self.state_path and self.state_path.exists() else {}
        self.secret = bytes.fromhex(saved['secret']) if saved else secrets.token_bytes(32)
        self.epochs = saved.get('epochs', {})
        for epoch in self.epochs.values(): epoch['miners'] = set(epoch['miners'])
        self.lock = threading.RLock()
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, code, data):
                self.send_response(code)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = urllib.parse.urlparse(self.path).path
                if not path.startswith('/public/') or '..' in path:
                    return self.reply(403, b'private')
                try:
                    self.reply(200, gateway.bucket.get(path.lstrip('/')))
                except Exception:
                    self.reply(404, b'not found')

            def do_PUT(self):
                parts = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(parts.query)
                try:
                    expiry = int(q['expires'][0])
                    signed = f'{parts.path}|{expiry}'.encode()
                    if not hmac.compare_digest(q['signature'][0], hmac.new(gateway.secret, signed, 'sha256').hexdigest()):
                        return self.reply(403, b'invalid signature')
                    _, _, epoch, miner = parts.path.split('/')
                    self.connection.settimeout(30)
                    size = int(self.headers['Content-Length'])
                    with gateway.lock:
                        upload_limit=gateway.epochs.get(epoch,{}).get('upload_limit',100_000_000)
                    if not 0 < size <= upload_limit:
                        return self.reply(413, b'upload size')
                    data = self.rfile.read(size)
                    if len(data) != size:
                        return self.reply(400, b'incomplete upload')
                    with gateway.lock:
                        state = gateway.epochs[epoch]
                        if time.time() >= expiry or state['closed'] or miner not in state['miners']:
                            return self.reply(403, b'epoch closed')
                        key = f'private/{epoch}/{miner}.zip'
                        gateway.bucket.put(key, data)
                        state['uploads'][miner] = {'key': key, 'sha256': sha(data), 'received_at': time.time()}
                        gateway.persist()
                    self.reply(200, b'accepted')
                except Exception:
                    self.reply(400, b'invalid upload')

        class BoundedServer(ThreadingHTTPServer):
            daemon_threads = True
            def __init__(self, *args):
                self.slots = threading.BoundedSemaphore(8)
                super().__init__(*args)
            def process_request(self, request, address):
                if not self.slots.acquire(blocking=False):
                    self.shutdown_request(request); return
                try: super().process_request(request, address)
                except Exception:
                    self.slots.release(); raise
            def process_request_thread(self, request, address):
                try: super().process_request_thread(request, address)
                finally: self.slots.release()
        self.server = BoundedServer((host, port), Handler)
        self.url = public_url or f'http://{host}:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def open(self, epoch, miners, deadline, *, upload_limit=100_000_000):
        if type(upload_limit)is not int or upload_limit not in (100_000_000,2_000_000_000):
            raise ValueError('upload limit policy')
        with self.lock:
            if epoch in self.epochs:
                raise ValueError('epoch already exists')
            self.epochs[epoch] = dict(closed=False, miners=set(miners), uploads={},transport='direct-r2-v1' if self.direct_r2 else 'gateway-v1',start=int(time.time()),deadline=deadline)
            if upload_limit!=100_000_000:self.epochs[epoch]['upload_limit']=upload_limit
            caps = {}
            for miner in miners:
                if self.direct_r2:
                    key=f'private/{epoch}/staging/{miner}.zip'
                    expiry=max(1,deadline-int(time.time()))
                    caps[miner]=encrypt(miner,dict(transport='direct-r2-v1',put_url=self.bucket.presign(key,'put_object',expiry),headers={'Content-Type':'application/octet-stream'},deadline=deadline))
                    continue
                path = f'/upload/{epoch}/{miner}'
                signature = hmac.new(self.secret, f'{path}|{deadline}'.encode(), 'sha256').hexdigest()
                caps[miner] = encrypt(miner, {'put_url': f'{self.url}{path}?expires={deadline}&signature={signature}'})
            self.persist()
            return caps

    def persist(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            value = dict(secret=self.secret.hex(), epochs={k:dict(v, miners=sorted(v['miners'])) for k,v in self.epochs.items()})
            temporary = self.state_path.with_suffix('.tmp')
            temporary.write_bytes(canonical(value)); temporary.chmod(0o600)
            temporary.replace(self.state_path)

    def freeze(self, epoch):
        with self.lock:
            state = self.epochs[epoch]
            state['closed'] = True
            self.persist()
            if state.get('transport')=='direct-r2-v1' and 'frozen_receipts' in state:
                self.bucket.json(f'public/{epoch}/receipts.json',state['frozen_receipts'])
                return state['frozen_receipts']
            if state.get('transport')=='direct-r2-v1':
                snapshots=state.setdefault('snapshots',{})
                rejections=state.setdefault('rejections',{})
                for miner in sorted(state['miners']):
                    if miner in snapshots or miner in rejections:continue
                    key=f'private/{epoch}/staging/{miner}.zip'
                    try:
                        if 'upload_limit' in state:snapshot=self.bucket.snapshot(key,limit=state['upload_limit'])
                        else:snapshot=self.bucket.snapshot(key)
                    except SubmissionPolicyError as exc:
                        rejections[miner]=str(exc);self.persist();continue
                    if snapshot is None:
                        rejections[miner]='no completed upload';self.persist();continue
                    if not state['start']<=snapshot['completed_at']<state['deadline']:
                        rejections[miner]='completion outside signed epoch window';self.persist();continue
                    digest=sha(snapshot['data'])
                    frozen=f'private/{epoch}/frozen/{miner}/{digest}.zip'
                    self.bucket.put(frozen,snapshot['data'])
                    snapshots[miner]=dict(key=key,snapshot_key=frozen,sha256=digest,size=snapshot['size'],etag=snapshot['etag'],received_at=snapshot['completed_at'],snapshotted_at=time.time())
                    self.persist()
                state['uploads']=dict(snapshots)
                self.persist()
            def publish(item):
                miner, receipt = item
                key = f'public/{epoch}/submissions/{miner}.zip'
                if state.get('transport')=='direct-r2-v1' and hasattr(self.bucket,'verified_copy'):
                    self.bucket.verified_copy(receipt['snapshot_key'],key,receipt['sha256'],receipt['size'])
                else:
                    data = self.bucket.get(receipt.get('snapshot_key',receipt['key']))
                    if sha(data) != receipt['sha256']:
                        raise ValueError('receipt hash changed')
                    if state.get('transport')=='direct-r2-v1' and hasattr(self.bucket,'copy'):
                        # Never copy the miner's mutable staging key.
                        self.bucket.copy(receipt['snapshot_key'],key)
                    else:self.bucket.put(key, data)
                value = dict(receipt, frozen_key=key)
                if state.get('transport')=='direct-r2-v1':value['read_url']=self.bucket.presign(key)
                return miner,value
            if state.get('transport')=='direct-r2-v1' and hasattr(self.bucket,'verified_copy'):
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.publication_workers) as pool:
                    result = dict(pool.map(publish,state['uploads'].items()))
            else:result=dict(map(publish,state['uploads'].items()))
            if state.get('transport')=='direct-r2-v1':state['frozen_receipts']=result;self.persist()
            self.bucket.json(f'public/{epoch}/receipts.json', result)
            return result

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
