"""Prospective size-bound direct R2 upload grants; not selected by live epochs.

An authority-authenticated epoch policy and registration snapshot define the
scope. SQLite atomically meters grant issuance, not every use of a reusable R2
URL. No wallet key or permanent R2 credential is sent to a miner.
"""
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import sqlite3
import time
from urllib.parse import parse_qs, unquote, urlparse

from nacl.signing import VerifyKey

from .storage import canonical

VERSION = 'bounded-size-upload-broker-v1'
TRANSPORT = 'direct-r2-size-bound-v2'
DELEGATION = 'epoch-upload-delegation-v1'
CONTENT_TYPE = 'application/octet-stream'
MAX_COMPRESSED_BYTES = 2_000_000_000
MAX_REQUEST_BYTES = 16_384
POLICY_FIELDS = {
    'version', 'transport', 'epoch', 'start', 'deadline', 'identities',
    'registration_document_sha256', 'max_compressed_bytes', 'grant_ttl_seconds',
    'max_grants_per_identity', 'max_declared_bytes_per_identity',
    'minimum_issue_interval_seconds', 'snapshot_mode',
}


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signed(payload, key):
    return dict(signer=key.verify_key.encode().hex(), payload=payload,
                signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())


def authenticate(document, expected):
    if not isinstance(document, dict) or set(document) != {'signer', 'payload', 'signature'}:
        raise ValueError('signed upload authorization framing')
    if document['signer'] != expected:
        raise ValueError('upload authorization signer')
    VerifyKey(bytes.fromhex(expected)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    return document['payload']


def _hex(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name)
    return value


def staging_key(epoch, identity):
    if (not isinstance(epoch, str) or re.fullmatch('[A-Za-z0-9_-]{1,200}', epoch) is None
            or not _hex(identity)):
        raise ValueError('exact upload epoch/identity namespace')
    return 'private/' + epoch + '/staging/' + identity + '.zip'


def immutable_snapshot_key(epoch, identity, nonce):
    staging_key(epoch, identity)
    if not _hex(nonce):
        raise ValueError('exact immutable upload nonce')
    return 'private/' + epoch + '/staging/' + identity + '/' + nonce + '.zip'


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != POLICY_FIELDS:
        raise ValueError('upload policy fields')
    if policy['version'] != VERSION or policy['transport'] != TRANSPORT:
        raise ValueError('upload policy version')
    if policy['snapshot_mode'] not in ('immutable-snapshot', 'cumulative-replace'):
        raise ValueError('explicit upload snapshot mode')
    _integer(policy['start'], 'upload start', 0, 2**53)
    _integer(policy['deadline'], 'upload deadline', policy['start'] + 1, 2**53)
    identities = policy['identities']
    if (not isinstance(identities, list) or not 1 <= len(identities) <= 4096
            or not all(_hex(i) for i in identities)
            or identities != sorted(set(identities))):
        raise ValueError('activated upload identity population')
    for identity in identities:
        staging_key(policy['epoch'], identity)
    if not _hex(policy['registration_document_sha256']):
        raise ValueError('upload registration snapshot hash')
    size = _integer(policy['max_compressed_bytes'], 'compressed upload cap', 1, MAX_COMPRESSED_BYTES)
    grants = _integer(policy['max_grants_per_identity'], 'upload grant quota', 1, 256)
    _integer(policy['max_declared_bytes_per_identity'], 'declared-byte grant quota', size, size * grants)
    _integer(policy['grant_ttl_seconds'], 'short upload grant lifetime', 1, 300)
    _integer(policy['minimum_issue_interval_seconds'], 'upload grant issue interval', 0, 300)
    return policy


def validate_registrations(document, policy, authority):
    registrations = authenticate(document, authority)
    if (sha(document) != policy['registration_document_sha256']
            or registrations.get('epoch') != policy['epoch']):
        raise ValueError('exact activated registration snapshot binding')
    rows = registrations.get('registrations')
    if not isinstance(rows, dict) or not 1 <= len(rows) <= 4096:
        raise ValueError('activated registration rows')
    keys = []
    for hotkey, row in rows.items():
        if (not isinstance(hotkey, str) or not hotkey or not isinstance(row, dict)
                or not _hex(row.get('public_key'))):
            raise ValueError('activated registration key')
        _integer(row.get('uid'), 'activated registration UID', 0, 65535)
        keys.append(row['public_key'])
    if sorted(keys) != policy['identities']:
        raise ValueError('exact activated identity population')
    # This authenticates the operator's already-validated on-chain activation
    # snapshot. It is not a replacement for chain registration/activation checks.


class SizeBoundUploadBroker:
    def __init__(self, database, authority_key, bucket, *, clock=time.time):
        self.path = Path(database)
        if self.path.is_symlink():
            raise ValueError('regular operator upload database required')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = authority_key
        self.authority = authority_key.verify_key.encode().hex()
        self.bucket = bucket
        self.clock = clock
        with self.transaction() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS upload_epochs (
              epoch TEXT PRIMARY KEY, policy_sha TEXT NOT NULL, document TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS upload_identities (
              epoch TEXT NOT NULL, identity TEXT NOT NULL, grants INTEGER NOT NULL DEFAULT 0,
              declared_bytes INTEGER NOT NULL DEFAULT 0, last_issued REAL,
              PRIMARY KEY(epoch, identity));
            CREATE TABLE IF NOT EXISTS upload_requests (
              epoch TEXT NOT NULL, identity TEXT NOT NULL, nonce TEXT NOT NULL,
              request_sha TEXT NOT NULL, grant TEXT NOT NULL,
              PRIMARY KEY(epoch, identity, nonce));
            ''')
        self.path.chmod(0o600)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def register_epoch(self, policy_document, registrations_document):
        policy = validate_policy(authenticate(policy_document, self.authority))
        validate_registrations(registrations_document, policy, self.authority)
        digest = sha(policy_document)
        with self.transaction() as db:
            old = db.execute('SELECT policy_sha FROM upload_epochs WHERE epoch=?',
                             (policy['epoch'],)).fetchone()
            if old:
                if old['policy_sha'] != digest:
                    raise ValueError('immutable upload policy collision')
                return digest
            db.execute('INSERT INTO upload_epochs VALUES(?,?,?)',
                       (policy['epoch'], digest, canonical(policy_document).decode()))
            db.executemany('INSERT INTO upload_identities(epoch,identity) VALUES(?,?)',
                           ((policy['epoch'], identity) for identity in policy['identities']))
        return digest

    def policy(self, epoch, db):
        row = db.execute('SELECT policy_sha,document FROM upload_epochs WHERE epoch=?',
                         (epoch,)).fetchone()
        if row is None:
            raise ValueError('unknown upload epoch')
        document = json.loads(row['document'])
        if sha(document) != row['policy_sha']:
            raise ValueError('persisted upload policy integrity')
        return validate_policy(authenticate(document, self.authority)), row['policy_sha']

    def delegate(self, epoch, identity):
        """Return a private scoped bearer authorization for operator-delegated mining.

        Caller encrypts it to the activated Ed25519 identity or writes it to a
        private remote cap file. Never place this plaintext in a public manifest.
        """
        with self.transaction() as db:
            policy, policy_sha = self.policy(epoch, db)
            if identity not in policy['identities'] or self.clock() >= policy['deadline']:
                raise ValueError('upload delegation scope')
            return signed(dict(version=DELEGATION, epoch=epoch, identity=identity,
                policy_sha256=policy_sha, deadline=policy['deadline'],
                bearer_secret=secrets.token_hex(32)), self.key)

    def request_payload(self, request):
        if not isinstance(request, dict):
            raise ValueError('upload grant request')
        if request.get('authentication') == 'ed25519' and set(request) == {'authentication', 'document'}:
            document = request['document']
            payload = authenticate(document, document.get('signer'))
            if payload.get('identity') != document['signer']:
                raise ValueError('activated request identity signature')
            authorization = None
        elif request.get('authentication') == 'delegated' and set(request) == {'authentication', 'payload', 'authorization'}:
            payload = request['payload']
            authorization = authenticate(request['authorization'], self.authority)
            if (set(authorization) != {'version', 'epoch', 'identity', 'policy_sha256',
                    'deadline', 'bearer_secret'} or authorization['version'] != DELEGATION
                    or not _hex(authorization['bearer_secret'])):
                raise ValueError('scoped delegated upload authorization')
        else:
            raise ValueError('upload request authentication mode')
        fields = {'version', 'action', 'epoch', 'identity', 'nonce', 'bytes', 'at'}
        if (not isinstance(payload, dict) or set(payload) != fields
                or payload['version'] != VERSION or payload['action'] != 'grant'
                or not _hex(payload['nonce']) or not _hex(payload['identity'])):
            raise ValueError('upload request fields/nonce/identity')
        staging_key(payload['epoch'], payload['identity'])
        _integer(payload['bytes'], 'requested compressed bytes', 1, MAX_COMPRESSED_BYTES)
        if type(payload['at']) not in (int, float) or not math.isfinite(payload['at']):
            raise ValueError('upload request finite timestamp')
        return payload, authorization

    def presign(self, key, size, expires, *, immutable):
        parameters = dict(Bucket=self.bucket.name, Key=key, ContentType=CONTENT_TYPE,
                          ContentLength=size)
        if immutable:
            parameters['IfNoneMatch'] = '*'
        return self.bucket.client.generate_presigned_url('put_object',
            Params=parameters, ExpiresIn=expires)

    def checked_url_expiry(self, url, key, expected_seconds, now, deadline, *, immutable):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        required_headers = {'content-length', 'content-type', 'host'}
        if immutable:
            required_headers.add('if-none-match')
        if (parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.r2.cloudflarestorage.com')
                or parsed.username or parsed.password or parsed.port not in (None, 443)
                or parsed.fragment or not unquote(parsed.path).endswith('/' + key)
                or query.get('X-Amz-Algorithm') != ['AWS4-HMAC-SHA256']
                or len(query.get('X-Amz-Signature', [])) != 1
                or not _hex(query['X-Amz-Signature'][0])
                or len(query.get('X-Amz-SignedHeaders', [])) != 1
                or not required_headers <= set(query['X-Amz-SignedHeaders'][0].split(';'))
                or query.get('X-Amz-Expires') != [str(expected_seconds)]
                or len(query.get('X-Amz-Date', [])) != 1):
            raise ValueError('exact key, size-signed headers and expiry required')
        issued = datetime.strptime(query['X-Amz-Date'][0], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
        expiry = issued + expected_seconds
        if abs(issued - now) > 5 or not now < expiry <= deadline:
            raise ValueError('actual signed URL exceeds original epoch boundary')
        return expiry

    def request(self, request):
        payload, authorization = self.request_payload(request)
        digest = sha(payload)
        with self.transaction() as db:
            now = self.clock()
            if type(now) not in (int, float) or not math.isfinite(now):
                raise ValueError('actual upload clock')
            policy, policy_sha = self.policy(payload['epoch'], db)
            identity = payload['identity']
            if (identity not in policy['identities'] or not policy['start'] <= now < policy['deadline']
                    or abs(payload['at'] - now) > 60
                    or payload['bytes'] > policy['max_compressed_bytes']):
                raise ValueError('upload identity/window/size scope')
            if authorization is not None and any(authorization[k] != v for k, v in
                    dict(epoch=policy['epoch'], identity=identity, policy_sha256=policy_sha,
                         deadline=policy['deadline']).items()):
                raise ValueError('delegated upload policy/identity/deadline binding')
            prior = db.execute('SELECT request_sha,grant FROM upload_requests WHERE epoch=? AND identity=? AND nonce=?',
                               (policy['epoch'], identity, payload['nonce'])).fetchone()
            if prior:
                if prior['request_sha'] != digest:
                    raise ValueError('upload nonce conflict')
                document = json.loads(prior['grant'])
                grant = authenticate(document, self.authority)
                if not now < grant['expires_at']:
                    raise ValueError('original upload grant expired; request a new nonce')
                return document
            counters = db.execute('SELECT * FROM upload_identities WHERE epoch=? AND identity=?',
                                  (policy['epoch'], identity)).fetchone()
            if counters is None:
                raise ValueError('registered upload counters required')
            if (counters['grants'] >= policy['max_grants_per_identity']
                    or counters['declared_bytes'] + payload['bytes'] > policy['max_declared_bytes_per_identity']):
                raise ValueError('upload issuance/declared-byte quota exceeded')
            if (counters['last_issued'] is not None and
                    now - counters['last_issued'] < policy['minimum_issue_interval_seconds']):
                raise ValueError('upload grant issue rate exceeded')
            seconds = min(policy['grant_ttl_seconds'], math.floor(policy['deadline'] - now))
            if seconds < 1:
                raise ValueError('insufficient original upload window')
            immutable = policy['snapshot_mode'] == 'immutable-snapshot'
            key = (immutable_snapshot_key(policy['epoch'], identity, payload['nonce'])
                   if immutable else staging_key(policy['epoch'], identity))
            url = self.presign(key, payload['bytes'], seconds, immutable=immutable)
            expiry = self.checked_url_expiry(url, key, seconds, now, policy['deadline'], immutable=immutable)
            # Reobserve the actual deadline after local signing. No failed
            # issuance consumes counters and no late original grant is revived.
            if self.clock() >= expiry:
                raise ValueError('upload signing completed after original grant expiry')
            headers = {'Content-Type': CONTENT_TYPE, 'Content-Length': str(payload['bytes'])}
            if immutable:
                headers['If-None-Match'] = '*'
            grant = dict(version=VERSION, transport=TRANSPORT, epoch=policy['epoch'],
                identity=identity, policy_sha256=policy_sha, nonce=payload['nonce'],
                snapshot_mode=policy['snapshot_mode'],
                staging_key=key, deadline=policy['deadline'], issued_at=now, expires_at=expiry,
                max_compressed_bytes=policy['max_compressed_bytes'], bytes=payload['bytes'],
                put_url=url, headers=headers,
                quota=dict(grants_remaining=policy['max_grants_per_identity']-counters['grants']-1,
                    declared_bytes_remaining=policy['max_declared_bytes_per_identity']-counters['declared_bytes']-payload['bytes'],
                    minimum_issue_interval_seconds=policy['minimum_issue_interval_seconds']))
            document = signed(grant, self.key)
            db.execute('UPDATE upload_identities SET grants=grants+1,declared_bytes=declared_bytes+?,last_issued=? WHERE epoch=? AND identity=?',
                       (payload['bytes'], now, policy['epoch'], identity))
            db.execute('INSERT INTO upload_requests VALUES(?,?,?,?,?)',
                       (policy['epoch'], identity, payload['nonce'], digest, canonical(document).decode()))
            return document

    def finalization_candidates(self, epoch, identity):
        """Operator-only known issued keys; not a miner-selected object listing.

        A future freezer selects the latest actually completed in-window object
        using atomic R2 metadata, then independently hashes/checks its body.
        This API supplies no plaintext URL or delegation secret.
        """
        with self.transaction() as db:
            policy, policy_sha = self.policy(epoch, db)
            if identity not in policy['identities'] or self.clock() < policy['deadline']:
                raise ValueError('only registered closed epoch may finalize')
            records = db.execute('SELECT grant FROM upload_requests WHERE epoch=? AND identity=?',
                                 (epoch, identity)).fetchall()
            result = []
            for row in records:
                document = json.loads(row['grant'])
                grant = authenticate(document, self.authority)
                if (grant['epoch'] != epoch or grant['identity'] != identity
                        or grant['policy_sha256'] != policy_sha):
                    raise ValueError('issued snapshot integrity')
                expected = (immutable_snapshot_key(epoch, identity, grant['nonce'])
                            if policy['snapshot_mode'] == 'immutable-snapshot'
                            else staging_key(epoch, identity))
                if grant['staging_key'] != expected:
                    raise ValueError('issued snapshot namespace')
                result.append({k: grant[k] for k in ('staging_key', 'bytes', 'deadline',
                    'nonce', 'issued_at', 'expires_at', 'snapshot_mode')})
            return result


class UploadBrokerServer(ThreadingHTTPServer):
    """Optional loopback HTTP endpoint behind the existing operator HTTPS proxy.

    Production proxy must also bound connections/request rate. Public admission
    should not expose a raw unauthenticated thread-per-request port.
    """
    daemon_threads = True

    def __init__(self, address, broker):
        if address[0] not in ('127.0.0.1', '::1', 'localhost'):
            raise ValueError('upload broker requires loopback and HTTPS proxy')

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, *_):
                pass

            def reply(self, code, value):
                body = canonical(value)
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != '/v1/upload-grants':
                    self.reply(404, {'error': 'unknown route'})
                    return
                try:
                    lengths = self.headers.get_all('Content-Length', [])
                    if (len(lengths) != 1 or not lengths[0].isdigit()
                            or self.headers.get('Transfer-Encoding') is not None
                            or not 0 < int(lengths[0]) <= MAX_REQUEST_BYTES):
                        self.reply(413, {'error': 'bounded request framing required'})
                        return
                    data = self.rfile.read(int(lengths[0]))
                    if len(data) != int(lengths[0]):
                        raise ValueError('incomplete upload grant request')
                    self.reply(200, broker.request(json.loads(data)))
                except Exception:
                    # Never echo raw bearer authorizations, capabilities, SDK
                    # exceptions, SQL paths, or request fields into responses.
                    self.reply(403, {'error': 'upload grant refused'})

        super().__init__(address, Handler)
