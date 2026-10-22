"""Offline replay harness: READ-ONLY R2 (no put/delete ever reaches R2), private temp coordinator root.
Builds the same tokenizer/resolve_task/identity as ops/mimo_lifecycle.run. Prints aggregates only."""
import importlib.util, json, shutil, sys, threading, time
from collections import defaultdict
from pathlib import Path
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
SRC = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
LIFE = RA/'20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json'
STATE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
spec = importlib.util.spec_from_file_location('affine_host_path_housekeeping', RA/'20261013-mimo-cutover-v1/capture-latency-v1/mimo_host_path_launcher.py')
paths = importlib.util.module_from_spec(spec); sys.modules[spec.name] = paths; spec.loader.exec_module(paths); paths.install()
sys.path.insert(0, str(SRC))
from botocore.exceptions import ClientError

ALLOWED = {'get_object', 'head_object', 'list_objects_v2', 'close', 'meta', 'exceptions'}
class ROClient:
    """Only GET/HEAD/LIST pass through. put_object NEVER reaches R2: an existing key yields the same 412
    the live conditional PUT returns (original readback-compare path runs); an absent key is recorded."""
    def __init__(self, real, stats):
        self._real, self._stats = real, stats
    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, **kw):
        assert IfNoneMatch == '*', 'only immutable conditional puts expected'
        try:
            self._real.head_object(Bucket=Bucket, Key=Key)
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
                with self._stats['lock']: self._stats['would_write_absent'] += 1
                return {}
            raise
        with self._stats['lock']: self._stats['put_suppressed_existing'] += 1
        raise ClientError({'Error': {'Code': 'PreconditionFailed'}, 'ResponseMetadata': {'HTTPStatusCode': 412}}, 'PutObject')
    def __getattr__(self, name):
        if name not in ALLOWED: raise PermissionError('read-only replay forbids ' + name)
        attr = getattr(self._real, name)
        if name in ('get_object', 'head_object', 'list_objects_v2'):
            def timed(*a, **k):
                t = time.perf_counter()
                try: return attr(*a, **k)
                finally:
                    with self._stats['lock']:
                        self._stats['n_' + name] += 1; self._stats['s_' + name] += time.perf_counter() - t
            return timed
        return attr

class ROBucket:
    def __init__(self, real, stats):
        self._real = real; self.name = real.name; self.client = ROClient(real.client, stats); self._stats = stats
    def commitment_read_client(self, **kw):
        return ROClient(self._real.commitment_read_client(**kw), self._stats)
    def capture_io_client(self, **kw):
        return ROClient(self._real.capture_io_client(**kw), self._stats)
    def complete_commitment_listing(self, *a, **k):
        from subnet.storage import Bucket
        return Bucket.complete_commitment_listing(self, *a, **k)
    def __getattr__(self, name):
        raise PermissionError('read-only replay forbids bucket.' + name)

def new_stats():
    return defaultdict(float, lock=threading.Lock())

def build():
    config = paths.private_json(LIFE)
    from ops.mimo_training_rehearsal import identity
    actor = identity(config['authority_seed_file'])
    from subnet.mimo_release_policy import ReleasePolicy
    policy = ReleasePolicy(config['release_policy'], actor.id, allow_public=True)
    from subnet.mimo_native_pool import authenticated_tokenizer
    tokenizer = authenticated_tokenizer(Path(config['tokenizer_directory']).absolute(), policy.scope['initial_checkpoint']['files'],
        policy.scope['immutable_manifest']['mimo_runtime']['chat_template_sha256'])
    from subnet.storage import Bucket
    real = Bucket(config['bucket'])
    return config, actor, policy, tokenizer, real

def coordinator_copy(epoch, root, bucket, actor, resolve, policy):
    from subnet.mimo_coordinator import Coordinator
    live = STATE/'epochs'/str(epoch)/'coordinator'
    if root.exists(): shutil.rmtree(root)
    root.mkdir(mode=0o700, parents=True)
    for n in ('OPENING.json', 'OPENING-REQUEST.json'): shutil.copy2(live/n, root/n)
    return Coordinator(root, bucket, actor, resolve_task=resolve, grade_batch=None, release_policy=policy), live
