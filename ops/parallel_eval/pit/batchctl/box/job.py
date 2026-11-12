#!/usr/bin/env python3
"""The job of one lease on one rented box. One file, standard library only.

    python3 job.py /path/to/job.json

The host writes ``job.json`` and starts this file. The box holds no
credential: every read and every write goes through a URL that the host
signed for this lease. A URL can read one object of the base state or write
one object under ``staging/<batch>/<fence>/``. Nothing else.

What the job does, in this order:

1. Take the lease on this box. A second start of the same lease stops.
2. Check the box: number and name of the GPUs, host RAM, free disk.
3. Read the manifest and the file index of the base state. The manifest is
   checked against the id of the base state, the index against the manifest.
4. Refuse to go on unless the fp32 master digest of the base is the digest
   that the lease expects. Every step gets that digest as
   ``AFFINE_FP32_EXPECTED_MASTER_HASH``. The bootstrap override is removed
   from the environment of every step and a job that sets it is refused.
5. Download the base state. Every object is checked against the manifest
   (size and sha256) and every file against the index.
6. Run the steps (train, probe). A step that writes a state file starts the
   upload of the new state while the next step runs. A file of the new state
   whose sha256 its step named is uploaded only if it still has that sha256.
7. Upload ``result.json``, then ``COMPLETE.json`` last. The controller reads
   nothing of this lease before ``COMPLETE.json`` exists.
8. Remove the weights, the inputs and the signed URLs from the box and stop.
   The pod itself is removed by the controller (by its exact id) and by the
   lifetime that the rental set: a pod holds no key that could remove it.
   The directory of the lease stays, with the claim of step 1 in it.

While it works, the job writes a heartbeat with a growing number.

Two ways to end without a new state:

* The batch cannot be trained on this base (it is bound to another
  checkpoint, or a step says that the batch is at fault). That is a
  decision: the job uploads a result that says so, and ``COMPLETE.json``
  with no state.
* The box or the network failed. That is no decision: the job writes no
  ``COMPLETE.json``. The lease ends on the controller and another box gets
  the batch.
"""
import hashlib
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

JOB_SCHEMA = 'affine-batchctl-job-v1'
RESULT_SCHEMA = 'affine-batchctl-result-v1'
FILES_SCHEMA = 'affine-batchctl-files-v1'
BOX_STATE_SCHEMA = 'affine-batchctl-box-state-v1'
BOX_OUTCOME_SCHEMA = 'affine-batchctl-box-outcome-v1'
# The formats of the controller (``layout.py``). A test compares them.
STATE_SCHEMA = 'affine-batchctl-state-v1'
COMPLETE_SCHEMA = 'affine-batchctl-complete-v1'
HEARTBEAT_SCHEMA = 'affine-batchctl-heartbeat-v1'
ROLES = ('fp32_master', 'bf16_weights', 'aux')
REQUIRED_ROLES = ('fp32_master', 'bf16_weights')
META_FP32_MASTER_DIGEST = 'fp32_master_digest'
INDEX_NAME = 'index.json'
SLOT_NAME = 'p-%06d'
MAX_OBJECTS = 4096
MAX_RESULT_BYTES = 64 << 20
MAX_COMPLETE_BYTES = 1 << 20
MAX_INDEX_BYTES = 8 << 20
MAX_MANIFEST_BYTES = 1 << 20

OUTCOME_PROBED = 'probed'
OUTCOME_NOT_TRAINED = 'not_trained'
REASON_OTHER_CHECKPOINT = 'bound_to_other_checkpoint'

ENV_EXPECTED_MASTER = 'AFFINE_FP32_EXPECTED_MASTER_HASH'
ENV_BOOTSTRAP_OVERRIDE = 'AFFINE_FP32_BOOTSTRAP_OVERRIDE'
# A script may remove the override from its environment (``env -u NAME``).
# It may never give it a value.
SETS_OVERRIDE = re.compile(r'BOOTSTRAP_OVERRIDE\s*=')

# The heartbeat number of the box starts above every number that the host
# writes while it prepares the box, so the number never goes back.
HEARTBEAT_BASE = 1000000
BLOCK = 1 << 20
# A job can change these three with its ``http`` object.
HTTP_TIMEOUT_S = 120.0
HTTP_TRIES = 5
RETRY_PAUSE_S = 2.0
EXIT_DONE, EXIT_FAILED, EXIT_BAD_JOB, EXIT_STARTED, EXIT_DEADLINE = 0, 1, 2, 3, 4

HEX64 = re.compile(r'[0-9a-f]{64}')
SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}')
META_KEY = re.compile(r'[a-z][a-z0-9_]{0,63}')
PATH = re.compile(r'([a-z0-9]+/)?[A-Za-z0-9][A-Za-z0-9._-]{0,199}')
# Directories that a job may never remove.
NEVER_WIPE = ('/', '/root', '/home', '/usr', '/etc', '/var', '/opt', '/bin', '/lib', '/workspace', '/tmp')


class JobError(Exception):
    """The box or the network failed. No decision is written."""


class BadJob(Exception):
    """``job.json`` is not a job that this file runs."""


class NotTrained(Exception):
    """The batch cannot be trained on this base. A decision."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def is_hex64(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def log(text):
    sys.stderr.write('%s %s\n' % (time.strftime('%H:%M:%S'), text))
    sys.stderr.flush()


# ----------------------------------------------------------------------------
# job.json
# ----------------------------------------------------------------------------

def need(cond, why):
    if not cond:
        raise BadJob(why)


def is_abs(path):
    return isinstance(path, str) and path.startswith('/') and os.path.normpath(path) == path and '\x00' not in path


def check_env(env, expected, where):
    need(isinstance(env, dict), '%s: env is not an object' % where)
    for key, value in env.items():
        need(isinstance(key, str) and isinstance(value, str), '%s: env must map text to text' % where)
        need('BOOTSTRAP' not in key.upper(), '%s: the bootstrap override is never set' % where)
        if key == ENV_EXPECTED_MASTER:
            need(value == expected, '%s: another fp32 master digest than the lease expects' % where)


def check_job(job):
    """Every rule that ``job.json`` must keep. Raises ``BadJob``."""
    need(isinstance(job, dict) and job.get('schema') == JOB_SCHEMA, 'not a job of this version')
    for key in ('run_id', 'batch_id'):
        need(isinstance(job.get(key), str) and job[key], key)
    for key in ('fence', 'attempt', 'base_seq'):
        need(is_int(job.get(key)) and job[key] >= 0, key)
    need(job['fence'] >= 1, 'fence')
    for key in ('base_state', 'expected_fp32_master_digest'):
        need(is_hex64(job.get(key)), key)
    for key in ('heartbeat_every_s', 'job_ttl_s'):
        need(isinstance(job.get(key), (int, float)) and not isinstance(job[key], bool) and job[key] > 0, key)
    need(isinstance(job.get('content_type'), str) and job['content_type'], 'content_type')
    need(is_int(job.get('chunk_bytes')) and job['chunk_bytes'] >= BLOCK, 'chunk_bytes')
    base, put = job.get('base'), job.get('put')
    need(isinstance(base, dict) and isinstance(base.get('manifest'), str), 'base.manifest')
    need(isinstance(base.get('objects'), dict), 'base.objects')
    need(isinstance(put, dict) and isinstance(put.get('objects'), dict), 'put.objects')
    for key in ('heartbeat', 'result', 'complete'):
        need(isinstance(put.get(key), str) and put[key], 'put.' + key)
    for group in (base['objects'], put['objects']):
        for name, url in group.items():
            need(isinstance(name, str) and SAFE_NAME.fullmatch(name) is not None and '..' not in name, 'object name')
            need(isinstance(url, str) and url, 'object url')
    need(INDEX_NAME in put['objects'], 'no slot for the index')
    expected = job['expected_fp32_master_digest']
    check_env(job.get('env', {}), expected, 'job')
    require = job.get('require', {})
    need(isinstance(require, dict), 'require')
    binding = job.get('binding')
    need(binding is None or (isinstance(binding, dict) and isinstance(binding.get('checkpoint_id'), str)), 'binding')
    places = job.get('place')
    need(isinstance(places, list) and places, 'place')
    for rule in places:
        need(isinstance(rule, dict) and isinstance(rule.get('prefix'), str), 'place.prefix')
        need(re.fullmatch(r'([a-z0-9]+/)?', rule['prefix']) is not None, 'place.prefix')
        need(is_abs(rule.get('dir')), 'place.dir')
        need(all(is_abs(link) for link in rule.get('links', [])), 'place.links')
    for check in job.get('expect', []):
        need(isinstance(check, dict) and is_abs(check.get('file')), 'expect.file')
        need(isinstance(check.get('key'), list) and check['key'], 'expect.key')
        need('equals' in check, 'expect.equals')
    steps = job.get('steps')
    need(isinstance(steps, list), 'steps')
    names = set()
    for step in steps:
        need(isinstance(step, dict), 'step')
        name = step.get('name')
        need(isinstance(name, str) and SAFE_NAME.fullmatch(name) is not None and name not in names, 'step.name')
        names.add(name)
        need(isinstance(step.get('script'), str) and step['script'], 'step.script')
        need(SETS_OVERRIDE.search(step['script']) is None, 'step %s: the bootstrap override is never set' % name)
        need(isinstance(step.get('timeout_s'), (int, float)) and step['timeout_s'] > 0, 'step.timeout_s')
        check_env(step.get('env', {}), expected, 'step ' + name)
        need(step.get('cwd') is None or is_abs(step['cwd']), 'step.cwd')
        need(step.get('state') is None or is_abs(step['state']), 'step.state')
        for token, source in step.get('subst', {}).items():
            need(isinstance(token, str) and len(token) >= 8, 'step.subst token')
            need(isinstance(source, dict) and is_abs(source.get('file')), 'step.subst.file')
            need(isinstance(source.get('key'), list) and source['key'], 'step.subst.key')
            need(isinstance(source.get('match'), str), 'step.subst.match')
        for pattern in step.get('batch_caused', []):
            need(isinstance(pattern, str) and pattern, 'step.batch_caused')
            try:
                re.compile(pattern)
            except re.error:
                raise BadJob('step.batch_caused is not a pattern') from None
    need(sum(1 for step in steps if step.get('state')) <= 1, 'more than one step writes a state')
    need(job.get('outcome') is None or is_abs(job['outcome']), 'outcome')
    for path in job.get('wipe', []):
        need(is_abs(path) and path not in NEVER_WIPE and path.count('/') >= 2, 'wipe: %r' % (path,))
    http = job.get('http', {})
    need(isinstance(http, dict) and set(http) <= {'tries', 'pause_s', 'timeout_s'}, 'http')
    need(is_int(http.get('tries', 1)) and http.get('tries', 1) >= 1, 'http.tries')
    for key in ('pause_s', 'timeout_s'):
        value = http.get(key, 1)
        need(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0, 'http.' + key)
    return job


def use_http_settings(job):
    global HTTP_TRIES, RETRY_PAUSE_S, HTTP_TIMEOUT_S
    http = job.get('http', {})
    HTTP_TRIES = http.get('tries', HTTP_TRIES)
    RETRY_PAUSE_S = http.get('pause_s', RETRY_PAUSE_S)
    HTTP_TIMEOUT_S = http.get('timeout_s', HTTP_TIMEOUT_S)


# ----------------------------------------------------------------------------
# HTTP through signed URLs
# ----------------------------------------------------------------------------

class Definite(Exception):
    """The server said no. Another try gives the same answer."""


def _retry(what, call, tries=None):
    last = None
    for attempt in range(HTTP_TRIES if tries is None else tries):
        try:
            return call(attempt)
        except Definite as exc:
            raise JobError('%s: %s' % (what, exc)) from None
        except (OSError, ValueError, http.client.HTTPException) as exc:
            last = exc
            time.sleep(min(30.0, RETRY_PAUSE_S * (2 ** attempt)))
    raise JobError('%s: %s' % (what, type(last).__name__))


def _open(request, timeout):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if 400 <= code < 500 and code not in (408, 429):
            raise Definite('HTTP %d' % code) from None
        raise OSError('HTTP %d' % code) from None


def http_get_bytes(url, limit, what):
    def once(_attempt):
        with _open(urllib.request.Request(url), HTTP_TIMEOUT_S) as response:
            data = response.read(limit + 1)
            length = response.headers.get('Content-Length')
        if len(data) > limit:
            raise Definite('larger than %d bytes' % limit)
        if length is not None and str(length).isdigit() and int(length) != len(data):
            # The connection ended early. A short read never raises by itself.
            raise OSError('short read')
        return data
    return _retry(what, once)


def http_put_bytes(url, data, content_type, what, tries=None):
    def once(_attempt):
        request = urllib.request.Request(url, data=data, method='PUT', headers={
            'Content-Type': content_type, 'Content-Length': str(len(data)),
        })
        with _open(request, HTTP_TIMEOUT_S) as response:
            response.read()
    return _retry(what, once, tries=tries)


class Slice:
    """``size`` bytes of a file from ``offset``, read once, hashed on the way."""

    def __init__(self, path, offset, size):
        self._file = open(path, 'rb')
        self._file.seek(offset)
        self._left = size
        self.sha = hashlib.sha256()

    def read(self, count=-1):
        if count is None or count < 0 or count > self._left:
            count = self._left
        data = self._file.read(count) if count else b''
        self._left -= len(data)
        self.sha.update(data)
        return data

    def close(self):
        self._file.close()


def http_put_slice(url, path, offset, size, sha256, content_type, what):
    """Upload one part of a file. The bytes that leave are hashed again: a
    file that changed after it was measured is not uploaded as the old one."""
    def once(_attempt):
        body = Slice(path, offset, size)
        try:
            request = urllib.request.Request(url, data=body, method='PUT', headers={
                'Content-Type': content_type, 'Content-Length': str(size),
            })
            with _open(request, HTTP_TIMEOUT_S) as response:
                response.read()
            if body._left or body.sha.hexdigest() != sha256:
                raise Definite('the file changed while it was uploaded')
        finally:
            body.close()
    return _retry(what, once)


def http_get_into(url, out, size, sha256, file_sha, what):
    """Download one object of exactly ``size`` bytes into ``out`` (an open
    file, at the place where the object starts). A broken read goes on from
    the byte that is missing. Raises unless size and sha256 are the ones of
    the manifest.

    ``file_sha`` is the hash of the file up to this object. It is not
    changed; the hash of the file with this object is returned.
    """
    start = out.tell()
    state = {'got': 0, 'sha': hashlib.sha256(), 'file_sha': file_sha.copy()}

    def restart():
        out.seek(start)
        out.truncate()
        state.update(got=0, sha=hashlib.sha256(), file_sha=file_sha.copy())

    def once(_attempt):
        headers = {'Range': 'bytes=%d-' % state['got']} if state['got'] else {}
        with _open(urllib.request.Request(url, headers=headers), HTTP_TIMEOUT_S) as response:
            ranged = str(response.headers.get('Content-Range', '')).startswith('bytes %d-' % state['got'])
            if state['got'] and not (response.status == 206 and ranged):
                # The server sent the object from its start again.
                restart()
            while True:
                data = response.read(BLOCK)
                if not data:
                    break
                state['got'] += len(data)
                if state['got'] > size:
                    raise Definite('more bytes than the manifest says')
                state['sha'].update(data)
                state['file_sha'].update(data)
                out.write(data)
        if state['got'] != size:
            raise OSError('short read')
        if state['sha'].hexdigest() != sha256:
            restart()
            raise ValueError('sha256 mismatch')

    _retry(what, once)
    return state['file_sha']


# ----------------------------------------------------------------------------
# The base state
# ----------------------------------------------------------------------------

def parse_manifest(raw, base_state):
    """The manifest of the base state: ``{name: (role, sha256, size)}`` and the meta."""
    if sha256_hex(raw) != base_state:
        raise JobError('the manifest is not the manifest of the base state')
    try:
        doc = json.loads(raw)
        objects = {}
        for item in doc['objects']:
            name, role, sha, size = item['name'], item['role'], item['sha256'], item['size']
            if (not isinstance(name, str) or SAFE_NAME.fullmatch(name) is None or '..' in name or name in objects
                    or role not in ROLES or not is_hex64(sha) or not is_int(size) or size <= 0):
                raise ValueError('object')
            objects[name] = (role, sha, size)
        meta = doc['meta']
        if doc['schema'] != STATE_SCHEMA or not isinstance(meta, dict) or not objects or len(objects) > MAX_OBJECTS:
            raise ValueError('manifest')
    except (ValueError, KeyError, TypeError, RecursionError):
        raise JobError('the manifest of the base state is not valid') from None
    return objects, meta


def check_path(path):
    return isinstance(path, str) and PATH.fullmatch(path) is not None and '..' not in path


def parse_index(raw, objects):
    """The file list of a state. Every part must be an object of the manifest,
    used once, with the role of its file. Every object but the index is a part."""
    try:
        doc = json.loads(raw)
        files, used = [], set()
        if doc['schema'] != FILES_SCHEMA or not isinstance(doc['files'], list):
            raise ValueError('index')
        for item in doc['files']:
            path, role, size, sha, parts = item['path'], item['role'], item['size'], item['sha256'], item['parts']
            if (not check_path(path) or role not in ROLES or not is_int(size) or size < 0 or not is_hex64(sha)
                    or not isinstance(parts, list) or any(f['path'] == path for f in files)):
                raise ValueError('file')
            total = 0
            for name in parts:
                if name not in objects or name in used or name == INDEX_NAME or objects[name][0] != role:
                    raise ValueError('part')
                used.add(name)
                total += objects[name][2]
            if total != size:
                raise ValueError('size')
            files.append({'path': path, 'role': role, 'size': size, 'sha256': sha, 'parts': list(parts)})
        if set(objects) - used - {INDEX_NAME}:
            raise ValueError('object without a file')
        info = {key: doc.get(key) for key in ('checkpoint_id', 'fp32_entry', 'fp32_master_digest')}
    except (ValueError, KeyError, TypeError, RecursionError):
        raise JobError('the file index of the base state is not valid') from None
    return files, info


def target_of(path, places):
    """Where a file of the state goes on this box: (file, directories that get a hard link)."""
    prefix, _, name = path.rpartition('/')
    prefix = prefix + '/' if prefix else ''
    for rule in places:
        if rule['prefix'] == prefix:
            return os.path.join(rule['dir'], name), [os.path.join(link, name) for link in rule.get('links', [])]
    raise JobError('the job names no place for %s' % path)


def fetch_file(entry, objects, urls, places):
    target, links = target_of(entry['path'], places)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + '.part'
    file_sha = hashlib.sha256()
    with open(partial, 'wb') as out:
        for name in entry['parts']:
            _role, sha, size = objects[name]
            if name not in urls:
                raise JobError('no URL for object %s' % name)
            file_sha = http_get_into(urls[name], out, size, sha, file_sha, 'object %s' % name)
        out.flush()
        os.fsync(out.fileno())
    if os.path.getsize(partial) != entry['size'] or file_sha.hexdigest() != entry['sha256']:
        os.remove(partial)
        raise JobError('file %s is not the file of the index' % entry['path'])
    os.replace(partial, target)
    for link in links:
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.remove(link)
        try:
            os.link(target, link)
        except OSError:
            shutil.copyfile(target, link)
    return entry['size']


def open_base(job):
    """Read the manifest and the file index of the base state, and check both.
    No weight is downloaded here. -> (objects of the manifest, files, info of the index)."""
    base = job['base']
    raw = http_get_bytes(base['manifest'], MAX_MANIFEST_BYTES, 'manifest')
    objects, meta = parse_manifest(raw, job['base_state'])
    expected = job['expected_fp32_master_digest']
    if meta.get(META_FP32_MASTER_DIGEST) != expected:
        raise JobError('the fp32 master digest of the base is not the digest that the lease expects')
    if INDEX_NAME not in objects or INDEX_NAME not in base['objects']:
        raise JobError('the base state has no file index')
    _role, index_sha, index_size = objects[INDEX_NAME]
    if index_size > MAX_INDEX_BYTES:
        raise JobError('the file index is too large')
    index_raw = http_get_bytes(base['objects'][INDEX_NAME], MAX_INDEX_BYTES, 'index')
    if len(index_raw) != index_size or sha256_hex(index_raw) != index_sha:
        raise JobError('the file index is not the object of the manifest')
    files, info = parse_index(index_raw, objects)
    if info.get('fp32_master_digest') != expected:
        raise JobError('the file index names another fp32 master digest')
    return objects, files, info


def fetch_files(job, objects, files, workers):
    """Download every file of the base state to its place. -> bytes."""
    urls, places = job['base']['objects'], job['place']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(lambda entry: fetch_file(entry, objects, urls, places), files))


def lookup(doc, keys):
    for key in keys:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def check_expect(job):
    """Facts about the files on the box that must hold before any step runs."""
    for check in job.get('expect', []):
        try:
            with open(check['file'], 'rb') as handle:
                doc = json.loads(handle.read())
        except (OSError, ValueError):
            raise JobError('cannot read %s' % os.path.basename(check['file'])) from None
        if lookup(doc, check['key']) != check['equals']:
            raise JobError('%s: %s is not what the lease expects' % (
                os.path.basename(check['file']), '.'.join(check['key'])))


# ----------------------------------------------------------------------------
# The box
# ----------------------------------------------------------------------------

def check_box(job):
    """-> facts about this box. Raises ``JobError`` if the box is too small."""
    require = job.get('require', {})
    facts = {}
    if require.get('gpus') or require.get('gpu_name'):
        argv = require.get('gpu_query') or ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            raise JobError('cannot list the GPUs') from None
        names = [line.strip() for line in out.splitlines() if line.strip()]
        facts['gpus'] = names
        if require.get('gpus') and len(names) != require['gpus']:
            raise JobError('the box has %d GPUs, the job needs %d' % (len(names), require['gpus']))
        want = re.sub(r'[^A-Z0-9]', '', str(require.get('gpu_name') or '').upper())
        if want and any(want not in re.sub(r'[^A-Z0-9]', '', name.upper()) for name in names):
            raise JobError('the box has another GPU class than the job needs')
    if require.get('ram_gb'):
        try:
            with open(require.get('meminfo') or '/proc/meminfo') as handle:
                found = re.search(r'^MemTotal:\s+(\d+)\s+kB', handle.read(), re.M)
        except OSError:
            found = None
        if found is None:
            raise JobError('cannot read the host RAM')
        facts['ram_gb'] = round(int(found.group(1)) / (1024.0 * 1024.0), 1)
        if facts['ram_gb'] < require['ram_gb']:
            raise JobError('the box has %.0f GB RAM, the job needs %d' % (facts['ram_gb'], require['ram_gb']))
    if require.get('disk_free_gb'):
        where = require.get('disk_path') or os.path.dirname(job['_path'])
        facts['disk_free_gb'] = round(shutil.disk_usage(where).free / float(1 << 30), 1)
        if facts['disk_free_gb'] < require['disk_free_gb']:
            raise JobError('the box has %.0f GB free disk, the job needs %d' % (
                facts['disk_free_gb'], require['disk_free_gb']))
    return facts


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------

def group_alive(group):
    """Does a process of this group still run? A process that ended and that
    nobody collected yet (a zombie) does not count: a signal cannot end it."""
    try:
        os.killpg(group, 0)
    except OSError:
        return False
    try:
        pids = [name for name in os.listdir('/proc') if name.isdigit()]
    except OSError:
        return True
    for pid in pids:
        try:
            with open('/proc/%s/stat' % pid) as handle:
                state, _parent, pgrp = handle.read().rpartition(')')[2].split()[:3]
        except (OSError, ValueError):
            continue
        if pgrp == str(group) and state != 'Z':
            return True
    return False


class Runner:
    """Runs the steps. Keeps the one process group that is running so that a
    deadline can end it."""

    def __init__(self, job, work):
        self.job = job
        self.work = work
        self.group = None
        self.lock = threading.Lock()

    def environment(self, step):
        env = dict(os.environ)
        env.update(self.job.get('env', {}))
        env.update(step.get('env', {}))
        env.pop(ENV_BOOTSTRAP_OVERRIDE, None)
        for key in [key for key in env if 'BOOTSTRAP_OVERRIDE' in key.upper()]:
            env.pop(key)
        env[ENV_EXPECTED_MASTER] = self.job['expected_fp32_master_digest']
        return env

    def script(self, step):
        text = step['script']
        for token, source in step.get('subst', {}).items():
            try:
                with open(source['file'], 'rb') as handle:
                    value = lookup(json.loads(handle.read()), source['key'])
            except (OSError, ValueError):
                raise JobError('step %s: cannot read %s' % (step['name'], os.path.basename(source['file']))) from None
            if not isinstance(value, str) or re.fullmatch(source['match'], value) is None:
                raise JobError('step %s: %s has no usable %s' % (
                    step['name'], os.path.basename(source['file']), '.'.join(source['key'])))
            text = text.replace(token, value)
        if SETS_OVERRIDE.search(text) is not None:
            raise JobError('step %s: the bootstrap override is never set' % step['name'])
        return text

    def kill(self):
        with self.lock:
            group = self.group
        if group is None:
            return
        for sig, wait in ((signal.SIGTERM, 20.0), (signal.SIGKILL, 5.0)):
            if not group_alive(group):
                return
            try:
                os.killpg(group, sig)
            except OSError:
                return
            end = time.time() + wait
            while time.time() < end and group_alive(group):
                time.sleep(0.1)

    def run(self, step):
        """Run one step to its end. Raises ``NotTrained`` if the step failed
        and its output says that the batch is at fault, ``JobError`` for
        every other failure."""
        name = step['name']
        logs = os.path.join(self.work, 'logs')
        os.makedirs(logs, exist_ok=True)
        path = os.path.join(logs, name + '.log')
        script = os.path.join(logs, name + '.sh')
        with open(os.open(script, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700), 'w') as handle:
            handle.write(self.script(step))
        with open(path, 'ab') as out:
            proc = subprocess.Popen(
                ['bash', script], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                env=self.environment(step), cwd=step.get('cwd') or self.work, start_new_session=True,
            )
            with self.lock:
                self.group = proc.pid
            try:
                code = proc.wait(timeout=step['timeout_s'])
            except subprocess.TimeoutExpired:
                self.kill()
                proc.wait()
                raise JobError('step %s ran longer than %ds' % (name, step['timeout_s'])) from None
            finally:
                # Whatever the step left behind in its group goes with it.
                self.kill()
                with self.lock:
                    self.group = None
        if code == 0:
            return
        tail = b''
        try:
            with open(path, 'rb') as handle:
                handle.seek(max(0, os.path.getsize(path) - 65536))
                tail = handle.read()
        except OSError:
            pass
        text = tail.decode('utf-8', 'replace')
        for pattern in step.get('batch_caused', []):
            if re.search(pattern, text) is not None:
                raise NotTrained('batch_rejected:%s' % name)
        raise JobError('step %s ended with code %d' % (name, code))


# ----------------------------------------------------------------------------
# The new state
# ----------------------------------------------------------------------------

def load_box_state(path, job):
    """What the training step says it made. Every rule fails closed."""
    try:
        with open(path, 'rb') as handle:
            doc = json.loads(handle.read())
        if doc['schema'] != BOX_STATE_SCHEMA:
            raise ValueError('schema')
        trainer, meta, files = doc['trainer'], doc['meta'], doc['files']
        if not isinstance(trainer, dict) or not isinstance(meta, dict) or not isinstance(files, list) or not files:
            raise ValueError('shape')
        seen = set()
        for item in files:
            if (not check_path(item['path']) or item['path'] in seen or item['role'] not in ROLES
                    or not is_abs(item['source']) or not os.path.isfile(item['source'])
                    or not (item.get('sha256') is None or is_hex64(item['sha256']))):
                raise ValueError('file')
            seen.add(item['path'])
        for key, value in meta.items():
            if (not isinstance(key, str) or META_KEY.fullmatch(key) is None
                    or not ((isinstance(value, str) and len(value) <= 256) or is_int(value))):
                raise ValueError('meta')
        for key in ('checkpoint_id', 'fp32_entry'):
            # The file index holds these as text. The next box reads them there.
            if key in meta and not (isinstance(meta[key], str) and meta[key]):
                raise ValueError('meta')
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        raise JobError('the state file of the training step is not valid') from None
    expected = job['expected_fp32_master_digest']
    # The trainer must have read the master of the base, and the master it
    # wrote must be the one that the state names.
    if trainer.get('input_master_digest') != expected:
        raise JobError('the trainer did not start from the fp32 master of the base')
    if not is_hex64(trainer.get('output_master_digest')) or meta.get(META_FP32_MASTER_DIGEST) != trainer[
            'output_master_digest']:
        raise JobError('the new state does not name the fp32 master that the trainer wrote')
    for role in REQUIRED_ROLES:
        if not any(item['role'] == role for item in files):
            raise JobError('the new state has no %s file' % role)
    return meta, files


def measure(source, chunk):
    """One read of a file: its size, its sha256, and (offset, size, sha256) of every part."""
    whole, parts, offset = hashlib.sha256(), [], 0
    with open(source, 'rb') as handle:
        while True:
            part, got = hashlib.sha256(), 0
            while got < chunk:
                data = handle.read(min(BLOCK, chunk - got))
                if not data:
                    break
                got += len(data)
                part.update(data)
                whole.update(data)
            if not got:
                break
            parts.append((offset, got, part.hexdigest()))
            offset += got
    return offset, whole.hexdigest(), parts


def upload_state(job, meta, files, workers):
    """Upload the files of the new state in parts, then the index.
    -> the manifest document of the new state."""
    chunk, slots, content_type = job['chunk_bytes'], job['put']['objects'], job['content_type']
    measured = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item, (size, sha, parts) in zip(files, pool.map(lambda item: measure(item['source'], chunk), files)):
            if item.get('sha256') is not None and item['sha256'] != sha:
                # The step that made the file named its sha256. The file on the disk is another one.
                raise JobError('file %s is not the file that its step named' % item['path'])
            measured.append((item, size, sha, parts))
    objects, index_files, uploads, number = [], [], [], 0
    for item, size, sha, parts in sorted(measured, key=lambda entry: entry[0]['path']):
        names = []
        for offset, part_size, part_sha in parts:
            name = SLOT_NAME % number
            number += 1
            if name not in slots:
                raise JobError('the new state needs more parts than the lease has slots')
            names.append(name)
            objects.append({'name': name, 'role': item['role'], 'sha256': part_sha, 'size': part_size})
            uploads.append((slots[name], item['source'], offset, part_size, part_sha, name))
        index_files.append({'path': item['path'], 'role': item['role'], 'size': size, 'sha256': sha, 'parts': names})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda up: http_put_slice(up[0], up[1], up[2], up[3], up[4], content_type, 'part ' + up[5]),
                      uploads))
    index = canonical({
        'schema': FILES_SCHEMA, 'files': index_files,
        'checkpoint_id': meta.get('checkpoint_id'), 'fp32_entry': meta.get('fp32_entry'),
        'fp32_master_digest': meta[META_FP32_MASTER_DIGEST],
    })
    if len(index) > MAX_INDEX_BYTES:
        raise JobError('the file index is too large')
    http_put_bytes(slots[INDEX_NAME], index, content_type, 'index')
    objects.append({'name': INDEX_NAME, 'role': 'aux', 'sha256': sha256_hex(index), 'size': len(index)})
    if len(objects) > MAX_OBJECTS:
        raise JobError('the new state has too many objects')
    return {
        'schema': STATE_SCHEMA, 'parent': job['base_state'], 'batch_id': job['batch_id'],
        'objects': sorted(objects, key=lambda entry: entry['name']), 'meta': dict(meta),
    }


def load_outcome(path):
    try:
        with open(path, 'rb') as handle:
            doc = json.loads(handle.read())
        if (doc['schema'] != BOX_OUTCOME_SCHEMA or doc['outcome'] not in (OUTCOME_PROBED, OUTCOME_NOT_TRAINED)
                or not isinstance(doc.get('reason', ''), str) or not isinstance(doc.get('payload', {}), dict)):
            raise ValueError('outcome')
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        raise JobError('the outcome file of the last step is not valid') from None
    return doc['outcome'], doc.get('reason', ''), doc.get('payload', {})


# ----------------------------------------------------------------------------
# The job
# ----------------------------------------------------------------------------

class Heartbeat(threading.Thread):
    """Writes the heartbeat with a growing number until it is stopped. The
    work never waits for a heartbeat: it only names the stage."""

    def __init__(self, job):
        super().__init__(name='heartbeat', daemon=True)
        self.job = job
        self.stage = 'start'
        self.seq = HEARTBEAT_BASE
        self.wake = threading.Event()
        self.stopped = threading.Event()

    def at(self, stage):
        self.stage = stage
        self.wake.set()

    def beat(self):
        self.seq += 1
        body = canonical({
            'schema': HEARTBEAT_SCHEMA, 'fence': self.job['fence'], 'seq': self.seq, 'stage': self.stage[:200],
        })
        try:
            http_put_bytes(self.job['put']['heartbeat'], body, self.job['content_type'], 'heartbeat', tries=1)
        except JobError as exc:
            log('heartbeat not written: %s' % exc)

    def run(self):
        while not self.stopped.is_set():
            self.beat()
            self.wake.wait(self.job['heartbeat_every_s'])
            self.wake.clear()

    def stop(self, stage):
        """Stop the thread and write the last heartbeat."""
        self.stage = stage
        self.stopped.set()
        self.wake.set()
        self.join(HTTP_TIMEOUT_S + 10)
        if not self.is_alive():
            self.beat()


def wipe(job, work):
    """Remove the weights, the inputs and the signed URLs from this box."""
    for path in list(job.get('wipe', [])) + [job['_path']]:
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.lexists(path):
                os.remove(path)
        except OSError:
            pass


def result_body(job, outcome, reason, payload, base_info, manifest, facts, seconds):
    candidate = None
    if manifest is not None:
        candidate = {
            'state_id': sha256_hex(canonical(manifest)),
            'checkpoint_id': manifest['meta'].get('checkpoint_id'),
            'fp32_master_digest': manifest['meta'][META_FP32_MASTER_DIGEST],
        }
    body = canonical({
        'schema': RESULT_SCHEMA, 'run_id': job['run_id'], 'batch_id': job['batch_id'], 'fence': job['fence'],
        'attempt': job['attempt'], 'base_state': job['base_state'], 'base_seq': job['base_seq'],
        'outcome': outcome, 'reason': reason,
        'base': {
            'checkpoint_id': (base_info or {}).get('checkpoint_id'),
            'fp32_master_digest': job['expected_fp32_master_digest'],
        },
        'binding': job.get('binding'), 'candidate': candidate, 'payload': payload, 'box': facts,
        'seconds': {key: round(value, 1) for key, value in seconds.items()},
    })
    if len(body) > MAX_RESULT_BYTES:
        raise JobError('the result is larger than %d bytes' % MAX_RESULT_BYTES)
    return body


def commit(job, body, manifest):
    """``result.json`` first, ``COMPLETE.json`` last."""
    put, content_type = job['put'], job['content_type']
    http_put_bytes(put['result'], body, content_type, 'result')
    complete = canonical({
        'schema': COMPLETE_SCHEMA, 'batch_id': job['batch_id'], 'fence': job['fence'],
        'base_state': job['base_state'], 'result': {'sha256': sha256_hex(body), 'size': len(body)},
        'state': manifest,
    })
    if len(complete) > MAX_COMPLETE_BYTES:
        raise JobError('COMPLETE.json is larger than %d bytes' % MAX_COMPLETE_BYTES)
    http_put_bytes(put['complete'], complete, content_type, 'complete')


def work_of(job, runner, heart, facts, seconds):
    """Everything between the claim and the commit. -> (outcome, reason, payload, base info, manifest)."""
    workers = int(job.get('workers') or 8)
    base_info = None
    upload = {}

    def timed(name, call):
        heart.at(name)
        started = time.time()
        try:
            return call()
        finally:
            seconds[name] = seconds.get(name, 0.0) + time.time() - started

    def start_upload(path):
        meta, files = load_box_state(path, job)

        def body():
            try:
                upload['manifest'] = upload_state(job, meta, files, workers)
            except BaseException as exc:
                upload['error'] = exc
        upload['thread'] = threading.Thread(target=body, name='upload', daemon=True)
        upload['started'] = time.time()
        upload['thread'].start()

    try:
        facts.update(timed('checks', lambda: check_box(job)))
        objects, files, base_info = timed('base', lambda: open_base(job))
        binding = job.get('binding')
        if binding is not None and binding['checkpoint_id'] != base_info.get('checkpoint_id'):
            # Known before one weight is downloaded.
            raise NotTrained(REASON_OTHER_CHECKPOINT)
        facts['base_bytes'] = timed('fetch', lambda: fetch_files(job, objects, files, workers))
        check_expect(job)
        for step in job['steps']:
            timed('step:' + step['name'], lambda: runner.run(step))
            if step.get('state'):
                start_upload(step['state'])
        outcome, reason, payload = (OUTCOME_PROBED, '', {})
        if job.get('outcome'):
            outcome, reason, payload = load_outcome(job['outcome'])
        if outcome == OUTCOME_NOT_TRAINED:
            raise NotTrained(reason or 'not_trained')
        if 'thread' not in upload:
            raise JobError('no step wrote a state')
        heart.at('upload')
        upload['thread'].join()
        seconds['upload'] = time.time() - upload['started']
        if 'error' in upload:
            error = upload['error']
            raise error if isinstance(error, JobError) else JobError('upload: %s' % type(error).__name__)
        return outcome, reason, payload, base_info, upload['manifest']
    except NotTrained as exc:
        # No state is committed. An upload that runs is left to end; its objects are garbage of this lease.
        return OUTCOME_NOT_TRAINED, str(exc), {}, base_info, None


def main(argv):
    if len(argv) != 2:
        sys.stderr.write('usage: job.py job.json\n')
        return EXIT_BAD_JOB
    path = os.path.abspath(argv[1])
    work = os.path.dirname(path)
    try:
        with open(path, 'rb') as handle:
            job = json.loads(handle.read())
        check_job(job)
        for wiped in job.get('wipe', []):
            # The claim is a directory in ``work``. It must outlive the job.
            need(work != wiped and not work.startswith(wiped + '/'), 'wipe: the directory of the lease must stay')
    except (OSError, ValueError, BadJob) as exc:
        log('bad job: %s' % exc)
        return EXIT_BAD_JOB
    job['_path'] = path
    use_http_settings(job)
    try:
        # One start per lease on this box. The directory is the claim.
        os.mkdir(os.path.join(work, 'claimed'))
    except FileExistsError:
        log('this lease already started on this box')
        return EXIT_STARTED
    started = time.time()
    runner = Runner(job, work)
    heart = Heartbeat(job)

    def deadline():
        log('the job ran longer than its lifetime: stop')
        runner.kill()
        wipe(job, work)
        os._exit(EXIT_DEADLINE)

    def stopped_from_outside(_signum, _frame):
        log('told to stop')
        runner.kill()
        wipe(job, work)
        os._exit(EXIT_FAILED)

    timer = threading.Timer(job['job_ttl_s'], deadline)
    timer.daemon = True
    timer.start()
    signal.signal(signal.SIGTERM, stopped_from_outside)
    signal.signal(signal.SIGINT, stopped_from_outside)
    heart.start()
    facts, seconds = {}, {}
    code = EXIT_FAILED
    try:
        outcome, reason, payload, base_info, manifest = work_of(job, runner, heart, facts, seconds)
        seconds['total'] = time.time() - started
        heart.at('commit')
        commit(job, result_body(job, outcome, reason, payload, base_info, manifest, facts, seconds), manifest)
        log('committed: %s %s' % (outcome, reason))
        code = EXIT_DONE
    except JobError as exc:
        log('failed, no decision: %s' % exc)
    except Exception as exc:
        log('failed, no decision: %s: %s' % (type(exc).__name__, exc))
    finally:
        timer.cancel()
        runner.kill()
        wipe(job, work)
        heart.stop('done' if code == EXIT_DONE else 'failed')
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv))
