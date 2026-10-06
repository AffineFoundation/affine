"""Default-off trusted CPU service supervision, separate from job authority.

A ROOT lifetime grant replaces neither a historical scope nor a miner proof.
It authorizes only exact CPU service entrypoints and bounded recovery after exit.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from nacl.signing import SigningKey
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ops.running_api_validation import authenticate, canonical, digest, file_bytes, sign, write_exclusive


VERSION = 'trusted-cpu-service-lifetime-authorization-v1'
STATUS = 'trusted-cpu-service-authorization-status-v1'
INSTANCE = 'trusted-cpu-service-instance-v1'


def process(pid):
    if pid <= 0:
        return None
    root = Path('/proc', str(pid))
    try:
        ticks = int((root/'stat').read_text().rsplit(')', 1)[1].split()[19])
        boot = int(next(row.split()[1] for row in Path('/proc/stat').read_text().splitlines() if row.startswith('btime ')))
        return dict(pid=pid, ticks=ticks, started_at=boot+ticks/os.sysconf('SC_CLK_TCK'),
                    argv=[v.decode() for v in (root/'cmdline').read_bytes().split(b'\0') if v])
    except FileNotFoundError:
        return None


def unit(name):
    return dict(row.split('=', 1) for row in subprocess.check_output(
        ['systemctl', '--user', 'show', name, '-p', 'MainPID', '-p', 'InvocationID', '-p', 'ActiveState'],
        text=True).splitlines())


def current(service):
    value = unit(service['unit'])
    return dict(systemd=value, process=process(int(value['MainPID'])))


def key_for(path, authority):
    path = Path(path)
    raw = file_bytes(path)
    if path.stat().st_mode & 0o077:
        raise ValueError('private local authority')
    key = SigningKey(bytes.fromhex(raw.decode().strip()))
    if key.verify_key.encode().hex() != authority:
        raise ValueError('exact original local authority')
    return key


def validate(envelope, authority, *, now=None, require_execution=False):
    now = time.time() if now is None else now
    policy = authenticate(envelope, authority)
    if (policy.get('version') != VERSION or policy.get('job_permissions_unchanged') is not True or
            policy.get('scientific_contracts_unchanged') is not True or policy.get('GPU_execution_allowed') is not False or
            policy.get('created_at', now+1) > now or policy.get('authority') != authority):
        raise ValueError('explicit trusted CPU lifetime authorization')
    if require_execution and policy.get('execute_allowed') is not True:
        raise ValueError('default-off service recovery')
    if policy['not_after'] is not None and now >= policy['not_after']:
        raise ValueError('lifetime authorization expired')
    status = authenticate(json.loads(file_bytes(policy['authorization_status_path'])), authority)
    if status['version'] != STATUS or status['active_policy_sha256'] != digest(policy):
        raise ValueError('lifetime authorization revoked or superseded')
    if policy['host_uid'] != os.getuid() or hashlib.sha256(file_bytes('/etc/machine-id')).hexdigest() != policy['host_machine_id_sha256']:
        raise ValueError('same trusted host and UID')
    for path, pin in policy['files'].items():
        if hashlib.sha256(file_bytes(path)).hexdigest() != pin:
            raise ValueError('pinned service/package/source/auth file changed')
    if policy['entrypoint'] != str(Path(__file__).resolve()) or policy['entrypoint'] not in policy['files']:
        raise ValueError('pinned exact lifetime entrypoint')
    q = Path(policy['database'])
    st = q.stat()
    if q != q.resolve() or [st.st_dev, st.st_ino] != policy['queue_inode']:
        raise ValueError('original queue inode')
    if set(policy['services']) != {'api', 'auditor'}:
        raise ValueError('exact two CPU service roles')
    seen = set()
    for role, service in policy['services'].items():
        if service['unit'] in seen or service['role'] != role:
            raise ValueError('distinct exact service units')
        seen.add(service['unit'])
        config = json.loads(file_bytes(service['config_path']))
        historical = policy['historical_execution_scopes'][role]
        if policy['files'].get(historical['path']) != historical['sha256']:
            raise ValueError('immutable historical execution scope')
        previous = authenticate(json.loads(file_bytes(historical['path'])), authority)
        initial = service['initial_instance']['process']
        if (initial['argv'][initial['argv'].index('--scope')+1] != historical['path'] or
                not previous['created_at'] <= initial['started_at'] < previous['expires_at'] or
                previous['queue_inode'] != policy['queue_inode']):
            raise ValueError('original instance execution provenance')
        if digest(config) != service['config_sha256'] or str(Path(config['state'])/'roles/verifier-queue.sqlite3') != policy['database']:
            raise ValueError('unchanged service config and queue')
        credential = config['bucket']['credentials_file']
        if credential not in policy['files'] or Path(credential).stat().st_mode & 0o077:
            raise ValueError('pinned private R2 authentication file')
        count = 173 if role == 'api' else 2102
        if len(service['operator_files']) != count:
            raise ValueError('full exact CPU operator inventory')
        for name, pin in service['operator_files'].items():
            path = str(Path(service['operator_tree'])/name)
            if policy['files'].get(path) != pin:
                raise ValueError('complete CPU operator pin coverage')
        for path in [service['config_path'], service['unit_fragment'], *service['dropins'], *service['environment_files']]:
            if path not in policy['files']:
                raise ValueError('complete service/config/R2 environment pin coverage')
        if service['expected_argv'] != [policy['python'], '-I', '-B', policy['entrypoint'], 'run-service',
                                         '--policy', policy['policy_path'], '--authority', authority, '--role', role]:
            raise ValueError('exact lifetime entrypoint argv')
        if not 1 <= service['max_failures'] <= 5 or not 60 <= service['circuit_seconds'] <= 3600:
            raise ValueError('bounded restart circuit')
        if not 1 <= service['backoff_seconds'] <= service['max_backoff_seconds'] <= 300:
            raise ValueError('bounded restart backoff')
        if not 10 <= service['readiness_seconds'] <= (120 if role == 'api' else 900):
            raise ValueError('bounded genuine readiness')
    # Source admission is independent of service authority and remains signed.
    api = policy['services']['api']
    if api['registry_path'] not in policy['files']:
        raise ValueError('pinned API source registry')
    registry = authenticate(json.loads(file_bytes(api['registry_path'])), authority)
    if registry.get('version') != 'source-specific-sampling-api-admission-v1' or set(registry) != {'version', 'sources'}:
        raise ValueError('exact source-specific API registry')
    approved = {source: row['runtime_files'] for source, row in registry['sources'].items()}
    api_config = json.loads(file_bytes(api['config_path']))
    if api['readiness_url'] != 'http://127.0.0.1:'+str(api_config['remote']['verifier_queue']['port'])+'/request':
        raise ValueError('exact local non-mutating readiness endpoint')
    audit_config = json.loads(file_bytes(policy['services']['auditor']['config_path']))
    audit = authenticate(audit_config['continuous_audit_service']['source_admission'], authority)
    if approved != audit['approved_sources']:
        raise ValueError('same source admission union')
    for source, pins in approved.items():
        for name, pin in pins.items():
            path = str(Path(api['source_trees'][source])/name)
            if policy['files'].get(path) != pin:
                raise ValueError('complete scientific source pins')
    key_for(policy['authority_seed_path'], authority)
    return policy


def api_ready(service):
    request = urllib.request.Request(service['readiness_url'], data=b'{}', method='POST',
                                     headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(request, timeout=2).close()
    except urllib.error.HTTPError as error:
        try:
            return error.code == 403 and json.loads(error.read(4096)) == {'error': 'request rejected'}
        except (json.JSONDecodeError, OSError):
            return False
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    return False


def auditor_ready(policy, service, since, authority):
    """Authentic producer progress; completed tick remains a separate metric."""
    path = Path(service['state_path'])
    healthpath = Path(service['health_path'])
    if healthpath.exists() and healthpath.stat().st_mtime >= since:
        health = json.loads(file_bytes(healthpath))
        if health.get('at', 0) >= since:
            return dict(producer_progress=False, completed_tick=True, health_at=health['at'])
    if not path.exists() or path.stat().st_mtime < since:
        return None
    state = json.loads(file_bytes(path))
    config = json.loads(file_bytes(service['config_path']))
    admission = authenticate(config['continuous_audit_service']['source_admission'], authority)
    for job_id, row in state.get('jobs', {}).items():
        jobpath = path.parent/(job_id+'-job.json')
        if not jobpath.exists() or jobpath.stat().st_mtime < since:
            continue
        job = authenticate(json.loads(file_bytes(jobpath)), authority)
        manifest = authenticate(job['manifest'], authority)
        source = manifest['source_bundle']['sha256']
        if (job['job_id'] != job_id or job.get('role') != 'verify' or job['created_at'] < since or
                row.get('job_sha256') != digest(job) or
                job.get('source_files') != admission['approved_sources'].get(source)):
            continue
        db = None
        try:
            db = sqlite3.connect(Path(policy['database']).as_uri()+'?mode=ro', uri=True, timeout=.25)
            actual = db.execute('SELECT digest,status FROM jobs WHERE id=?', (job_id,)).fetchone()
        except sqlite3.OperationalError as error:
            if any(word in str(error).lower() for word in ('locked', 'busy')):
                return None
            raise
        finally:
            if db is not None:
                db.close()
        if actual and actual[0] == digest(job) and actual[1] in ('queued', 'leased', 'complete'):
            health = json.loads(file_bytes(service['health_path'])) if Path(service['health_path']).exists() else {}
            return dict(job_id=job_id, job_sha256=actual[0], status=actual[1], source=source,
                        producer_progress=True, completed_tick=health.get('at', 0) >= since)
    return None


def readiness(policy, role, since, authority):
    service = policy['services'][role]
    return api_ready(service) if role == 'api' else auditor_ready(policy, service, since, authority)


class Supervisor:
    """Injected observations/actions make every recovery failure path testable."""
    def __init__(self, policy, role, observe=current, start=None, stop=None, ready=None,
                 record=None, guard=None, clock=time.monotonic, wall=time.time, sleep=time.sleep):
        self.policy = policy
        self.role = role
        self.service = policy['services'][role]
        self.observe = observe
        self.start = start or (lambda: subprocess.run(['systemctl', '--user', 'start', self.service['unit']], check=True, timeout=45))
        self.stop = stop or (lambda: subprocess.run(['systemctl', '--user', 'stop', self.service['unit']], check=True, timeout=45))
        self.ready = ready or (lambda since: readiness(policy, role, since, policy['authority']))
        self.record = record or (lambda value: None)
        self.guard = guard or (lambda: None)
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.failures = []
        self.attempts = 0
        self.instance = None
        self.initial = self.service['initial_instance']
        self.pending = None
        self.circuit_recorded = False

    def allowed(self, value):
        proc = value['process']
        if not proc:
            return False
        if value == self.initial:
            return True
        return (value['systemd']['ActiveState'] == 'active' and
                value['systemd']['MainPID'] == str(proc['pid']) and
                proc['argv'] == self.service['expected_argv'])

    def step(self):
        value = self.observe(self.service)
        if value['process']:
            if not self.allowed(value):
                raise ValueError('unrelated live service instance preserved')
            if self.instance != value and value != self.initial:
                self.guard()
                proof = self.ready(value['process'].get('started_at', self.wall()))
                if not proof:
                    if self.pending is None or self.pending[0] != value:
                        self.pending = (value, self.clock()+self.service['readiness_seconds'])
                    if self.clock() < self.pending[1]:
                        return 'awaiting_readiness'
                    if self.observe(self.service) != value:
                        raise ValueError('changed unready instance preserved')
                    self.stop()
                    self.failures.append(self.clock())
                    self.record(dict(event='unready_instance_retired', at=self.wall(), instance=value))
                    return 'failed'
            if self.instance != value:
                self.record(dict(event='instance_observed', at=self.wall(), instance=value,
                                 adopted_original=value == self.initial))
                self.instance = value
                self.pending = None
            return 'live'
        if value['systemd']['MainPID'] != '0':
            raise ValueError('inconclusive process absence')
        now = self.clock()
        self.failures = [at for at in self.failures if now-at < self.service['circuit_seconds']]
        if len(self.failures) >= self.service['max_failures']:
            if not self.circuit_recorded:
                self.record(dict(event='circuit_open', at=self.wall(), failures=len(self.failures)))
                self.circuit_recorded = True
            return 'circuit_open'
        self.circuit_recorded = False
        delay = min(self.service['max_backoff_seconds'], self.service['backoff_seconds']*2**len(self.failures))
        self.sleep(delay)
        # Never start over an instance that appeared during backoff.
        latest = self.observe(self.service)
        if latest != value:
            if latest['process'] and self.allowed(latest):
                return 'live'
            raise ValueError('service changed during backoff')
        self.guard()
        since = self.wall()
        self.attempts += 1
        # Count every restart, including short-lived instances that bind then crash.
        self.failures.append(self.clock())
        self.record(dict(event='restart_attempt', at=since, attempt=self.attempts))
        try:
            self.start()
            deadline = self.clock()+self.service['readiness_seconds']
            while self.clock() < deadline:
                fresh = self.observe(self.service)
                if fresh['process']:
                    if not self.allowed(fresh) or fresh == self.initial:
                        raise ValueError('unrelated restart instance preserved')
                    proof = self.ready(since)
                    if proof and self.observe(self.service) == fresh:
                        self.guard()
                        self.instance = fresh
                        self.record(dict(event='restart_ready', at=self.wall(), instance=fresh, readiness=proof))
                        return 'restarted'
                elif fresh['systemd']['ActiveState'] == 'failed':
                    break
                self.sleep(.25)
            # Only stop the exact new CPU instance we observed, never an original.
            last = self.observe(self.service)
            if last['process'] and last != self.initial and self.allowed(last):
                if self.observe(self.service) != last:
                    raise ValueError('changed unready instance preserved')
                self.stop()
            raise TimeoutError('genuine service readiness absent')
        except (subprocess.SubprocessError, TimeoutError):
            last = self.observe(self.service)
            if last['process']:
                if last == self.initial or not self.allowed(last):
                    raise ValueError('unrelated failed-start instance preserved')
                proof = self.ready(since)
                if proof and self.observe(self.service) == last:
                    self.guard()
                    self.instance = last
                    self.record(dict(event='restart_ready_after_start_timeout', at=self.wall(), instance=last, readiness=proof))
                    return 'restarted'
                if self.observe(self.service) != last:
                    raise ValueError('changed failed-start instance preserved')
                self.stop()
            self.record(dict(event='restart_failed', at=self.wall(), attempt=self.attempts))
            return 'failed'


def assemble_service(policy, role):
    service = policy['services'][role]
    for name in list(sys.modules):
        if name == 'subnet' or name.startswith('subnet.'):
            del sys.modules[name]
    sys.path.insert(0, service['operator_tree'])
    if role == 'api':
        api = importlib.import_module('subnet.coordinator_api_service')
        guard = importlib.import_module('subnet.source_sampling_admission')
        if Path(api.__file__).resolve() != Path(service['operator_tree'])/'subnet/coordinator_api_service.py' or Path(guard.__file__).resolve() != Path(service['operator_tree'])/'subnet/source_sampling_admission.py':
            raise ValueError('exact original API/guard import origins')
        admission = guard.SamplingAdmission(json.loads(file_bytes(service['registry_path'])), policy['authority'], service['source_trees'])
        api.Coordinator = guard.guarded_coordinator(api.Coordinator, admission)
        return lambda: api.main(['--config', service['config_path'], '--authority-seed', policy['authority_seed_path'],
                                 '--expected-authority', policy['authority']])
    queue = importlib.import_module('subnet.distributed_roles')
    config = json.loads(file_bytes(service['config_path']))
    original = queue.Coordinator.__init__
    def scoped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for identity in config['historical_trusted_worker_identities']:
            self.workers.setdefault(identity, [])
        for identity in config['inactive_claim_worker_identities']:
            self.workers[identity] = []
    queue.Coordinator.__init__ = scoped
    # Reviewed pure read-only status fix, without changing the sealed CPU tree.
    from ops.readonly_coordinator_status import status
    queue.Coordinator.status = status
    auditor = importlib.import_module('subnet.continuous_audit_service')
    if Path(auditor.__file__).resolve() != Path(service['operator_tree'])/'subnet/continuous_audit_service.py':
        raise ValueError('exact original auditor import origin')
    auditor.admitted_service_config(config['continuous_audit_service'], policy['authority'])
    return lambda: auditor.main(['--config', service['config_path']])


def run_service(policy, role):
    service = policy['services'][role]
    live = current(service)
    if (live['process'] != process(os.getpid()) or live['process']['argv'] != service['expected_argv'] or
            live['systemd']['ActiveState'] != 'active'):
        raise ValueError('only authorized systemd service invocation')
    return assemble_service(policy, role)()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'run-service', 'supervise'])
    parser.add_argument('--policy', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--role', choices=['api', 'auditor'], required=True)
    args = parser.parse_args()
    envelope = json.loads(file_bytes(args.policy))
    policy = validate(envelope, args.authority, require_execution=args.action != 'check')
    if str(Path(args.policy)) != policy['policy_path']:
        raise ValueError('exact lifetime policy path')
    if args.action == 'check':
        print(json.dumps(dict(checked=True, default_off=policy['execute_allowed'] is not True,
                              services_changed=False, job_permissions_unchanged=True)))
        return
    if args.action == 'run-service':
        return run_service(policy, args.role)
    directory = Path(policy['record_directory'])
    if directory != directory.resolve() or not directory.is_absolute():
        raise ValueError('owned instance record directory')
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        raise ValueError('private instance records')
    key = key_for(policy['authority_seed_path'], args.authority)
    def record(value):
        body = dict(version=INSTANCE, policy_sha256=digest(policy), role=args.role, **value)
        write_exclusive(directory/(args.role+'-'+str(time.time_ns())+'.json'), sign(key, body))
    guard = lambda: validate(json.loads(file_bytes(args.policy)), args.authority, require_execution=True)
    supervisor = Supervisor(policy, args.role, record=record, guard=guard)
    expected = list(policy['services'][args.role]['expected_argv'])
    expected[4] = 'supervise'
    live = current(dict(unit=policy['services'][args.role]['supervisor_unit']))
    if live['process'] != process(os.getpid()) or live['process']['argv'] != expected:
        raise ValueError('only scoped systemd supervisor invocation')
    lock = os.open(directory/(args.role+'.lock'), os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Restart circuit survives supervisor crashes; records remain immutable.
        now, monotonic = time.time(), time.monotonic()
        for path in directory.glob(args.role+'-*.json'):
            row = authenticate(json.loads(file_bytes(path)), args.authority)
            if row['version'] != INSTANCE or row['policy_sha256'] != digest(policy):
                raise ValueError('owned same-policy instance history')
            if row['event'] == 'restart_attempt':
                supervisor.attempts += 1
            if row['event'] in ('restart_attempt', 'unready_instance_retired') and now-row['at'] < supervisor.service['circuit_seconds']:
                supervisor.failures.append(monotonic-max(0, now-row['at']))
        while True:
            # Revocation and all bindings are rechecked before each possible start.
            guard()
            supervisor.step()
            time.sleep(5)
    finally:
        os.close(lock)


if __name__ == '__main__':
    main()
