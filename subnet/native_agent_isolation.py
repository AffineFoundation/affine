"""Controlled original GeneralAgent fixtures in distinct immutable Docker images.

This is a prospective native adapter boundary, not a production registration.
The actor image contains public tools/database/instructions; its private grader
image contains original checks/gold. Images are operator-approved local image
IDs, never a rollout-selected image or uploaded Python implementation.
"""
import hashlib
import json
import re
import selectors
import subprocess

REVISION = 'controlled-general-agent-fixtures-v1'
LIMIT = 1_000_000

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def validate_descriptor(value):
    required = {'revision', 'task_name', 'actor_image', 'grader_image',
                'public_files', 'private_files', 'original_grader_sha256',
                'dependency_scope'}
    if set(value) != required or value['revision'] != REVISION:
        raise ValueError('unapproved native fixture descriptor')
    if value['dependency_scope'] != 'immutable-controlled-images-not-full-upstream-closure':
        raise ValueError('native fixture dependency scope')
    for field in ('actor_image', 'grader_image'):
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', value[field]):
            raise ValueError('immutable native image ID required')
    if value['actor_image'] == value['grader_image']:
        raise ValueError('private grader must be a distinct image')
    public = value['public_files']; private = value['private_files']
    if not {'tools.py', 'base.py', 'db.json', 'instruction.md', 'worker.py'} <= set(public):
        raise ValueError('incomplete public native fixture')
    if {'gold.json', 'original-taskset.py', 'task.toml'} & set(public):
        raise ValueError('private grader leaked into miner fixture')
    if not {'gold.json', 'original-taskset.py', 'worker.py'} <= set(private):
        raise ValueError('incomplete private native grader')
    for inventory in (public, private):
        for name, digest in inventory.items():
            if '/' in name or '\\' in name or name in ('', '.', '..') or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('native fixture file identity')
    if private['original-taskset.py'] != value['original_grader_sha256']:
        raise ValueError('native original grader identity')
    return value

def docker_command(descriptor, role):
    validate_descriptor(descriptor)
    if role not in ('actor', 'grader'):
        raise ValueError('unknown native container role')
    image = descriptor['actor_image' if role == 'actor' else 'grader_image']
    return ['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--pids-limit', '64', '--cpus', '1', '--memory', '512m',
            '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
            '--label', 'affine.native-agent-controlled='+REVISION,
            image, 'python', '-B', '/fixture/worker.py', role]

def checked_grade(value, descriptor):
    if value.get('original_source_sha256') != descriptor['original_grader_sha256']:
        raise ValueError('native grader response source mismatch')
    if type(value.get('reward')) not in (int, float) or value['reward'] not in (0, 1):
        raise ValueError('native binary original reward')
    if set(value.get('metrics', {})) != {'db_hash', 'verify'} or any(
            type(v) not in (int, float) or v not in (0, 1) for v in value['metrics'].values()):
        raise ValueError('native original grading metrics')
    if value['reward'] != max(value['metrics'].values()):
        raise ValueError('native original solved reward mismatch')
    return value

class NativeAgentSession:
    """One original mutable TaskDB and tool trace; private grading is separate.

    Models receive only instruction, schemas and actual tool observations.
    State snapshots are operator/replay artifacts, never candidate-generation
    input. This class is not yet dispatched by the production env registry.
    """
    def __init__(self, descriptor, instruction, timeout=30):
        self.descriptor = validate_descriptor(descriptor)
        if not isinstance(instruction, str) or len(instruction.encode()) > LIMIT:
            raise ValueError('native instruction budget')
        if hashlib.sha256(instruction.encode()).hexdigest() != descriptor['public_files']['instruction.md']:
            raise ValueError('native public instruction source mismatch')
        self.instruction = instruction; self.timeout = timeout
        self.process = None; self.events = []

    def start(self):
        if self.process is not None:
            raise ValueError('native session already started')
        self.process = subprocess.Popen(docker_command(self.descriptor, 'actor'),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools = self.request({'operation': 'list_tools'})['result']
        if not isinstance(tools, list) or len(tools) > 128:
            raise ValueError('native tool schema budget')
        return dict(messages=[dict(role='user', content=self.instruction)], tools=tools,
                    task_hash=hashlib.sha256(canonical(self.descriptor)).hexdigest())

    def request(self, value):
        data = canonical(value)
        if len(data) > LIMIT or self.process is None:
            raise ValueError('native request budget/session')
        self.process.stdin.write(data+b'\n'); self.process.stdin.flush()
        selector = selectors.DefaultSelector()
        try:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError('native isolated tool deadline')
            line = self.process.stdout.readline(LIMIT+1)
            if not line or len(line) > LIMIT or not line.endswith(b'\n'):
                raise ValueError('native response framing')
            result = json.loads(line)
            if set(result) != {'result', 'state_hash'} or not re.fullmatch(r'[0-9a-f]{12}', result['state_hash']):
                raise ValueError('native response state identity')
            return result
        finally:
            selector.close()

    def call(self, name, arguments):
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ValueError('native tool action')
        response = self.request(dict(operation='call', name=name, arguments=arguments))
        self.events.append(dict(name=name, arguments=arguments, response=response))
        return response['result']

    def grade(self):
        state = self.request({'operation': 'state'})['result']
        data = canonical(state)
        if len(data) > LIMIT:
            raise ValueError('native grading state budget')
        result = subprocess.run(docker_command(self.descriptor, 'grader'),
            input=data, capture_output=True, timeout=self.timeout, check=True)
        if len(result.stdout) > LIMIT:
            raise ValueError('native grading response budget')
        grade = checked_grade(json.loads(result.stdout), self.descriptor)
        return dict(grade=grade, state_sha256=hashlib.sha256(data).hexdigest(),
                    events_sha256=hashlib.sha256(canonical(self.events)).hexdigest())

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:self.process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
            self.process.stdout.close(); self.process.stderr.close()
            self.process = None
