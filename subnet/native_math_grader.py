"""Prospective immutable native MATH grader profile, without importing grader code."""
import ast
import hashlib
import json
from pathlib import Path

VERSION = 'native-math-grader-pinned-indeterminate-v1'
ASSET = Path(__file__).parent / 'vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1/verify.py'


def runtime_lock():
    tree = ast.parse(ASSET.read_text())
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'RUNTIME_LOCK' for target in node.targets)]
    if len(values) != 1:
        raise ValueError('native MATH grader lock missing or ambiguous')
    value = values[0]
    if set(value) != {'python_version', 'profiles', 'distributions'}:
        raise ValueError('native MATH grader lock schema')
    if value['python_version'] != [3, 12, 3]:
        raise ValueError('native MATH grader interpreter profile')
    if not isinstance(value['profiles'], list) or len(value['profiles']) != 2:
        raise ValueError('native MATH grader exact approved profile count')
    seen = set()
    for profile in value['profiles']:
        if set(profile) != {'python_executable_sha256', 'stdlib'}:
            raise ValueError('native MATH grader interpreter profile schema')
        if len(bytes.fromhex(profile['python_executable_sha256'])) != 32:
            raise ValueError('native MATH grader interpreter digest')
        standard = profile['stdlib']
        if set(standard) != {'files_count', 'files_sha256'} or type(standard['files_count']) is not int or standard['files_count'] < 1:
            raise ValueError('native MATH grader standard library inventory')
        if len(bytes.fromhex(standard['files_sha256'])) != 32:
            raise ValueError('native MATH grader standard library digest')
        identity = json.dumps(profile, sort_keys=True)
        if identity in seen:raise ValueError('native MATH grader duplicate profile')
        seen.add(identity)
    required = {'math-verify':'0.9.0', 'latex2sympy2-extended':'1.11.0', 'sympy':'1.14.0',
                'antlr4-python3-runtime':'4.13.2', 'mpmath':'1.3.0'}
    if set(value['distributions']) != set(required):
        raise ValueError('native MATH grader dependency closure')
    for name, expected in required.items():
        entry = value['distributions'][name]
        if set(entry) != {'version', 'python_files_count', 'python_files_sha256'} or entry['version'] != expected:
            raise ValueError('native MATH grader dependency profile')
        if type(entry['python_files_count']) is not int or entry['python_files_count'] < 1:
            raise ValueError('native MATH grader file inventory')
        if len(bytes.fromhex(entry['python_files_sha256'])) != 32:
            raise ValueError('native MATH grader inventory digest')
    return value


def dependency_binding():
    lock = runtime_lock()
    return {VERSION: hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


BOOTSTRAP = ("import sys,runpy,json,hashlib,importlib.metadata,sysconfig;"
             "site=sys.argv.pop(1);script=sys.argv.pop(1);"
             "sys.path.insert(0,site);sys.argv[0]=script;"
             "runpy.run_path(script,run_name='__main__')")


def isolated_argv(interpreter, script, args):
    interpreter = Path(interpreter)
    if not interpreter.is_absolute() or interpreter.parent.name != 'bin':
        raise ValueError('native MATH grader prepared interpreter path')
    site = interpreter.parent.parent / 'lib/python3.12/site-packages'
    return [str(interpreter), '-I', '-S', '-c', BOOTSTRAP, str(site), str(script), *args]
