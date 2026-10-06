"""CPU-only genuine archived pair comparison; ephemeral test authority only."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import zipfile
from nacl.signing import SigningKey
from subnet.job_scoped_native_math_grader import (
    VERSION, _digest, _source_members, canonical, prepare_job_scoped_grader)
from subnet.native_math_grader import ASSET, dependency_binding, isolated_argv


def qualify(*, archive, snapshot, interpreter, environment_sha256, output, outer_timeout=30):
    root = Path(__file__).resolve().parents[1]
    snapshot = Path(snapshot).resolve()
    rows = json.loads(snapshot.read_bytes())
    cases = []
    with tarfile.open(archive) as tar:
        for index in (4876, 6350):
            name = f'original-supervision-v1/control/fresh-pair-{index}.zip'
            member = tar.getmember(name)
            if not member.isfile() or member.size > 10 * 1024**2:
                raise ValueError('bounded genuine pair member')
            data = tar.extractfile(member).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if z.getinfo('manifest.json').file_size > 1024**2:
                    raise ValueError('bounded manifest')
                manifest = json.loads(z.read('manifest.json'))
            for item in manifest:
                for rollout in item['batch']['rollouts']:
                    if rollout['index'] != index or len(rollout['turns']) != 1:
                        raise ValueError('genuine task/turn binding')
                    cases.append(dict(index=index, task_hash=rollout['task_hash'],
                        artifact_sha256=hashlib.sha256(data).hexdigest(),
                        text_sha256=hashlib.sha256(rollout['turns'][0]['text'].encode()).hexdigest(),
                        reply=rollout['turns'][0]['text'], claimed_classification=rollout['classification']))
    if len(cases) != 4:
        raise ValueError('exact genuine four-case cohort')
    now = time.time()
    p = dict(version=VERSION, execute_allowed=True, job_id='genuine-four-case-CPU-qualification-only',
        created_at=now-1, expires_at=now+300, source_root=str(root),
        source_files={name: _digest(root/name) for name in _source_members(root)},
        asset_files={str(snapshot): _digest(snapshot)}, grader_path=str(ASSET.resolve()),
        grader_sha256=_digest(ASSET), interpreter=str(Path(interpreter).absolute()),
        environment_sha256=environment_sha256, snapshot_path=str(snapshot),
        snapshot_sha256=_digest(snapshot), tasks={str(c['index']): dict(task_hash=c['task_hash'],
            row_sha256=hashlib.sha256(canonical(rows[c['index']])).hexdigest()) for c in cases},
        max_requests=4, outer_timeout_seconds=outer_timeout, memory_bytes=768*1024**2,
        native_runtime_binding=next(iter(dependency_binding().values())))
    key = SigningKey.generate()
    scope = dict(payload=p, signer=key.verify_key.encode().hex(),
                 signature=base64.b64encode(key.sign(canonical(p)).signature).decode())
    comparison = []
    for c in cases:
        began = time.monotonic()
        r = subprocess.run(isolated_argv(Path(interpreter), ASSET,
            ['--json-arguments', json.dumps([rows[c['index']]['data']['answer'], c['reply']])]),
            capture_output=True, text=True, timeout=outer_timeout)
        comparison.append(dict(index=c['index'], task_hash=c['task_hash'],
            artifact_sha256=c['artifact_sha256'], text_sha256=c['text_sha256'],
            claimed_classification=c['claimed_classification'], legacy=dict(returncode=r.returncode,
                stdout=r.stdout, stderr=r.stderr, elapsed_seconds=time.monotonic()-began)))
    began = time.monotonic()
    with prepare_job_scoped_grader(scope, authority=scope['signer'], job_id=p['job_id'],
            environment_sha256=environment_sha256, before_model_construction=True) as grader:
        preparation_seconds = time.monotonic() - began
        parent_pid = grader._process.pid
        for c, record in zip(cases, comparison):
            call = time.monotonic()
            result = grader.grade(index=c['index'], task_hash=c['task_hash'], reply=c['reply'])
            result['whole_request_seconds'] = time.monotonic()-call
            record['job_scoped'] = result
            record['identical_outcome'] = all(result[k] == record['legacy'][k]
                                              for k in ('returncode', 'stdout', 'stderr'))
        total_seconds = time.monotonic()-began
    if not all(r['identical_outcome'] for r in comparison):
        raise ValueError('genuine outcome mismatch')
    result = dict(version=VERSION, qualification_only=True, production_activated=False,
        GPU_used=False, ROOT_authority_used=False, ephemeral_test_authority=scope['signer'],
        source_files=p['source_files'], snapshot_sha256=p['snapshot_sha256'],
        environment_sha256=environment_sha256, native_grader_sha256=p['grader_sha256'],
        original_inner_timeout_unchanged=True, CPU_control_outer_timeout_seconds=outer_timeout,
        parent_pid=parent_pid, dedicated_parent_reaped=True,
        preparation_seconds=preparation_seconds, four_case_total_seconds=total_seconds,
        comparison=comparison)
    path = Path(output)
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    path.chmod(0o600)
    return result


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    for name in ('archive', 'snapshot', 'interpreter', 'environment-sha256', 'output'):
        a.add_argument('--'+name, required=True)
    a.add_argument('--outer-timeout', type=float, default=30)
    args = a.parse_args()
    result = qualify(archive=args.archive, snapshot=args.snapshot, interpreter=args.interpreter,
                     environment_sha256=args.environment_sha256, output=args.output,
                     outer_timeout=args.outer_timeout)
    print(json.dumps(dict(output=args.output, preparation_seconds=result['preparation_seconds'],
        four_case_total_seconds=result['four_case_total_seconds'],
        outcomes=[dict(index=x['index'], legacy=x['legacy']['elapsed_seconds'],
                      scoped=x['job_scoped']['whole_request_seconds'],
                      native_child=x['job_scoped']['elapsed_seconds'],
                      score=x['job_scoped']['score'], identical=x['identical_outcome'])
                  for x in result['comparison']])))
