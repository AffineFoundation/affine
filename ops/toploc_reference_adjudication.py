"""Offline research replay of an original audit; never writes production evidence.

Invoke in a fresh process on an independently qualified GPU with the original
frozen source, checkpoint and downloaded committed artifact. No credentials or
network are used. The original verifier is wrapped solely to record its native
TOPLOC result metrics; its results and thresholds are never changed.
"""
import argparse
import base64
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def authenticated(envelope, expected):
    from nacl.signing import VerifyKey
    if envelope['signer'] != expected:
        raise ValueError('expected original signer')
    VerifyKey(bytes.fromhex(expected)).verify(canonical(envelope['payload']),
        base64.b64decode(envelope['signature'], validate=True))
    return envelope['payload']


def inputs(job_envelope, request_envelope, authority, worker, child, artifact):
    job = authenticated(job_envelope, authority)
    manifest = authenticated(job['manifest'], authority)
    request = authenticated(request_envelope, worker)
    report = request['report']
    if request['action'] != 'report' or request['job_id'] != job['job_id'] or report['job_id'] != job['job_id'] or report['job_sha256'] != digest(job):
        raise ValueError('original report and job binding')
    if report.get('success') is not True or report.get('role') != 'verify' or report['epoch'] != manifest['epoch'] or report['checkpoint'] != manifest['checkpoint']['id'] or report['source_files'] != job['source_files'] or report['runtime_versions'] != job['runtime_versions'] or report['backend_profile'] != manifest['backend_profile'] or report['numerical_policy'] != manifest['numerical_policy']:
        raise ValueError('original execution binding')
    if type(child) is not int or not 0 <= child < len(job['submissions']):
        raise ValueError('original child index')
    obj = job['submissions'][child]
    audit = report['audits'][child]
    if hashlib.sha256(artifact).hexdigest() != obj['sha256'] or audit['submission_sha256'] != obj['sha256']:
        raise ValueError('immutable original artifact digest')
    if audit['epoch'] != manifest['epoch'] or len(audit['outcomes']) != 1:
        raise ValueError('original single child outcome')
    return job, manifest, obj, audit


def committed_batch(manifest, obj, audit, batch):
    ref = obj['commitment_ref']
    if obj.get('commitment_miner') != ref['miner'] or type(ref['slot']) is not int or ref['slot'] < 0:
        raise ValueError('original committed miner and slot')
    if batch['epoch'] != manifest['epoch'] or batch['checkpoint'] != manifest['checkpoint']['id'] or batch['env_id'] != ref['env_id'] or batch['index'] != ref['index'] or batch.get('sample_index') != ref['index'] or digest(batch) != ref['batch_sha256']:
        raise ValueError('original full committed task tuple')
    if audit.get('selected_batches') != [0] or audit['outcomes'][0].get('batch') != 0 or audit['outcomes'][0].get('fully_audited') is not True:
        raise ValueError('original selected fully audited child')
    return True


def instrument(runtime, diagnostics):
    """Delegate unchanged; record every segment without copying hidden states."""
    original = runtime.verify_proofs
    def measured(acts, proofs, **kwargs):
        results = original(acts, proofs, **kwargs)
        rows = [dict(segment=i, proof_sha256=hashlib.sha256(proofs[i].encode()).hexdigest() if i < len(proofs) else None,
                     exp_mismatches=int(r.exp_mismatches), mant_err_mean=float(r.mant_err_mean),
                     mant_err_median=float(r.mant_err_median)) for i, r in enumerate(results)]
        diagnostics.append(dict(expected_segments=len(proofs), returned_segments=len(results), segments=rows))
        return results
    runtime.verify_proofs = measured


def reference_check(runtime, rollout, probabilities, InvalidSample, NumericalAmbiguity):
    try:
        valid = runtime.verify(rollout, probabilities) is True
        return dict(reference_valid=True if valid else None, classification='reference_valid' if valid else 'unresolved_runtime_result')
    except NumericalAmbiguity as error:
        return dict(reference_valid=None, classification='numerical_ambiguous', error_type=type(error).__name__, reason=str(error))
    except InvalidSample as error:
        return dict(reference_valid=False, classification='reference_rejected', error_type=type(error).__name__, reason=str(error))
    except Exception as error:
        return dict(reference_valid=None, classification='research_infrastructure_error', error_type=type(error).__name__, reason=str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'report-request', 'artifact', 'source-root', 'checkpoint', 'asset-root', 'authority', 'worker', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--child', type=int, required=True)
    args = parser.parse_args(argv)
    read = lambda path: json.loads(Path(path).read_bytes())
    artifact = Path(args.artifact).read_bytes()
    job, manifest, obj, audit = inputs(read(args.job), read(args.report_request), args.authority, args.worker, args.child, artifact)
    source = Path(args.source_root).resolve()
    # Job source pins come from the authenticated approved source admission.
    # ROOT must separately approve their whole-bundle SHA before invocation.
    for name, pin in job['source_files'].items():
        target = (source / name).resolve()
        if not target.is_relative_to(source) or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != pin:
            raise ValueError('exact original scientific source file: ' + name)
    if set(job['runtime_versions']) != {'torch', 'transformers', 'toploc'}:
        raise ValueError('exact scientific package pins')
    for package, expected in job['runtime_versions'].items():
        if version(package) != expected:
            raise ValueError('original scientific package version: ' + package)
    if any(name == 'subnet' or name.startswith('subnet.') for name in sys.modules):
        raise ValueError('fresh process required')
    output = Path(args.output)
    if output.exists():
        raise ValueError('research output must be new')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['AFFINE_MATH_CORPUS_ASSET_ROOT'] = str(Path(args.asset_root).resolve())
    sys.path.insert(0, str(source))
    from subnet import backend_jobs
    backend_jobs.install_source_loader(source, tuple(job['source_files']))
    initial_configuration = backend_jobs.initial_configuration
    from subnet.backend_profiles import resolve
    resolve(manifest)
    from subnet.gpu_runtime import GPURuntime
    from subnet.forced_sampling import bind_runtime
    from subnet.probability_artifacts import bind_runtime as bind_artifacts
    from subnet.batches import unpack
    from subnet.artifact_budget import for_manifest
    from subnet.audit_policy import InvalidSample
    from subnet.protocol import entry, harness_for
    from subnet.fast_prefill_audit import NumericalAmbiguity
    definition, harness = initial_configuration(manifest, job)
    runtime = GPURuntime(args.checkpoint, manifest['checkpoint']['files'], definition['spec'], harness,
                         runtime_revision=manifest['model_runtime_revision'])
    bind_runtime(runtime, manifest)
    bind_artifacts(runtime, manifest)
    rows = unpack(artifact, budget=for_manifest(manifest))
    if len(rows) != 1:
        raise ValueError('single committed task batch required')
    batch, arrays = rows[0]
    if len(batch['rollouts']) != manifest['K'] + manifest['L'] or len(arrays) != len(batch['rollouts']):
        raise ValueError('original full rollout population')
    committed_batch(manifest, obj, audit, batch)
    definition = entry(manifest, batch['env_id'])
    if type(batch['index']) is not int or batch['index'] not in definition['indices'] or batch.get('schema') != 2:
        raise ValueError('original approved task eligibility')
    runtime = runtime.for_environment(definition['spec'], harness_for(definition, batch['index']))
    if batch['environment_version'] != runtime.spec.version or any(r['index'] != batch['index'] or r['env_id'] != batch['env_id'] for r in batch['rollouts']):
        raise ValueError('original environment/rollout binding')
    diagnostics = []
    instrument(runtime, diagnostics)
    results = []
    for i, (rollout, probabilities) in enumerate(zip(batch['rollouts'], arrays)):
        before = len(diagnostics)
        result = dict(rollout=i, **reference_check(runtime, rollout, probabilities, InvalidSample, NumericalAmbiguity))
        result['toploc_calls'] = diagnostics[before:]
        results.append(result)
    document = dict(version='toploc-reference-research-v1', production_evidence=False,
        rewards_or_original_reports_modified=False, historical_execution_proven=False,
        original_job_sha256=digest(job), original_report_request_sha256=digest(read(args.report_request)),
        original_worker=args.worker, epoch=manifest['epoch'], checkpoint=manifest['checkpoint']['id'],
        source_bundle_sha256=manifest['source_bundle']['sha256'], artifact_sha256=obj['sha256'],
        original_outcome=audit['outcomes'][0], calibration=manifest['sampling_contract']['calibration'], results=results)
    with output.open('xb') as stream:
        os.chmod(output, 0o600)
        stream.write(canonical(document))
    print(json.dumps(dict(research_only=True, rollouts=len(results), reference_valid=sum(r['reference_valid'] is True for r in results))))


if __name__ == '__main__':
    main()
