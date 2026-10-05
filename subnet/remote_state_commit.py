"""Explicit prospective remote readback admission; authority writes remain local."""
import hashlib
import json
import math

from . import remote_optimizer_readback as reader
from . import persistent_training_protocol as state


def readback_objects(descriptor):
    """Transport hashes three fields; full descriptor retains tensor metadata.

    Callers must first scientifically validate the complete original descriptor.
    Its digest binds every tensor slot, range, parameter and optimizer counter.
    """
    return [{k:shard[k] for k in ('name','size','sha256')}
            for shard in descriptor['shards']]


def independently_commit_remote(controller, report, original_job_envelope,
        request_bytes, receipt, launch_envelope, terminal, *, qualified_reader,
        reader_host, trainer_host, storage_binding, original_child, now):
    """Expectations come only from original root records, never returned receipts.

    ``terminal`` and ``original_child`` must be independently observed by root
    over its approved host route. This trusts an operator-controlled reader and
    its key; it does not prove correctness against a malicious reader host.
    This explicit entrypoint never silently falls back or accepts CPU controls.
    """
    authority = controller.authority.id
    job = reader.verify(original_job_envelope, authority)
    manifest = reader.verify(job['manifest'], authority)
    descriptor = state.validate_report(report, job, manifest)
    namespace = job['persistent_training']['output_namespace']
    if reader.canonical(state.read_json(controller.bucket, namespace+'/staged-state.json')) != reader.canonical(descriptor):
        raise ValueError('independent original staged descriptor')
    for host in (reader_host, trainer_host):
        if set(host) != {'provider_UUID','ssh_host_key_sha256','evidence_sha256'}:
            raise ValueError('root physical host record')
        if not isinstance(host['provider_UUID'], str) or not host['provider_UUID']:
            raise ValueError('root provider machine identity')
        for key in ('ssh_host_key_sha256','evidence_sha256'):
            state.checkpoint_id(host[key])
    if reader_host['provider_UUID'] == trainer_host['provider_UUID']:
        raise ValueError('different actual physical reader/trainer machines')
    if set(storage_binding) != {'storage_origin','storage_bucket','storage_addressing'}:
        raise ValueError('root storage binding')
    binding = dict(purpose='production-training-state',
        provenance=dict(signed_job_envelope_sha256=reader.sha(original_job_envelope),
            original_report_sha256=reader.sha(report),
            signed_manifest_envelope_sha256=reader.sha(job['manifest'])),
        job_id=job['job_id'], job_sha256=reader.sha(job),
        source_sha256=manifest['trainer_state_binding']['source_sha256'],
        namespace=namespace, descriptor_sha256=reader.sha(descriptor),
        reader_host_record_sha256=reader.sha(reader_host),
        trainer_host_record_sha256=reader.sha(trainer_host), **storage_binding)
    if not isinstance(request_bytes, bytes) or len(request_bytes)>4_000_000:
        raise ValueError('bounded original signed request file')
    request = json.loads(request_bytes)
    request_file_sha256 = hashlib.sha256(request_bytes).hexdigest()
    launch = reader.verify(launch_envelope, authority)
    if (launch.get('version') != 'independent-state-readback-launch-v1' or
            launch.get('request_sha256') != request_file_sha256 or
            launch.get('module_sha256') != hashlib.sha256(
                __import__('pathlib').Path(reader.__file__).read_bytes()).hexdigest()):
        raise ValueError('original root-approved reader launch bytes')
    payload = reader.validate_receipt(receipt, request, authority,
        approved_binding=binding, approved_objects=readback_objects(descriptor),
        qualified_reader=qualified_reader, now=now)
    if (terminal.get('actual_child_wait_completed') is not True or
            type(terminal.get('exit_code')) is not int or terminal['exit_code'] != 0 or
            terminal.get('timed_out') is not False or
            terminal.get('request_sha256') != request_file_sha256 or
            terminal.get('authority_publication_written') is not False or
            terminal.get('gpu_imports_requested') is not False or
            type(original_child.get('pid')) is not int or original_child['pid'] <= 0 or
            not isinstance(original_child.get('ticks'), str) or
            not original_child['ticks'].isdigit() or
            terminal.get('pid') != original_child['pid'] or
            terminal.get('ticks') != original_child['ticks']):
        raise ValueError('actual original successful bounded reader process')
    start=terminal.get('started_at');finish=terminal.get('finished_at')
    if (not all(type(v) in (int,float) and math.isfinite(v) for v in (start,finish)) or
            not launch['created_at'] <= start <= payload['started_at'] <=
                payload['completed_at'] <= finish <= now < launch['expires_at'] or
            finish-start > launch['max_wall_seconds']):
        raise ValueError('original reader terminal time budget')
    # Persist authenticated complete readback evidence before authority-last.
    evidence=dict(version='independent-state-readback-evidence-v1',
        request=request,receipt=receipt,launch=launch_envelope,
        terminal=terminal,original_child=original_child,
        request_file_sha256=request_file_sha256)
    evidence_key=namespace+'/independent-readbacks/'+reader.sha(evidence)+'.json'
    controller.bucket.json(evidence_key,controller.signed(evidence))
    observed=reader.verify(state.read_json(controller.bucket,evidence_key),authority)
    if reader.canonical(observed)!=reader.canonical(evidence):
        raise ValueError('durable original independent reader evidence')
    return state._publish_verified_descriptor(controller,descriptor,job,namespace)
