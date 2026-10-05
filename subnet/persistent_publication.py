"""Prospective signed overlap of checkpoint transport and full state readback."""
import time
from concurrent.futures import ThreadPoolExecutor

VERSION='parallel-persistent-publication-v1'
EXPORT_POLICY='upload-only-independent-full-v1'

def export_policy(manifest):
    value=manifest.get('optimizer_state_export_policy')
    if value is None:return 'trainer-full'
    if value!=EXPORT_POLICY:raise ValueError('signed optimizer state export policy')
    policy=validate_policy(manifest.get('persistent_publication_policy'))
    if policy['state_readback']!='qualified-remote-full':raise ValueError('upload-only export requires qualified independent full readback')
    return value

def validate_policy(value):
    if (not isinstance(value,dict) or set(value)!=
            {'version','state_readback','checkpoint_readback_workers'} or
            value['version']!=VERSION or
            value['state_readback'] not in ('local-full','qualified-remote-full') or
            type(value['checkpoint_readback_workers']) is not int or
            not 1<=value['checkpoint_readback_workers']<=4):
        raise ValueError('signed bounded persistent publication policy')
    return dict(value)

def complete(controller,report,job,manifest,checkpoint_path):
    """No next-state journal or authority state until both complete paths pass.

    A failed path cannot silently switch readback hosts or retrain the model.
    The unchanged serial path remains default when no signed policy is present.
    """
    from .persistent_training_protocol import (independently_commit,
        independently_verify,_publish_verified_descriptor,validate_report)
    from .remote_state_commit import independently_commit_remote
    from .storage import canonical
    raw=manifest.get('persistent_publication_policy')
    mode=export_policy(manifest)
    new=dict(report['new_checkpoint']);new.pop('path')
    checkpoint_manifest=dict(manifest,checkpoint=new)
    if raw is None:
        checkpoint=controller.publish_remote_checkpoint(checkpoint_manifest,checkpoint_path)
        return checkpoint,independently_commit(controller,report,job,manifest),None
    policy=validate_policy(raw)
    validate_report(report,job,manifest)
    remote_reader=getattr(controller,'independent_state_reader',None)
    if policy['state_readback']=='qualified-remote-full' and remote_reader is None:
        raise ValueError('qualified independent reader required; no silent fallback')
    timings={};started=time.monotonic()
    def timed(name,operation):
        start=time.monotonic();result=operation()
        timings[name]=time.monotonic()-start
        return result
    if policy['state_readback']=='qualified-remote-full':
        envelope=__import__('json').loads((controller.state/'roles'/
            (job['job_id']+'-job.json')).read_text())
        from .backend_jobs import signed
        if canonical(signed(envelope,controller.authority.id))!=canonical(job):
            raise ValueError('original production state readback envelope')
        prepare=lambda:remote_reader.prepare_original_readback(controller,report,envelope)
    else:
        prepare=lambda:independently_verify(controller,report,job,manifest)
    if mode==EXPORT_POLICY:
        # Trainer receipts explicitly prove only local bytes/PUT completion.
        # Authenticate actual full independent GET/terminal before publishing
        # either authority checkpoint or optimizer-state descriptor.
        with ThreadPoolExecutor(max_workers=2)as pool:
            state_future=pool.submit(timed,'state_independent_readback_seconds',prepare)
            checkpoint_future=pool.submit(timed,'checkpoint_staging_seconds',lambda:
                controller.stage_remote_checkpoint(checkpoint_manifest,checkpoint_path))
            staged_checkpoint=checkpoint_future.result()
            state_result=state_future.result()
        state_result=dict(state_result,now=time.time())
        independently_commit_remote(controller,report,envelope,verify_only=True,**state_result)
        checkpoint=controller.commit_remote_checkpoint(checkpoint_manifest,staged_checkpoint)
        state_result=dict(state_result,now=time.time())
        pointer=independently_commit_remote(controller,report,envelope,**state_result)
        timings.update(overlapped_publication_seconds=time.monotonic()-started,
            publication_policy=policy,optimizer_state_export_policy=mode,
            authority_state_signed_after_checkpoint=True,
            authority_checkpoint_signed_after_independent_state_readback=True,
            trainer_verification_performed=False)
        return checkpoint,pointer,timings
    with ThreadPoolExecutor(max_workers=2)as pool:
        state_future=pool.submit(timed,'state_independent_readback_seconds',prepare)
        checkpoint_future=pool.submit(timed,'checkpoint_publication_seconds',lambda:
            controller.publish_remote_checkpoint(checkpoint_manifest,checkpoint_path))
        # Complete both before descriptor-last. Context drains any started work
        # on error; it never writes a partial optimizer authority publication.
        checkpoint=checkpoint_future.result()
        state_result=state_future.result()
    if policy['state_readback']=='qualified-remote-full':
        state_result=dict(state_result,now=time.time())
        pointer=independently_commit_remote(controller,report,envelope,**state_result)
    else:
        descriptor,namespace=state_result
        pointer=_publish_verified_descriptor(controller,descriptor,job,namespace)
    timings.update(overlapped_publication_seconds=time.monotonic()-started,
        publication_policy=policy,authority_state_signed_after_checkpoint=True,
        trainer_verification_performed=False)
    return checkpoint,pointer,timings
