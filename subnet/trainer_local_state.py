"""Trainer-only storage projection; inference/miner contracts stay unchanged."""
import copy
from .persistent_publication import LOCAL_POLICY, VERSION

FIELD='trainer_local_state_original_manifest'


def original_manifest(manifest,authority=None):
    if FIELD not in manifest:return manifest
    from .backend_jobs import signed
    document=manifest[FIELD]
    if authority is None:authority=document.get('signer')
    original=signed(document,authority)
    if FIELD in original:raise ValueError('local trainer projection cannot nest')
    expected=project(original,lambda _:document)
    if expected!=manifest:raise ValueError('local trainer projection changed computation or inputs')
    return original


def project(manifest,sign):
    if FIELD in manifest:
        original_manifest(manifest)
        return manifest
    result=copy.deepcopy(manifest)
    result[FIELD]=sign(manifest)
    result['optimizer_state_export_policy']=LOCAL_POLICY
    result['persistent_publication_policy']=dict(version=VERSION,state_readback='trainer-local',checkpoint_readback_workers=4)
    if result.get('optimizer_state_local_cache') is None:raise ValueError('trainer requires explicit retained optimizer state')
    result.pop('independent_state_readback_budget',None)
    return result


def selected(controller):
    jobs=controller.jobs
    if hasattr(jobs,'roles'):jobs=jobs.roles['train']
    mode=getattr(jobs,'config',{}).get('optimizer_state_lifecycle')
    if mode not in (None,LOCAL_POLICY):raise ValueError('explicit trainer optimizer lifecycle')
    return mode==LOCAL_POLICY
