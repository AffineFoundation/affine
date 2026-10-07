"""Default-off ROOT-approved training-only blacklist snapshot; no audit barrier.

A status restricts training supply, never asserts the candidate itself fraudulent.
The original audit/reward population remains independent of this filter.
"""
import math
from .training_receipts import authenticate, digest, sha

FIELD = 'learner_blacklist_selection_policy'
VERSION = 'confirmed-blacklist-training-selection-v1'


def opening_freshness_time(manifest, at):
    """Use the authenticated epoch opening, without expiring saved snapshots.

    Callers authenticate the complete original manifest. Minimal pre-opening
    contexts have neither start nor deadline and use the actual check time.
    Returned admission fields stay compatible with historical signed jobs.
    """
    if type(at) not in (int, float) or not math.isfinite(at):
        raise ValueError('finite original assessment admission time')
    if 'start' not in manifest and 'deadline' not in manifest:
        return at
    start, deadline = manifest.get('start'), manifest.get('deadline')
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, deadline)) or
            not 0 < deadline - start <= 7200 or at < start):
        raise ValueError('bounded original signed epoch opening/deadline')
    return start


def admit(document, manifest, authority, *, at, round_number=None):
    p = authenticate(document, authority)
    fields = {'version', 'checkpoint', 'source_sha256', 'target_round',
              'maximum_age_seconds', 'assessment_document', 'writer_policy_sha256', 'audit_policy'}
    if (set(p) != fields or p['version'] != VERSION or
            p['checkpoint'] != manifest['checkpoint']['id'] or
            p['source_sha256'] != manifest['source_bundle']['sha256'] or
            type(p['target_round']) is not int or p['target_round'] < 0 or
            (round_number is not None and p['target_round'] != round_number) or
            type(p['maximum_age_seconds']) is not int or not 1 <= p['maximum_age_seconds'] <= 7200):
        raise ValueError('ROOT training blacklist policy context/bounds')
    digest(p['writer_policy_sha256'])
    audit = validate_audit_policy(p['audit_policy'])
    a = authenticate(p['assessment_document'], authority)
    freshness_at = opening_freshness_time(manifest, at)
    if (a.get('version') != 'hourly-current-miner-assessment-v1' or
            a.get('assessment_stale') is not False or
            a.get('writer_policy_sha256') != p['writer_policy_sha256'] or
            type(a.get('cutoff')) is not int or a['cutoff'] % 3600 or
            a.get('evidence_cutoff') != a['cutoff'] or
            type(at) not in (int, float) or not math.isfinite(at) or
            not 0 <= freshness_at - a['cutoff'] <= p['maximum_age_seconds'] or
            type(a.get('miner_estimates')) is not dict):
        raise ValueError('fresh original authenticated blacklist assessment')
    excluded = []
    for miner, detail in a['miner_estimates'].items():
        digest(miner)
        if type(detail) is not dict or type(detail.get('blacklisted')) is not bool:
            raise ValueError('explicit authenticated blacklist status')
        if not detail['blacklisted']:
            continue
        count, latest, observed = (detail.get(k) for k in
                                  ('confirmed_invalid_recent', 'latest_bad_round', 'current_estimate_round'))
        if (any(type(v) is not int for v in (count, latest, observed)) or
                not audit['blacklist_after'] or count < audit['blacklist_after'] or
                not 0 <= latest <= observed <= p['target_round'] or
                not 0 <= observed - latest < audit['blacklist_epochs'] or
                detail.get('unresolved_is_fraud') is not False or
                detail.get('infrastructure_counted_in_coverage') is not False):
            raise ValueError('confirmed-invalid blacklist evidence bounds')
        if p['target_round'] - latest < audit['blacklist_epochs']:
            excluded.append(miner)
    return dict(version=VERSION, policy_sha256=sha(document),
                assessment_sha256=sha(p['assessment_document']),
                writer_policy_sha256=p['writer_policy_sha256'], audit_policy_sha256=sha(audit),
                assessment_cutoff=a['cutoff'], target_round=p['target_round'],
                excluded_miners=sorted(excluded),
                audit_barrier=False, candidate_fraud_verdict=False)


def partition(eligible, manifest, authority, *, at, round_number=None):
    """Keep ordering and original objects; only explicit active blacklist excludes."""
    if FIELD not in manifest:
        return eligible, None
    document = manifest[FIELD]
    binding = admit(document, manifest, authority, at=at, round_number=round_number)
    blocked = set(binding['excluded_miners']); kept = []; excluded = []
    for obj in eligible:
        value = authenticate(obj['learner_admission'], authority)
        if value['epoch'] != manifest['epoch'] or value['checkpoint'] != manifest['checkpoint']['id']:
            raise ValueError('training blacklist candidate original context')
        if value['miner_identity'] in blocked:
            excluded.append(dict(document_sha256=obj['sha256'],
                                 miner_identity=value['miner_identity'],
                                 reason='active-authenticated-confirmed-blacklist'))
        else:
            kept.append(obj)
    return kept, dict(binding, exclusions=excluded,
                      structural_eligible_count=len(eligible), training_eligible_count=len(kept))

AUTHORIZATION_FIELD = 'learner_blacklist_selection_authorization'
AUTHORIZATION_VERSION = 'automatic-confirmed-blacklist-training-selection-v1'


def prepare_opening(controller, config, status, contract):
    """Snapshot existing signed writer evidence once, under ROOT-approved mechanism.

    Called only while constructing a new opening; saved openings never refresh it.
    The authority remains local to the already authorized coordinator.
    """
    if AUTHORIZATION_FIELD not in config:
        if FIELD in contract:
            import time
            admit(contract[FIELD],dict(checkpoint=status['checkpoint'],source_bundle=config['source_bundle']),controller.authority.id,at=time.time(),round_number=status['round'])
        return contract
    document = config[AUTHORIZATION_FIELD]
    if FIELD in config:
        raise ValueError('one automatic or fixed training blacklist policy')
    import os, stat, time
    from pathlib import Path
    import json
    approval = authenticate(document, controller.authority.id)
    fields = {'version', 'source_sha256', 'writer_policy_sha256', 'audit_policy',
              'maximum_age_seconds', 'assessment_path'}
    if set(approval) != fields or approval['version'] != AUTHORIZATION_VERSION:
        raise ValueError('ROOT automatic training blacklist authorization')
    if approval['source_sha256'] != config['source_bundle']['sha256']:
        raise ValueError('automatic training blacklist source authorization')
    if config.get('training_input_policy') != 'committed-unaudited-training-v1':
        raise ValueError('automatic training blacklist requires committed learner')
    path = Path(approval['assessment_path'])
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError('ordinary owned assessment path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size > 32 * 1024**2:
            raise ValueError('bounded owned assessment file')
        data = stream.read(32 * 1024**2 + 1)
    if len(data) > 32 * 1024**2:
        raise ValueError('bounded assessment snapshot')
    assessment = json.loads(data)
    p = dict(version=VERSION, checkpoint=status['checkpoint']['id'],
             source_sha256=approval['source_sha256'], target_round=status['round'],
             maximum_age_seconds=approval['maximum_age_seconds'], assessment_document=assessment,
             writer_policy_sha256=approval['writer_policy_sha256'], audit_policy=approval['audit_policy'])
    envelope = controller.signed(p)
    admit(envelope, dict(checkpoint=status['checkpoint'], source_bundle=config['source_bundle']),
          controller.authority.id, at=time.time(), round_number=status['round'])
    return dict(contract, **{FIELD: envelope, 'learner_blacklist_selection_round': status['round']})


def validate_audit_policy(value):
    """CPU-only scalar schema, including deployed writer v3; no audit execution.

    Frozen scientific continuous_audit_policy stays unchanged. This separate
    parser must be explicitly pinned in the ROOT selection operator admission.
    """
    fields={'version','recent_epochs','decay','prior_alpha','prior_beta','invalid_multiplier','zero_epoch_after','blacklist_after','blacklist_epochs'}
    if type(value)is not dict or set(value)!=fields or value['version']not in ('continuous-probabilistic-audit-v1','continuous-probabilistic-audit-v2','continuous-probabilistic-audit-v3'):raise ValueError('exact continuous audit policy')
    result=dict(value)
    for k,low,high in [('recent_epochs',1,128),('zero_epoch_after',0,10000),('blacklist_after',0,10000),('blacklist_epochs',0,128)]:
        if type(value[k])is not int or not low<=value[k]<=high:raise ValueError('bounded audit integer')
    for k,low,high in [('decay',.01,1),('prior_alpha',.01,100),('prior_beta',.01,100),('invalid_multiplier',0,1)]:
        if type(value[k])not in(int,float)or not math.isfinite(value[k])or not low<=value[k]<=high:raise ValueError('bounded audit number')
        result[k]=float(value[k])
    return result
