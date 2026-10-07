"""Local reaper ownership for retained research pods; no rental/delete authority.

Call bind immediately after an authenticated rental receipt, and require verify
before every qualification/reference/study launch. Observers heartbeat the same
binding. Job completion records evidence but never marks the retained pod done.
"""
import math

VERSION = 'retained-research-pod-ownership-v1'
OWNER = 'manual:explicit-retained-research-operator'


def validate(binding):
    fields = {'version', 'name', 'pod_id', 'purpose', 'price_usd_h',
              'rental_intent_sha256', 'retention_authorized'}
    if type(binding) is not dict or set(binding) != fields or binding['version'] != VERSION:
        raise ValueError('exact research ownership binding')
    if binding['retention_authorized'] is not True:
        raise ValueError('explicit retained-pod authorization required')
    for field in ('name', 'pod_id', 'purpose'):
        if type(binding[field]) is not str or not binding[field].strip() or '\n' in binding[field]:
            raise ValueError('exact research identity')
    value = binding['price_usd_h']
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('actual positive hourly rental cost')
    digest = binding['rental_intent_sha256']
    if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('original rental intent binding')
    return binding


def verify(registry, binding):
    b = validate(binding)
    record = registry.load().get(b['name'])
    if not record or record.get('released_at') or record.get('release_requested_at'):
        raise ValueError('retained research pod is not owned by reaper registry')
    meta = record.get('meta', {})
    if (record.get('owner') != OWNER or record.get('expected_hours') != 0
            or meta.get('provider_pod_id') != b['pod_id']
            or meta.get('research_rental_intent_sha256') != b['rental_intent_sha256']
            or meta.get('explicit_retention') is not True):
        raise ValueError('exact retained research owner/identity required')
    return record


def bind(registry, binding):
    """Fail closed on another owner; never disable or relax the global reaper."""
    b = validate(binding)
    prior = registry.load().get(b['name'])
    if prior and not prior.get('released_at'):
        # Never overwrite another active ownership record, even if the caller
        # merely reused a name. A separately reviewed adoption is required.
        meta = prior.get('meta', {})
        if (meta.get('provider_pod_id') == 'pending-original-rental'
                and meta.get('research_rental_intent_sha256') == b['rental_intent_sha256']
                and prior.get('owner') == OWNER and prior.get('expected_hours') == 0
                and meta.get('explicit_retention') is True):
            registry.register(b['name'], purpose=b['purpose'], owner=OWNER,
                expected_hours=0, price_usd_h=b['price_usd_h'], source='explicit',
                meta={'provider_pod_id': b['pod_id'], 'explicit_retention': True,
                      'research_rental_intent_sha256': b['rental_intent_sha256'],
                      'research_ownership_version': VERSION})
            return verify(registry, b)
        verify(registry, b)
        registry.touch(b['name'])
        return verify(registry, b)
    registry.register(b['name'], purpose=b['purpose'], owner=OWNER,
        expected_hours=0, price_usd_h=b['price_usd_h'], source='explicit',
        meta={'provider_pod_id': b['pod_id'], 'explicit_retention': True,
              'research_rental_intent_sha256': b['rental_intent_sha256'],
              'research_ownership_version': VERSION})
    return verify(registry, b)


def heartbeat(registry, binding):
    verify(registry, binding)
    registry.touch(binding['name'])
    return verify(registry, binding)


def before_launch(registry, binding, launch):
    """No GPU launch if registration/readback fails; launch is called once."""
    bind(registry, binding)
    verify(registry, binding)
    return launch()


def completed(registry, binding, evidence_sha256):
    """Completion is not retirement; a retained pod is usable for the next job."""
    verify(registry, binding)
    if type(evidence_sha256) is not str or len(evidence_sha256) != 64 or any(c not in '0123456789abcdef' for c in evidence_sha256):
        raise ValueError('original completion evidence digest')
    # The registry remains retained. Callers persist the terminal digest in
    # their original job journal; this helper intentionally has no delete API.
    return heartbeat(registry, binding)


def rent_once(registry, intent_binding, rent):
    """Register the intended name BEFORE renting; never retry the rental.

    The supplied rental callback is the reviewed provider command, which must
    return its original authenticated receipt's pod_id. A callback timeout
    propagates with the pending owned name intact for read-only reconciliation.
    """
    pending = dict(intent_binding, pod_id='pending-original-rental')
    validate(pending)
    prior = registry.load().get(pending['name'])
    if prior and not prior.get('released_at'):
        raise ValueError('original rental name already reserved; observe, never reissue')
    bind(registry, pending)
    receipt = rent()
    if type(receipt) is not dict or type(receipt.get('pod_id')) is not str or not receipt['pod_id']:
        raise ValueError('original authenticated provider pod identity required')
    bound = dict(pending, pod_id=receipt['pod_id'])
    bind(registry, bound)
    return receipt, bound
