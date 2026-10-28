"""v8.2.3 batch lifetime (Jacob pt765u): replace-until-seal and head-bound draws.

Replace until seal
  Every upload slot is a write-once object (signed If-None-Match PUT). A miner
  holds `upload_slots` slots per epoch (bounded storage), and the commitment
  object is the one replaceable pointer naming which slots form the batch.
  The host captures the newest commitment that landed before the signed
  deadline; the admission ledger records every superseding commitment
  (append-only, hash-bound) and the seal reads only the newest admitted one.
  Captured inputs are snapshotted content-addressed by commitment SHA-256.

Head-bound draws
  The token-draw context is bound to (head checkpoint, miner) instead of
  (epoch, head). The host keeps one secret `randomness` per head: generated
  fresh at the first opening after a head change, kept until the next head
  change. The draws are a deterministic hash of that seed and the public task,
  so a miner can never re-roll them, not even by waiting for a later epoch.
  This is strictly less grindable than per-epoch randomness, which gave a
  fresh roll every epoch. A head change gives a new checkpoint and a fresh
  seed, so the old draws (and the old model outputs) are invalid.
  No new per-epoch manifest field: the seed rides in the existing
  sampling_contract.randomness, and the signed immutable policy key turns the
  head-bound derivation on. Within one epoch exactly one derivation is valid
  (never "either"), so a miner cannot pick the better of two draw sequences.
  The batch envelope stays epoch-labelled (epoch, sampling_context_sha256);
  carrying a batch forward is a cheap re-stamp of those fields, never a
  resample. Unsigned manifests without this policy keep the original
  per-epoch semantics (dual-accept for the previous client).
"""
import copy
import re

POLICY_KEY = 'batch_lifetime_policy'
VERSION = 'replace-until-seal-head-bound-v1'
CONTEXT_KEY = 'batch_lifetime'   # draw-context marker, set by binding() from the signed policy
HEAD_VERSION = 'head-bound-draws-v1'
MAX_UPLOAD_SLOTS = 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _safe_epoch(value):
    return type(value) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value) is not None


def policy(manifest):
    value = manifest.get(POLICY_KEY)
    if value is None:
        return None
    need(type(value) is dict and set(value) == {'version', 'upload_slots'} and value['version'] == VERSION,
         'known signed batch lifetime policy')
    cap = manifest.get('max_batches')
    need(type(cap) is int and cap >= 1 and type(value['upload_slots']) is int
         and cap <= value['upload_slots'] <= min(MAX_UPLOAD_SLOTS, 8 * cap),
         'bounded signed per-hotkey upload slots')
    return value


def upload_slots(manifest):
    value = policy(manifest)
    return manifest['max_batches'] if value is None else value['upload_slots']


def draw_context(context):
    """Context hashed into token draws and behavior witnesses.

    Under the signed lifetime policy the epoch label (the only field that changes
    between epochs on an unchanged head; measured on live e196-e204 openings) is
    replaced by a constant, so draws depend on (head checkpoint, seed, miner,
    task, nonce) only. The host keeps the seed fixed per head (see carry()).
    """
    if context.get(CONTEXT_KEY) is None:
        return context
    out = dict(context)
    out['epoch'] = HEAD_VERSION
    return out


def carry(previous, checkpoint_id, epoch, fresh):
    """Host-private head seed: keep it while the head is unchanged."""
    need(_hash(checkpoint_id) and _safe_epoch(epoch) and _hash(fresh), 'head carry inputs')
    if (type(previous) is dict and previous.get('version') == HEAD_VERSION
            and previous.get('head') == checkpoint_id and _hash(previous.get('randomness'))
            and _safe_epoch(previous.get('origin_epoch'))):
        return dict(version=HEAD_VERSION, head=checkpoint_id, origin_epoch=previous['origin_epoch'],
                    randomness=previous['randomness'])
    return dict(version=HEAD_VERSION, head=checkpoint_id, origin_epoch=epoch, randomness=fresh)


def carries(old_manifest, new_manifest):
    """True when a batch sampled under `old_manifest` has identical draws now."""
    return (policy(old_manifest) is not None and policy(new_manifest) is not None
            and old_manifest['checkpoint']['id'] == new_manifest['checkpoint']['id']
            and old_manifest['sampling_contract']['randomness'] == new_manifest['sampling_contract']['randomness'])


def restamp(batch, old_context, new_context):
    """Re-label a head-bound batch for a later epoch on the same head (no resample)."""
    from .complete_group_sampling import digest
    need(old_context.get(CONTEXT_KEY) is not None and new_context.get(CONTEXT_KEY) is not None,
         'batch can only carry over under the signed lifetime policy')
    need(digest(draw_context(old_context)) == digest(draw_context(new_context)),
         'draw context changed; resample required')
    need(batch.get('epoch') == old_context['epoch'] and batch.get('miner') == new_context['miner']
         and batch.get('sampling_context_sha256') == digest(old_context), 'batch belongs to the old context')
    from .forced_sampling import receipt
    out = copy.deepcopy(batch)
    out['epoch'] = new_context['epoch']
    out['sampling_context_sha256'] = digest(new_context)
    # Per-rollout receipts are epoch labels over the same head-bound draws.
    for nonce, row in enumerate(out['rollouts']):
        need(row.get('sampling') == receipt(old_context, nonce), 'rollout receipt belongs to the old context')
        row['sampling'] = receipt(new_context, nonce)
    return out
