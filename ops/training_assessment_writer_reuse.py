"""Refresh learner status, reusing only the reviewed fresh hourly writer artifact.

Prospective CPU-only candidate. The original writer envelope is copied intact;
there is no dependency on chain inclusion and no new evidence producer.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from nacl.exceptions import BadSignatureError


# These exact two ROOT policies have equivalent fresh evidence calculation.
# Their reader, audit semantics, source/numerical admission and verifier roster
# were compared independently; fallback/transaction semantics are NOT reusable.
REVIEWED_REUSE = dict(
    producer_policy_sha256='6546841061eed97d7b450c357c747a1186330daa4b4523cb44b63bad2d74c209',
    writer_policy_sha256='dbac8a75fc94e0f8afbf5aff01b418fe022fd5f12c70ab7cabb671073dd8408a',
    audit_config_file_sha256='57b4491f0b185309a24992fb2a9134fc0fa9290cceb49f2026e5a6f98a7dfca3',
    source_admission_sha256='4323f7c08e13de047543cf71087b33000bba322d8802a98adfff93bedaa64655',
    numerical_resolution_policy_sha256='54617f4aab8082658bd4ae2c84c4da27ffaa0e903212690c83c3ea7f2457028a',
    directory='/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-never-burn-v1')
MAX_ASSESSMENT_BYTES = 32 * 1024**2


def reusable(assessment, cutoff, writer_policy_sha256, source_admission_sha256):
    """Original producer cache rule; never renew its evidence timestamps."""
    return (assessment.get('version') == 'hourly-current-miner-assessment-v1'
            and assessment.get('cutoff') == cutoff
            and assessment.get('evidence_cutoff') == cutoff
            and assessment.get('assessment_stale') is False
            and assessment.get('writer_policy_sha256') == writer_policy_sha256
            and type(assessment.get('miner_estimates')) is dict
            and assessment.get('evidence_hashes', {}).get('source_admission_sha256')
                == source_admission_sha256)


def read_owned(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError('ordinary absolute assessment file')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or not 0 < before.st_size <= MAX_ASSESSMENT_BYTES):
            raise ValueError('bounded owned regular assessment file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(before.st_size + 1)
        after = os.fstat(fd)
        if (len(raw) != before.st_size or
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ValueError('assessment changed while reading')
    finally:
        os.close(fd)
    return raw


def atomic_output(path, raw):
    """Unique temporary path; failures preserve the previous signed envelope."""
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError('ordinary assessment output')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def current_hour(now):
    return int(now()) // 3600 * 3600


def writer_admissible(a, p, producer_sha, cutoff, config_sha, review=REVIEWED_REUSE):
    """Only same-input fresh W results; stale payout fallback is never training status."""
    if (producer_sha != review['producer_policy_sha256'] or
            config_sha != review['audit_config_file_sha256'] or
            any(p.get(k) != review[k] for k in
                ('source_admission_sha256', 'numerical_resolution_policy_sha256')) or
            not reusable(a, cutoff, review['writer_policy_sha256'], review['source_admission_sha256'])):
        return False
    hashes = a.get('evidence_hashes', {})
    if any(hashes.get(k) != review[k] for k in
           ('audit_config_file_sha256', 'source_admission_sha256', 'numerical_resolution_policy_sha256')):
        return False
    if (type(a.get('cutoff')) is not int or type(a.get('evidence_cutoff')) is not int or
            a.get('half_life_hours') != 6 or a.get('history_hours') != 168 or
            a.get('smoothing_basis') != 'estimated-valid-contribution-before-penalty' or
            a.get('penalties_applied_after_smoothing') is not True or
            a.get('training_completion_required') is not False or
            a.get('unaudited_samples_claimed_verified') is not False):
        return False
    alpha = a.get('hourly_alpha')
    if (type(alpha) not in (int, float) or not math.isfinite(alpha) or
            not math.isclose(alpha, 1 - 2 ** (-1 / 6), rel_tol=1e-12)):
        return False
    # Full target-round expiry and confirmed-invalid evidence validation remain
    # in the unchanged selection.admit/partition and CPU operator admission.
    return all(isinstance(m, str) and len(m) == 64 and
               all(c in '0123456789abcdef' for c in m) and
               type(d) is dict and type(d.get('blacklisted')) is bool
               for m, d in a['miner_estimates'].items())


def reuse_cache(output, p, producer_sha, authority, *, signed, now=time.time,
                review=REVIEWED_REUSE):
    """Bounded read-only source lookup plus exact-envelope publication on hit.

    A miss invokes the original G producer in main. No wait for a chain receipt,
    producer process, queue lock or network service is introduced here.
    """
    cutoff = current_hour(now)
    out = Path(output)
    config_sha = None
    paths = [out, Path(review['directory']) / ('assessment-' + str(cutoff) + '.json')]
    for path in paths:
        try:
            raw = read_owned(path)
            a = signed(json.loads(raw), authority)
            if path == out and reusable(a, cutoff, producer_sha, p['source_admission_sha256']):
                producer = 'original'
            else:
                if config_sha is None:
                    config_sha = hashlib.sha256(read_owned(p['audit_config'])).hexdigest()
                if not writer_admissible(a, p, producer_sha, cutoff, config_sha, review):
                    continue
                producer = 'hourly-writer'
            if current_hour(now) != cutoff:
                return None
            if path != out:
                atomic_output(out, raw)  # Preserve exact bytes/signature/producer.
            if current_hour(now) != cutoff:
                return None
            return dict(assessment_reused=True, producer=producer, cutoff=cutoff,
                        chain_transactions=False)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, BadSignatureError):
            continue
    return None


def main():
    a = argparse.ArgumentParser()
    for name in ('runtime', 'writer-policy', 'authority-seed', 'output'):
        a.add_argument('--' + name, required=True)
    args = a.parse_args()
    sys.path.insert(0, args.runtime)
    from nacl.signing import SigningKey
    from subnet.storage import canonical
    from subnet.live_reward_bridge import signed, sha
    from ops.current_assessment_evidence import load_evidence
    from subnet.current_assessment import calculate
    document = json.loads(Path(args.writer_policy).read_bytes())
    key = SigningKey(bytes.fromhex(Path(args.authority_seed).read_text().strip()))
    authority = key.verify_key.encode().hex(); p = signed(document, authority)
    for path, expected in p['module_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('pinned assessment implementation')
    hit = reuse_cache(args.output, p, sha(document), authority, signed=signed, now=time.time)
    if hit is not None:
        print(json.dumps(hit)); return
    # Original evidence calculation and fresh producer signature are unchanged.
    cutoff = int(time.time()) // 3600 * 3600
    kwargs = dict(expected_numerical_resolution_policy_sha256=p['numerical_resolution_policy_sha256']) if 'numerical_resolution_policy_sha256' in p else {}
    evidence = load_evidence(p['audit_config'], authority=authority, cutoff=cutoff,
        verifiers=p['verifiers'], expected_source_admission_sha256=p['source_admission_sha256'], **kwargs)
    assessment = calculate(evidence['snapshots'], evidence['committed_at_by_epoch'], cutoff)
    assessment.update(evidence_cutoff=cutoff, assessment_stale=False,
        evidence_hashes=evidence['evidence_hashes'], evidence_refusals=evidence.get('refused', []),
        evidence_exclusions=evidence.get('excluded', []), evidence_deferrals=evidence.get('deferred', []),
        writer_policy_sha256=sha(document))
    envelope = dict(payload=assessment, signer=authority,
        signature=base64.b64encode(key.sign(canonical(assessment)).signature).decode())
    atomic_output(Path(args.output), canonical(envelope))
    print(json.dumps(dict(assessment_refreshed=True, cutoff=cutoff,
                         miners=len(assessment['miner_estimates']), chain_transactions=False)))


if __name__ == '__main__': main()
