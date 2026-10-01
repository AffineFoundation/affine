"""Frozen-artifact random auditing; unchecked data never enters training."""
import hashlib
import math
import random


def select(total, policy, seed, submission_hash):
    if total < 0:
        raise ValueError('audit population')
    mode = policy.get('mode', 'full')
    if mode == 'full':
        return list(range(total))
    if mode != 'sampled' or not isinstance(seed, str) or len(seed) != 64:
        raise ValueError('post-freeze audit policy')
    try: bytes.fromhex(seed)
    except ValueError: raise ValueError('audit seed encoding')
    count = policy.get('count', 1)
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError('audit count')
    digest = hashlib.sha256((seed+submission_hash).encode()).digest()
    return sorted(random.Random(int.from_bytes(digest, 'big')).sample(range(total), min(count,total)))


def assurance(total, checked, assumed_bad_fraction=.1):
    # Exact probability of detecting >=1 bad batch without replacement given
    # the assumption stated here; no claim every unchecked batch is correct.
    bad = math.ceil(total*assumed_bad_fraction)
    missed = (math.comb(total-bad,checked)/math.comb(total,checked)
              if 0 <= checked <= total-bad and total else 0.)
    return dict(population=total,checked=checked,assumed_bad_fraction=assumed_bad_fraction,
                assumed_bad_count=bad,detection_probability=1-missed if total else None,
                certainty_about_unchecked=False)
