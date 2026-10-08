"""Bounded CPU comparison of inference checkpoints; does not mutate models."""
import math
from pathlib import Path


def compare_checkpoints(parent, candidate, *, chunk_elements=1_000_000):
    """Report actual BF16 weight changes with bounded tensor chunks in FP32."""
    import torch
    from safetensors import safe_open
    if type(chunk_elements) is not int or not 1 <= chunk_elements <= 4_000_000:
        raise ValueError('bounded comparison chunk')

    def inventory(root):
        rows = {}
        for path in sorted(Path(root).glob('*.safetensors')):
            if path.is_symlink() or not path.is_file():
                raise ValueError('regular inference model shards required')
            with safe_open(path, framework='pt', device='cpu') as reader:
                for name in reader.keys():
                    if name in rows: raise ValueError('unique model tensor names')
                    view = reader.get_slice(name)
                    rows[name] = (path, view.get_shape(), view.get_dtype())
        if not rows: raise ValueError('nonempty inference tensor inventory')
        return rows

    before, after = inventory(parent), inventory(candidate)
    if set(before) != set(after): raise ValueError('matched checkpoint tensor inventory')
    total = changed = 0
    norm = delta = 0.0
    maximum = 0.0
    tensors = []
    for name in sorted(before):
        left, shape, dtype = before[name]
        right, new_shape, new_dtype = after[name]
        if shape != new_shape or dtype != new_dtype:
            raise ValueError('matched checkpoint tensor shape and dtype')
        count = math.prod(shape)
        with safe_open(left, framework='pt', device='cpu') as a, safe_open(right, framework='pt', device='cpu') as b:
            old = a.get_tensor(name).reshape(-1)
            new = b.get_tensor(name).reshape(-1)
            n = d = 0.0
            c = 0
            m = 0.0
            for start in range(0, count, chunk_elements):
                x = old[start:start+chunk_elements].float()
                y = new[start:start+chunk_elements].float()
                if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                    raise ValueError('finite checkpoint weights required')
                difference = y - x
                n += float(torch.sum(x.double().square()))
                d += float(torch.sum(difference.double().square()))
                c += int(torch.count_nonzero(difference))
                m = max(m, float(difference.abs().max()))
            del old, new
        total += count; changed += c; norm += n; delta += d; maximum = max(maximum, m)
        tensors.append(dict(name=name, elements=count, changed_elements=c,
                            relative_l2=math.sqrt(d/n) if n else None, max_absolute_change=m))
    return dict(elements=total, changed_elements=changed, changed_fraction=changed/total,
                relative_l2=math.sqrt(delta/norm) if norm else None,
                max_absolute_change=maximum, tensors=tensors,
                protocol_changed=False, model_mutations=False, learning_gain_claimed=False)
