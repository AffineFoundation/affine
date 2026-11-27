"""v8.3 declared training kernels. The signed runtime fingerprint pins the optional fast-path packages to None
(sampling/evaluation must not use them). The v8.3 TRAINER host installs them for its training forwards only and
declares the exact versions in its ROOT-admitted host config (engine.training_kernels). allow() makes
subnet.mimo_runtime.runtime_fingerprint() report those packages as None ONLY if the installed versions equal the
declared ones; anything else (another version, an undeclared package) is reported truthfully and fails the pin."""
import importlib.metadata
DECLARED = {'flash-linear-attention': '0.6.0', 'fla-core': '0.6.0', 'causal-conv1d': None, 'flash-attn': None}


def installed():
    out = {}
    for name in DECLARED:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def allow(declared=None):
    import subnet.mimo_runtime as R
    declared = dict(DECLARED if declared is None else declared)
    if declared != DECLARED:
        raise ValueError('v8.3 training kernel declaration differs from the host module')
    if getattr(R.runtime_fingerprint, '_v83', False):
        return R.runtime_fingerprint._kernels
    real = installed()
    if real != declared:
        raise ValueError('installed training kernels differ from the declaration: %r' % real)
    original = R.runtime_fingerprint

    def runtime_fingerprint():
        fp = original()
        for name, want in DECLARED.items():
            if fp['versions'].get(name) == want:   # exactly the declared training kernel -> the pinned None
                fp['versions'][name] = None
        return fp                                   # anything undeclared stays visible and fails the pin
    runtime_fingerprint._v83 = True
    runtime_fingerprint._kernels = real
    runtime_fingerprint.__qualname__ = original.__qualname__
    R.runtime_fingerprint = runtime_fingerprint
    return real
