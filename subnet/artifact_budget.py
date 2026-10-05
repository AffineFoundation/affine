"""Operator-selected transport budgets, never inferred from submitted ZIPs."""
LEGACY = dict(compressed_bytes=100_000_000, raw_bytes=500_000_000, tensor_rows=512)
LONG_REVISION = 'full-vocabulary-long-v1'
LONG = dict(compressed_bytes=2_000_000_000, raw_bytes=3_000_000_000, tensor_rows=2048)

def for_manifest(manifest):
    selected=manifest.get('artifact_policy')
    if selected is None:return dict(LEGACY)
    from .backend_profiles import HOPPER_REVISION,HOPPER_FP32_REVISION
    if selected!=LONG_REVISION or manifest.get('model_runtime_revision')not in (HOPPER_REVISION,HOPPER_FP32_REVISION):
        raise ValueError('unapproved signed artifact policy')
    from .backend_profiles import resolve
    resolve(manifest)
    return dict(LONG)
