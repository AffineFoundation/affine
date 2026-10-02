"""Hydrate separate task data solely from authenticated manifest capabilities."""
from types import SimpleNamespace

def bindings(manifest):
    from .math_corpus_provider import is_corpus_id,validate
    expected={}
    for row in manifest.get('environments',[]):
        raw=row['spec'];config=raw.get('config',{})
        if not is_corpus_id(raw.get('id')):
            if 'math_corpus_asset' in config:raise ValueError('task asset provider identity')
            continue
        binding=validate(SimpleNamespace(**raw))
        old=expected.setdefault(binding['sha256'],binding)
        if old!=binding:raise ValueError('task asset descriptor collision')
    registry=manifest.get('task_assets',{})
    if not isinstance(registry,dict) or set(registry)!=set(expected):
        raise ValueError('exact signed task asset registry')
    from .source_bootstrap import r2_url
    for sha,binding in expected.items():
        item=registry[sha]
        if (not isinstance(item,dict) or set(item)!={'read_url','compressed_sha256','compressed_size'} or
                item['compressed_sha256']!=binding['compressed_sha256'] or
                type(item['compressed_size'])is not int or item['compressed_size']!=binding['compressed_size']):
            raise ValueError('task asset capability binding')
        r2_url(item['read_url'])
    return expected

def hydrate_manifest(root,manifest,fetch=None):
    expected=bindings(manifest)
    from .math_corpus_assets import hydrate
    return {sha:hydrate(root,binding,manifest['task_assets'][sha]['read_url'],fetch=fetch)
            for sha,binding in expected.items()}
