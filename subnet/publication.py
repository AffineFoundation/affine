"""Renewable signed audit routes over immutable, frozen public artifacts."""
import json
import time
from pathlib import PurePosixPath
from .storage import sha


def public_key(key):
    path=PurePosixPath(key)
    if not isinstance(key,str) or not key.startswith('public/') or '..' in path.parts or '\\' in key:
        raise ValueError('audit route must refer to a public artifact')
    return key


def publish_source_bundle(controller,path):
    """Caller supplies an already reviewed, credential-free source archive."""
    data=path.read_bytes();digest=sha(data);key=f'public/sources/{digest}/source.tar.gz'
    try:existing=controller.bucket.get(key)
    except Exception as exc:
        from botocore.exceptions import ClientError
        if not isinstance(exc,(KeyError,FileNotFoundError)) and not (isinstance(exc,ClientError) and str(exc.response.get('Error',{}).get('Code')) in ('404','NoSuchKey','NotFound')):raise
        controller.bucket.put(key,data)
    else:
        if sha(existing)!=digest:raise ValueError('immutable source archive changed')
    return dict(key=key,sha256=digest,size=len(data))


def source_archive_key(controller,source):
    """Resolve bootstrap descriptors without rewriting their signed payload.

    Bootstrap descriptors carry archive URLs and digests, while older history
    descriptors carry a bucket key. Private bootstrap archives are copied to
    the public content-addressed namespace only after their digest and size
    match. Original epoch descriptors and private objects remain untouched.
    """
    if 'key' in source:
        return public_key(source['key'])
    digest=source.get('sha256');size=source.get('size')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('source archive digest')
    if type(size)is not int or size<=0:
        raise ValueError('source archive size')
    key=f'public/sources/{digest}/source.tar.gz'
    try:data=controller.bucket.get(key)
    except Exception as exc:
        from botocore.exceptions import ClientError
        if not isinstance(exc,(KeyError,FileNotFoundError)) and not (isinstance(exc,ClientError) and str(exc.response.get('Error',{}).get('Code')) in ('404','NoSuchKey','NotFound')):raise
        # This is the frozen-source publisher's specific namespace, never a
        # miner staging path or an arbitrary key parsed from an uploaded URL.
        data=controller.bucket.get(f'private/source-bundles/{digest}.tar.gz')
        if sha(data)!=digest or len(data)!=size:
            raise ValueError('source archive content mismatch')
        controller.bucket.put(key,data)
    if sha(data)!=digest or len(data)!=size:
        raise ValueError('source archive content mismatch')
    return key


def history(controller,ledger,source_bundle=None,source_reconstructions=()):
    rows=[]
    def route(key):return controller.bucket.presign(public_key(key))
    for result in ledger:
        epoch=result['epoch_id']
        if '/' in epoch or '..' in epoch:raise ValueError('epoch audit path')
        manifest=json.loads((controller.state/f'{epoch}-manifest.json').read_text())
        objects={name:route(f'public/{epoch}/{name}.json') for name in ('manifest','scores','audit-challenge','receipts','training')}
        frozen={miner:dict(url=route(receipt['frozen_key']),sha256=receipt['sha256'],size=receipt.get('size')) for miner,receipt in result['receipts'].items()}
        audits={miner:route(f'public/{epoch}/audits/{miner}.json') for miner in result['receipts']}
        approved=manifest['checkpoint']
        checkpoint=dict(id=approved['id'],files=approved['files'],read_urls={name:route(f"public/checkpoints/{approved['id']}/{name}") for name in approved['files']})
        source=manifest.get('source_bundle') or source_bundle
        if source:
            source_url=route(source_archive_key(controller,source))
            source=dict(source,url=source_url,read_url=source_url,binding='epoch-signed' if manifest.get('source_bundle') else 'reviewed-compatible-reference')
        output=None
        metrics_path=controller.state/f'{epoch}-training-metrics.json'
        if metrics_path.exists():
            trained=json.loads(metrics_path.read_text())['checkpoint']
            descriptor_key=f'public/checkpoints/{trained}/authorities/{controller.authority.id}/checkpoint.json'
            try:signed=json.loads(controller.bucket.get(descriptor_key))
            except Exception as exc:
                from botocore.exceptions import ClientError
                if not isinstance(exc,ClientError) or str(exc.response.get('Error',{}).get('Code')) not in ('404','NoSuchKey','NotFound'):raise
                descriptor_key=f'public/checkpoints/{trained}/checkpoint.json'
                signed=json.loads(controller.bucket.get(descriptor_key))
            from nacl.signing import VerifyKey
            import base64
            from .storage import canonical
            if signed['signer']!=controller.authority.id:raise ValueError('trained checkpoint authority')
            VerifyKey(bytes.fromhex(controller.authority.id)).verify(canonical(signed['payload']),base64.b64decode(signed['signature'],validate=True))
            descriptor=signed['payload']
            if descriptor['id']!=trained:raise ValueError('trained checkpoint identity')
            output=dict(id=trained,files=descriptor['files'],descriptor_url=route(descriptor_key),read_urls={name:route(f'public/checkpoints/{trained}/{name}') for name in descriptor['files']})
        rows.append(dict(epoch_id=epoch,payable=result.get('payable',False),deadline=manifest['deadline'],checkpoint=checkpoint,trained_checkpoint=output,objects=objects,audits=audits,frozen=frozen,source_bundle=source))
    supplements=[]
    for descriptor in source_reconstructions:
        if descriptor.get('binding')!='reviewed-reconstruction-not-original-epoch-archive':
            raise ValueError('source reconstruction provenance')
        key=public_key(descriptor['key']);digest=descriptor['sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('source reconstruction digest')
        data=controller.bucket.get(key)
        if sha(data)!=digest or len(data)!=descriptor['size']:
            raise ValueError('source reconstruction content mismatch')
        supplements.append(dict(descriptor,read_url=route(key)))
    return dict(version=1,authority=controller.authority.id,refreshed_at=time.time(),routes_expire_at=time.time()+604800,epochs=rows,source_reconstruction_supplements=supplements)


def publish_history(controller,prefix,ledger,source_bundle=None,source_reconstructions=()):
    key=public_key(f'public/streams/{prefix}/history.json')
    controller.bucket.json(key,controller.signed(history(controller,ledger,source_bundle,source_reconstructions)))
    return controller.bucket.presign(key)
