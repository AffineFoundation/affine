"""Renewable signed audit routes over immutable, frozen public artifacts."""
import base64
import json
import time
from pathlib import Path,PurePosixPath
from .storage import sha,canonical


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


def frozen_submission(controller,manifest,miner,receipt):
    """Renew frozen routes without promoting declared child hashes to audits."""
    def route(key):return controller.bucket.presign(public_key(key))
    from .commitment_transport import VERSION,validate,is_digest,MAX_BYTES
    if manifest.get('submission_transport_policy') != VERSION:
        if 'commitment_key' in receipt or 'artifacts' in receipt:
            raise ValueError('commitment history requires signed transport policy')
        return dict(url=route(receipt['frozen_key']),sha256=receipt['sha256'],size=receipt.get('size'))
    if not is_digest(receipt.get('sha256')) or type(receipt.get('size')) is not int:
        raise ValueError('commitment history digest/size')
    root=f"public/{manifest['epoch']}/submissions/{miner}/{receipt['sha256']}"
    key=public_key(receipt['commitment_key'])
    if key!=root+'/commitment.json':raise ValueError('frozen commitment history scope')
    # Read only the bounded commitment. Child ZIP hashes remain declared;
    # actual verifier evidence is available through the separate audit route.
    if not 0<receipt['size']<=MAX_BYTES:raise ValueError('bounded commitment history')
    binding=dict(manifest_sha256=sha(canonical(manifest)),receipt_sha256=sha(canonical(receipt)),
        epoch=manifest['epoch'],miner=miner,source=manifest['source_bundle']['sha256'],
        commitment_key=key,sha256=receipt['sha256'],size=receipt['size'])
    folder=Path(controller.state)/'history-commitment-admissions'
    journal=folder/(sha(canonical(binding))+'.json')
    if folder.is_symlink():raise ValueError('commitment history journal directory')
    cached=journal.exists() or journal.is_symlink()
    if cached:
        if journal.is_symlink() or not journal.is_file():raise ValueError('commitment history journal file')
        with journal.open('rb')as stream:encoded=stream.read(2*MAX_BYTES+1)
        if len(encoded)>2*MAX_BYTES:raise ValueError('bounded commitment history journal')
        saved=json.loads(encoded)
        if (set(saved)!={'version','binding','original_bytes_base64'} or
                saved['version']!='frozen-commitment-history-admission-v1' or saved['binding']!=binding):
            raise ValueError('commitment history journal binding')
        data=base64.b64decode(saved['original_bytes_base64'],validate=True)
    elif hasattr(controller.bucket,'client'):
        from .commitment_transport import _read_small_commitment
        data=_read_small_commitment(controller.bucket,key)['data']
    else:data=controller.bucket.get(key)
    if len(data)!=receipt['size'] or sha(data)!=receipt['sha256']:
        raise ValueError('frozen commitment history content mismatch')
    # Historical frozen receipts can predate the prospective canonical wire gate.
    # Authenticate the parsed signed document while retaining its original byte SHA.
    document=validate(canonical(json.loads(data)),manifest['epoch'],miner)
    if (document!=receipt['commitment_document'] or
            document['payload']['checkpoint']!=manifest['checkpoint']['id'] or
            document['payload']['source']!=manifest['source_bundle']['sha256']):
        raise ValueError('frozen commitment history model/source binding')
    declared=document['payload']['batches'];artifacts=receipt['artifacts']
    if type(artifacts)is not list or len(artifacts)!=len(declared):
        raise ValueError('frozen commitment history inventory')
    children=[]
    for batch,artifact in zip(declared,artifacts):
        if (any(artifact.get(field)!=value for field,value in batch.items()) or
                artifact.get('frozen_key')!=root+'/'+str(batch['slot'])+'.zip'):
            raise ValueError('frozen commitment history inventory')
        if manifest.get('proof_copy_policy') is not None:
            from .selected_proof_copy import validate_policy
            validate_policy(manifest['proof_copy_policy'])
            copied=controller.gateway.epochs[manifest['epoch']].get('selected_proof_copies',{}).get(miner,{}).get(str(batch['slot']))
            expected={k:artifact[k]for k in ('sha256','size','etag','key','frozen_key')}
            if copied is not None and copied!=expected:raise ValueError('public selected proof copy binding')
            children.append(dict(batch,**({'url':route(artifact['frozen_key']),'availability':'copied-selected-proof'}if copied is not None else {'availability':'not-publicly-copied'}),hash_assurance='declared-payload-hash-until-selected-verifier'))
        else:
            children.append(dict(batch,url=route(artifact['frozen_key']),
                hash_assurance='declared-payload-hash-until-selected-verifier'))
    if not cached:
        # Root-local immutable admissions bind original bytes, not renewable URLs.
        # Every reuse still authenticates these bytes and the complete inventory.
        from .controller import save_manifest
        folder.mkdir(mode=0o700,parents=True,exist_ok=True)
        if folder.is_symlink():raise ValueError('commitment history journal directory')
        save_manifest(journal,dict(version='frozen-commitment-history-admission-v1',
            binding=binding,original_bytes_base64=base64.b64encode(data).decode()))
        journal.chmod(0o600)
    commitment=dict(url=route(key),sha256=receipt['sha256'],size=receipt['size'])
    return dict(transport_policy=VERSION,commitment=commitment,artifacts=children,
        hash_assurance='declared-payload-hashes-until-selected-verifier')


def infrastructure_history(controller):
    """Separate incomplete captures from finalized scientific epoch history."""
    ledger=controller.state/'infrastructure-skipped-epochs.json'
    if not ledger.exists():return []
    from .backend_jobs import signed
    rows=[]
    for closure in json.loads(ledger.read_bytes()):
        epoch=closure['epoch']
        if not isinstance(epoch,str) or '/'in epoch or '..'in epoch:raise ValueError('infrastructure history epoch')
        raw=(controller.state/(epoch+'-capture-status.json')).read_bytes()
        envelope=json.loads(raw);payload=signed(envelope,controller.authority.id)
        manifest=json.loads((controller.state/(epoch+'-manifest.json')).read_bytes())
        if (closure.get('status')!='infrastructure_skipped_metadata_incomplete' or
                closure.get('capture_status_sha256')!=sha(canonical(envelope)) or
                type(closure.get('training_updates'))is not int or closure['training_updates']!=0 or
                any(closure.get(k)is not False for k in ('verification_claim','payable','chain_transactions')) or
                payload.get('version')!='commitment-capture-status-v1' or payload.get('epoch')!=epoch or
                payload.get('status')!='metadata_incomplete' or
                any(payload.get(k)is not False for k in ('complete','verification_claim','audits_started','rewards_eligible')) or
                type(payload.get('accepted_batches'))is not int or payload['accepted_batches']!=0 or
                payload.get('manifest_sha256')!=sha(canonical(manifest)) or
                payload.get('source_sha256')!=manifest['source_bundle']['sha256'] or
                payload.get('checkpoint')!=manifest['checkpoint']['id'] or
                closure.get('checkpoint')!=manifest['checkpoint']['id']):
            raise ValueError('incomplete infrastructure history binding')
        rows.append(dict(closure,capture_status=dict(
            url=controller.bucket.presign(public_key('public/'+epoch+'/capture-status.json')),
            sha256=sha(raw),size=len(raw)),infrastructure_status_url=controller.bucket.presign(
                public_key('public/'+epoch+'/infrastructure-skipped.json'))))
    return rows


def history(controller,ledger,source_bundle=None,source_reconstructions=()):
    rows=[]
    def route(key):return controller.bucket.presign(public_key(key))
    for result in ledger:
        epoch=result['epoch_id']
        if '/' in epoch or '..' in epoch:raise ValueError('epoch audit path')
        manifest=json.loads((controller.state/f'{epoch}-manifest.json').read_text())
        objects={name:route(f'public/{epoch}/{name}.json') for name in ('manifest','scores','audit-challenge','receipts','training')}
        frozen={miner:frozen_submission(controller,manifest,miner,receipt) for miner,receipt in result['receipts'].items()}
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
    return dict(version=1,authority=controller.authority.id,refreshed_at=time.time(),routes_expire_at=time.time()+604800,epochs=rows,source_reconstruction_supplements=supplements,infrastructure_skips=infrastructure_history(controller))


def publish_history(controller,prefix,ledger,source_bundle=None,source_reconstructions=()):
    key=public_key(f'public/streams/{prefix}/history.json')
    controller.bucket.json(key,controller.signed(history(controller,ledger,source_bundle,source_reconstructions)))
    return controller.bucket.presign(key)
