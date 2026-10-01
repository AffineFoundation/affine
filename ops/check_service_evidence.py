"""Read-only audit of completed persistent-controller epochs, never chain writes."""
import argparse
import base64
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.parse import urlsplit

from nacl.signing import VerifyKey
from ops.check_epoch_evidence import checkpoint_descriptor, digest, require
from subnet.chain import hourly_points
from subnet.storage import Bucket, Identity, canonical


def direct_read_routes(manifest):
    urls=manifest['checkpoint'].get('read_urls')
    direct=manifest.get('transport_policy') == 'direct-r2-v1' or urls is not None
    if not direct:return False
    require(isinstance(urls,dict) and set(urls)==set(manifest['checkpoint']['files']),
            'incomplete direct checkpoint routes')
    for url in urls.values():
        parts=urlsplit(url)
        require(parts.scheme=='https' and parts.hostname is not None
                and parts.hostname.endswith('.r2.cloudflarestorage.com')
                and parts.username is None and parts.password is None,
                'checkpoint route is not direct R2')
    return True


def check(state, bucket, evaluation_root):
    # Local operator state is the explicit trust anchor; no submitted key is trusted.
    authority = Identity(bytes.fromhex((state/'authority.seed').read_text())).id
    def signed(key):
        envelope = json.loads(bucket.get(key))
        require(envelope['signer'] == authority, 'wrong artifact authority: '+key)
        VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),
            base64.b64decode(envelope['signature'], validate=True))
        return envelope['payload']

    checked, scores_seen, pending, empty = [], [], [], []
    manifests = [json.loads(p.read_text()) for p in state.glob('*-manifest.json')
                 if not p.name.endswith('-audit-manifest.json')]
    for local_manifest in sorted(manifests, key=lambda m:m['start']):
        epoch = local_manifest['epoch']
        empty_path=state/f'{epoch}-empty-closed.json'
        if empty_path.exists():
            manifest=signed(f'public/{epoch}/manifest.json')
            require(manifest==local_manifest and manifest['payable'] is False
                    and epoch.startswith('nonpayable-'),'empty epoch manifest binding')
            direct_read_routes(manifest)
            scores=signed(f'public/{epoch}/scores.json')
            require(scores['epoch_id']==epoch and scores['payable'] is False
                    and scores['checkpoint']==manifest['checkpoint']['id'],'empty epoch score binding')
            require(scores['finalized_at']>=manifest['deadline'],'empty epoch froze early')
            require(scores['total']==0 and not scores['receipts'] and not scores['weights']
                    and not any(scores['points'].values()),'nonempty epoch discarded')
            training=signed(f'public/{epoch}/training.json')
            require(training.get('status') in ('paused_no_verified_pairs','paused_no_verified_training_data'),
                    'empty epoch training status')
            following=[m for m in manifests if m['start']>=manifest['deadline'] and m['epoch']!=epoch]
            next_manifest=min(following,key=lambda m:m['start']) if following else None
            if next_manifest:
                require(signed(f'public/{next_manifest["epoch"]}/manifest.json')==next_manifest,
                        'empty epoch next manifest signature binding')
                require(next_manifest['checkpoint']['id']==manifest['checkpoint']['id'],
                        'empty epoch changed checkpoint without training')
            empty.append(dict(epoch=epoch,deadline_honored=True,zero_submissions=True,
                              checkpoint=manifest['checkpoint']['id'],payable=False,
                              next_epoch=next_manifest['epoch'] if next_manifest else None))
            scores_seen.append(scores)
            continue
        metrics_path = state/f'{epoch}-training-metrics.json'
        evaluation_paths = [evaluation_root/f'{epoch}-{phase}-{suite["env_id"]}.json'
                            for suite in local_manifest.get('evaluation',{}).get('suites',[])
                            for phase in ('before','after')]
        if not metrics_path.exists() or any(not p.exists() for p in evaluation_paths):
            pending.append(epoch)
            continue
        manifest = signed(f'public/{epoch}/manifest.json')
        require(manifest == local_manifest, 'local/public manifest mismatch')
        direct=direct_read_routes(manifest)
        require(epoch.startswith('nonpayable-') and manifest['payable'] is False, 'payable pilot epoch')
        scores = signed(f'public/{epoch}/scores.json')
        challenge = signed(f'public/{epoch}/audit-challenge.json')
        require(scores['epoch_id'] == epoch and scores['payable'] is False, 'score epoch binding')
        require(scores['checkpoint'] == manifest['checkpoint']['id'], 'score model binding')
        require(scores['finalized_at'] >= manifest['deadline'], 'controller froze before deadline')
        require(challenge['generated_after_freeze_at'] >= manifest['deadline'], 'early audit challenge')
        require(challenge['receipts'] == scores['receipts'], 'challenge receipt mismatch')
        accepted, envs = 0, set()
        for miner, receipt in scores['receipts'].items():
            require(manifest['start'] <= receipt['received_at'] <= manifest['deadline'], 'upload outside epoch window')
            if direct:
                require(receipt['received_at'] < manifest['deadline'], 'direct upload completed at deadline')
                require(receipt.get('snapshot_key') and receipt.get('etag') and receipt.get('size',0)>0,
                        'missing direct R2 snapshot metadata')
            frozen = bucket.get(receipt['frozen_key'])
            require(hashlib.sha256(frozen).hexdigest() == receipt['sha256'], 'frozen artifact changed')
            audit = signed(f'public/{epoch}/audits/{miner}.json')
            require(audit['epoch'] == epoch and audit['submission_sha256'] == receipt['sha256'], 'audit binding')
            require(audit['audit_seed'] == challenge['seed'], 'audit randomness binding')
            require(all(o.get('fully_audited') for o in audit['outcomes'] if o.get('valid')), 'unchecked accepted data')
            accepted += len(audit['accepted'])
            envs.update(b['env_id'] for b in audit['accepted'])
        require(accepted > 0, 'no independently accepted training batches')
        weights = scores['weights']
        require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in weights.values()), 'invalid weight')
        require(math.isclose(sum(weights.values()),1) if scores['total'] else not weights, 'unnormalized weights')
        training = signed(f'public/{epoch}/training.json')
        require(training == json.loads(metrics_path.read_text()), 'training report mismatch')
        require(training['weights_changed'] and training['steps'] > 0, 'missing actual training')
        folder = state/'checkpoints'/epoch
        files = {p.name:digest(p) for p in folder.iterdir() if p.is_file()
                 and p.suffix in ('.json','.safetensors','.txt','.model','.jinja','.bin','.pt','.tiktoken')}
        require(hashlib.sha256(canonical(files)).hexdigest() == training['checkpoint'], 'trained checkpoint byte binding')
        descriptor=checkpoint_descriptor(bucket,training['checkpoint'],authority)
        require(descriptor['files']==files,'signed checkpoint file map mismatch')
        old = {k:v for k,v in manifest['checkpoint']['files'].items() if k.endswith('.safetensors')}
        new = {k:v for k,v in files.items() if k.endswith('.safetensors')}
        require(old and new and old != new, 'model weight bytes unchanged')
        paired = []
        for suite in manifest.get('evaluation',{}).get('suites',[]):
            env = suite['env_id']
            before = json.loads((evaluation_root/f'{epoch}-before-{env}.json').read_text())
            after = json.loads((evaluation_root/f'{epoch}-after-{env}.json').read_text())
            require(before['status'] == after['status'] == 'complete', 'unfinished held-out evaluation')
            require(before['dataset_id'] == after['dataset_id'] and before['task_hashes'] == after['task_hashes']
                    and before['runtime_profile'] == after['runtime_profile'], 'incomparable held-out evaluations')
            require(before['checkpoint'] == manifest['checkpoint']['id'] and after['checkpoint'] == training['checkpoint'], 'evaluation model binding')
            paired.append(dict(env_id=env,before_reward=before['mean_reward'],after_reward=after['mean_reward']))
        scores_seen.append(scores)
        following = [m for m in manifests if m['start'] >= manifest['deadline']
                     and m['epoch'] != epoch]
        next_manifest = min(following,key=lambda m:m['start']) if following else None
        if next_manifest:
            require(signed(f'public/{next_manifest["epoch"]}/manifest.json') == next_manifest, 'next manifest signature binding')
            require(next_manifest['checkpoint']['id'] == training['checkpoint'], 'next epoch used stale checkpoint')
        checked.append(dict(epoch=epoch,deadline_honored=True,signatures_verified=True,
                            frozen_bytes_verified=True,accepted_batches=accepted,environments=sorted(envs),
                            proposed_weights_normalized=True,payable=False,checkpoint=training['checkpoint'],
                            model_weight_bytes_changed=True,heldout_pairs=paired,
                            direct_r2_file_routes_verified=direct,
                            next_epoch=next_manifest['epoch'] if next_manifest else None))
    require(hourly_points(scores_seen,(int(time.time())//3600+1)*3600) == {}, 'pilot reached payout aggregation')
    return dict(timestamp=time.time(),epochs=checked,empty_epochs=empty,pending_epochs=pending,chain_write_operations=0,
                payout_filter_result={},goal_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,default=Path('state/service-conformance'))
    parser.add_argument('--bucket-config',type=Path,default=Path('state/mock-r2.json'))
    parser.add_argument('--evaluations',type=Path,default=Path('state/evaluations'))
    parser.add_argument('--output',type=Path,default=Path('state/service-conformance/independent-evidence.json'))
    args=parser.parse_args()
    result=check(args.state,Bucket(json.loads(args.bucket_config.read_text())),args.evaluations)
    args.output.write_text(json.dumps(result,indent=2)+'\n');args.output.chmod(0o600)
    print(json.dumps({'epochs_verified':len(result['epochs']),'pending_epochs':result['pending_epochs'],
                      'output':str(args.output)}))


if __name__=='__main__':
    main()
