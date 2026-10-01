"""Prepare signed scoped SDK resources; never exports authority private bytes."""
import argparse,base64,json
from pathlib import Path
from importlib.metadata import version
from nacl.signing import SigningKey
from subnet.environment_resources import collect_dependencies,export_dependencies,canonical
from subnet.adapter_resources import build_contract
from subnet.resource_session import build_spec as bridge_spec,bridge_code_hash
from subnet.environments import build_spec,snapshot_spec

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authority-seed',default='state/service-conformance/authority.seed');args=parser.parse_args()
    root=Path('state/resource-control');root.mkdir(parents=True,exist_ok=True)
    deps=collect_dependencies(['verifiers']);archive=root/'verifiers-namespace.zip';receipt=export_dependencies(deps,archive)
    (root/'verifiers-namespace.descriptor.json').write_bytes(canonical(deps))
    ref={'descriptor_id':deps['id'],'archive_sha256':receipt['archive_sha256'],'object_key':'public/resource-controls/'+deps['id']+'/verifiers-namespace.zip'}
    contract=build_contract(environment_version='prime-resource-controlled-v2',dependencies=ref,
            dependency_scope='provider-namespace-controlled',dependency_closure_reviewed=False)
    base=build_spec('affine_verbatim',{'taskset':{'num_samples':4,'target_length':8,'content_type':'codes'}},num_samples=4,max_turns=1,max_output_tokens=128)
    base=snapshot_spec(base,'state/original-task-snapshots/resource-verbatim-base.tasks.json')
    payload=bridge_spec(base.to_dict(),contract)
    signer=SigningKey(bytes.fromhex(Path(args.authority_seed).read_text().strip()));authority=signer.verify_key.encode().hex()
    envelope={'payload':payload,'signer':authority,'signature':base64.b64encode(signer.sign(canonical(payload)).signature).decode()}
    outer={**base.to_dict(),'adapter':'resource_prime_v1_controlled','version':contract['environment_version'],
           'source_hash':payload['source_hash'],'config':{'signed_resource_spec':envelope,'authority':authority}}
    (root/'verbatim-controlled.spec.json').write_text(json.dumps(outer,indent=2)+'\n')
    (root/'verbatim-base.spec.json').write_text(json.dumps(base.to_dict(),indent=2)+'\n')
    bindings={'descriptors':{deps['id']:deps},'archives':{deps['id']:str(archive.resolve())},'cache':str((root/'operator-cache').resolve()),'audience':'verifier'}
    (root/'operator-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');(root/'operator-bindings.json').chmod(0o600)
    remote='/root/affine-resource-control'
    remote_bindings={**bindings,'archives':{deps['id']:remote+'/state/resource-control/verifiers-namespace.zip'},'cache':remote+'/state/resource-control/miner-cache','audience':'miner'}
    (root/'remote-miner-bindings.json').write_text(json.dumps(remote_bindings,indent=2)+'\n')
    report={'adapter':'resource_prime_v1_controlled','source_hash':payload['source_hash'],'bridge_code_hash':bridge_code_hash(),
        'contract_id':contract['id'],'provider_descriptor_id':deps['id'],'archive_sha256':receipt['archive_sha256'],'archive_bytes':archive.stat().st_size,
        'authority':authority,'dependency_scope':'provider-namespace-controlled','dependency_closure_reviewed':False,
        'coverage':'reviewed verifiers namespace source and packaged bytes; transitive/native closure remains unverified',
        'platform_inventory_observation':{name:version(name) for name in ('torch','transformers','numpy','pydantic','pydantic-core','PyNaCl')},
        'platform_attestation':False,'private_grader_bundle_transported':False}
    (root/'coverage.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
