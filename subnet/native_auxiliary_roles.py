"""Environment-independent, operator-authenticated model role admission.

Auxiliary tokens authenticate observations but are never agent training targets.
Model computation evidence does not establish unbiased sampling provenance.
"""
from .native_tau2_model import authenticate
from .native_tau2_probe import digest

VERSION='native-auxiliary-role-contract-v1'

def validate_contract(signed, authority):
    contract=authenticate(signed,authority)
    if contract.get('version')!=VERSION or contract.get('objective')!='agent-only-curated-supervised-v1':raise ValueError('role objective')
    roles=contract.get('roles',{})
    if not roles or not any(r.get('kind')=='agent' for r in roles.values()):raise ValueError('agent role required')
    for name,role in roles.items():
        if role.get('kind') not in ('agent','auxiliary') or role.get('training_eligible')!=(role['kind']=='agent'):raise ValueError('auxiliary loss mask')
        if not isinstance(role.get('checkpoint'),str) or len(role['checkpoint'])!=64 or not role.get('source_hash') or not role.get('numerical_policy'):raise ValueError('pinned role model')
    if contract.get('payable') is not False:raise ValueError('native controlled contract nonpayable')
    return contract

def admit_records(signed_contract,signed_records,signed_audit,authority):
    """An audit signature binds exact contract and receipts, never a caller flag."""
    contract=validate_contract(signed_contract,authority)
    records=[authenticate(r,authority) for r in signed_records]
    audit=authenticate(signed_audit,authority)
    if audit.get('contract_hash')!=digest(contract) or audit.get('receipts_hash')!=digest(signed_records) or audit.get('full_native_trajectory_verified') is not True or audit.get('all_model_roles_verified') is not True:raise ValueError('native audit admission')
    views=[]
    for record in records:
        role=contract['roles'].get(record.get('role'))
        if role is None or record.get('checkpoint')!=role['checkpoint'] or record.get('request_hash')!=digest(record.get('request')) or not record.get('proofs') or not record.get('probabilities_sha256'):raise ValueError('role computation binding')
        views.append({'role':record['role'],'prompt':record['prompt'],'output':record['output'],'loss_mask':[role['training_eligible']]*len(record['output']),'training_eligible':role['training_eligible']})
    return views
