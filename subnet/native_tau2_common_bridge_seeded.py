"""Prospective seed-bound heldout identity; preserves the historical v1 bridge.

This contract validates an authenticated challenge and identifies a dataset. It
performs no model computation, native replay, or retrospective source upgrade.
"""
from .native_tau2_common_bridge import heldout_contract as historical_contract
from .native_tau2_common_search_contract import validate_epoch, digest

VERSION = 'native-tau2-common-fixed-auxiliary-heldout16-seeded-v2'


def heldout_contract(epoch, authority, fixed_user, public_tasks):
    manifest = validate_epoch(epoch, authority, fixed_user)
    historical = historical_contract(epoch, authority, fixed_user, public_tasks)
    body = {k: v for k, v in historical.items() if k != 'dataset_id'}
    body['version'] = VERSION
    geometry = dict(body['agent_geometry_and_policy'])
    agent = manifest['roles']['agent']
    geometry['seed_start'] = agent['seed_start']
    geometry['seed_policy'] = agent['seed_policy']
    body['agent_geometry_and_policy'] = geometry
    return {**body, 'dataset_id': digest(body)}
