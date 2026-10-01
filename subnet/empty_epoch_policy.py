"""Prospective signed, nonpayable controlled-empty recovery window policy."""
VERSION = 'controlled-empty-epoch-recovery-v1'


def selected(config, round_number):
    rounds = config.get('controlled_empty_rounds', [])
    if not isinstance(rounds, list) or len(rounds) > 1 or any(type(r) is not int or not 0 <= r <= 100000 for r in rounds):
        raise ValueError('at most one explicitly designated empty round')
    if type(round_number) is not int or round_number < 0:
        raise ValueError('round number')
    if not rounds:
        return None
    if not config.get('epoch_prefix', '').startswith('nonpayable-') or config.get('payable_epochs', False):
        raise ValueError('controlled recovery is permanently nonpayable')
    if round_number not in rounds:
        return None
    return {'version': VERSION, 'purpose': 'empty-epoch-recovery', 'round': round_number,
            'miner_dispatch': False, 'payable': False, 'chain_transactions': False}


def validate(value):
    if not isinstance(value, dict) or set(value) != {'version', 'purpose', 'round', 'miner_dispatch', 'payable', 'chain_transactions'}:
        raise ValueError('exact signed operator test policy')
    if value['version'] != VERSION or value['purpose'] != 'empty-epoch-recovery' or type(value['round']) is not int or not 0 <= value['round'] <= 100000:
        raise ValueError('controlled empty policy identity')
    if any(value[name] is not False for name in ('miner_dispatch', 'payable', 'chain_transactions')):
        raise ValueError('nonpayable zero-dispatch policy')
    return value


def dispatch_allowed(manifest):
    value = manifest.get('operator_test_policy')
    if value is None:
        return True
    validate(value)
    if manifest.get('payable') is not False or not manifest.get('epoch', '').startswith('nonpayable-'):
        raise ValueError('signed controlled window scope')
    return False


def validate_empty_completion(manifest, result, reports):
    if dispatch_allowed(manifest):
        return
    if result.get('points') or result.get('weights') or any(report.get('accepted') for report in reports.values()):
        raise ValueError('controlled empty window unexpectedly contained credited data')
