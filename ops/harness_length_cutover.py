"""Admit a prospective output-budget change without relaxing other contracts."""
import copy


def validate_cutover(config, previous, grant, *, previous_config_sha256, new_config_sha256):
    fields = {'version', 'previous_config_sha256', 'new_config_sha256',
              'minimum_round', 'old_max_output_tokens', 'new_max_output_tokens', 'env_id'}
    if set(grant) != fields or grant['version'] != 'signed-harness-output-budget-v1':
        raise ValueError('exact signed output-budget grant')
    if (grant['previous_config_sha256'] != previous_config_sha256 or
            grant['new_config_sha256'] != new_config_sha256):
        raise ValueError('exact predecessor and successor configs')
    if type(grant['minimum_round']) is not int or grant['minimum_round'] < 0:
        raise ValueError('prospective epoch boundary')
    before, after = grant['old_max_output_tokens'], grant['new_max_output_tokens']
    if type(before) is not int or type(after) is not int or not 1 <= before < after <= 2048:
        raise ValueError('qualified output budget')
    normalized = copy.deepcopy(config)
    changed = 0
    for row, old in zip(normalized['environments'], previous['environments']):
        if old['spec']['id'] == grant['env_id']:
            if (old['harness']['version'] != 'text-tools-long-v2' or
                    old['harness']['max_output_tokens'] != before or
                    row['harness']['max_output_tokens'] != after or
                    row['spec']['max_output_tokens'] < after):
                raise ValueError('supported harness and environment budget')
            row['harness']['max_output_tokens'] = before
            changed += 1
    if changed != 1 or normalized != previous:
        raise ValueError('output budget is the only configuration change')
    return grant
