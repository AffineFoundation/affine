"""One prospective batch-size setting; historical K/L contracts remain intact."""
MAX_SAMPLES_PER_BATCH = 128


def configured_quotas(config):
    if 'samples_per_batch' not in config:
        return config.get('K', 1), config.get('L', 1)
    count = config['samples_per_batch']
    if type(count) is not int or count < 4 or count > MAX_SAMPLES_PER_BATCH or count % 2:
        raise ValueError('samples_per_batch must be an even integer from 4 to 128')
    quota = count // 2
    for name in ('K', 'L'):
        if name in config and (type(config[name]) is not int or config[name] != quota):
            raise ValueError('samples_per_batch conflicts with explicit class quotas')
    return quota, quota


def normalize_config(config):
    """Derive legacy wire fields without mutating caller or signed input bytes."""
    result = dict(config)
    if 'samples_per_batch' in config:
        result['K'], result['L'] = configured_quotas(config)
    return result
