import functools
from typing import Callable

import requests

import affine as af

class RetryNeeded(ValueError):
    pass

def retry(fn: Callable | int = 5, retries: int | None = None) -> Callable:
    if retries is None:
        if not isinstance(fn, int):
            raise ValueError("retry() has to be closed when used as a decorator")
        return functools.partial(retry, retries=fn)

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                af.logger.trace(f'Error encountered: {e} - Retry {i}/{retries}')
                if i == retries - 1:
                    raise

    return _wrapped


@retry()
def fallback_models(retries: int = 3, min_completion_cost: float = 0.0, min_context: int = 65536,
                    owners: tuple = ('chutesai',), max_completion_cost: float = 1e9):
    models = requests.get('https://llm.chutes.ai/v1/models')
    models.raise_for_status()
    models = models.json()['data']
    models = [mod["id"] for mod in models  #
              if max_completion_cost >= mod['pricing']['completion'] >= min_completion_cost  #
              and mod['context_length'] > min_context  #
              and mod["id"].split('/')[-1] in owners]
    if not models:
        raise RetryNeeded
    return models
