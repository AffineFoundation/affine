"""Code-owned controlled EOG broker routing; no job-selected factories.

Only a NEW isolated portable-v4 native source may deploy this module. Private
fixtures belong on the trusted controlled host, never in source/spec uploads.
"""
from pathlib import Path
from .native_eog_deployment import deployment

PRIVATE_TASKS=Path('/root/native-eog-common-v4/operator/private-tasks.json')

def create(spec):
    if spec.adapter!='native_eog_broker':raise ValueError('unapproved long-context native adapter')
    return deployment(spec,PRIVATE_TASKS)
