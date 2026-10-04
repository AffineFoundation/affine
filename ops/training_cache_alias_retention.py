"""Retire an obsolete cache alias while preserving its completed training export."""
import re
import secrets
import stat
from pathlib import Path

from ops import training_retention as training


def remove_training_cache_alias(plan):
    workspace, job, report = training.completed_job(plan)
    checkpoint = plan['checkpoint']
    protected = plan['protected_checkpoints']
    active = plan['active_checkpoints']
    if (not isinstance(protected, list) or not 1 <= len(protected) <= 32
            or not isinstance(active, list) or len(active) > 512
            or any(not isinstance(c, str) or not re.fullmatch('[0-9a-f]{64}', c)
                   for c in protected + active)
            or checkpoint in protected + active):
        raise ValueError('current or active checkpoint alias protected')
    if plan.get('archive_verified') is not True or plan.get('archive_authenticated') is not True:
        raise ValueError('authenticated complete archive readback required')
    source = Path(plan['source_directory'])
    alias = Path(plan['directory'])
    covered = job.get('training_policy') == 'bf16-full-adamw-covered-fixed-reference-v3'
    name = 'checkpoint-covered-final' if covered else 'checkpoint-step-' + str(job['steps'])
    files = plan['files']
    output = report.get('new_checkpoint', {})
    if (not isinstance(checkpoint, str) or not re.fullmatch('[0-9a-f]{64}', checkpoint)
            or source != workspace/'jobs'/job['job_id']/name
            or alias != workspace/'checkpoints'/checkpoint
            or output.get('path') != str(source) or output.get('id') != checkpoint
            or not isinstance(files, dict) or not 1 <= len(files) <= 32
            or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files)
            or (covered and report.get('training', {}).get('training_policy') != job['training_policy'])):
        raise ValueError('exact original final export and default cache alias required')
    for n, value in files.items():
        if (not isinstance(n, str) or not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*', n)
                or Path(n).suffix not in {'.json', '.safetensors', '.txt', '.model', '.jinja', '.tiktoken'}
                or not isinstance(value, dict) or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', ''))
                or type(value.get('size')) is not int
                or not 0 < value['size'] <= (32 if n.endswith('.safetensors') else 5)*1024**3):
            raise ValueError('bounded archived checkpoint inventory required')
    hashes = {n: value['sha256'] for n, value in files.items()}
    if training.digest(hashes) != checkpoint or output.get('files') != hashes:
        raise ValueError('original report and archive checkpoint bytes required')
    for root in (source, alias):
        if (not root.is_absolute() or root.resolve() != root or root.is_symlink()
                or not root.is_dir() or {p.name for p in root.iterdir()} != set(files)):
            raise ValueError('canonical exact source and alias membership required')
    before = {}
    for n, value in files.items():
        s, a = (source/n).lstat(), (alias/n).lstat()
        if (not stat.S_ISREG(s.st_mode) or not stat.S_ISREG(a.st_mode)
                or s.st_nlink != 2 or a.st_nlink != 2
                or (s.st_dev, s.st_ino) != (a.st_dev, a.st_ino)
                or s.st_size != value['size'] or training.hash_file(source/n) != value['sha256']):
            raise ValueError('exact two known immutable hardlinks required')
        before[n] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    training.unreferenced([root/n for root in (source, alias) for n in files])
    training.completed_job(plan)
    for root in (source, alias):
        if {p.name for p in root.iterdir()} != set(files):
            raise ValueError('checkpoint membership changed')
        for n, previous in before.items():
            s = (root/n).lstat()
            if s.st_nlink != 2 or (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) != previous:
                raise ValueError('checkpoint links changed during verification')
    if training.gpu_processes():
        raise ValueError('training alias retention requires idle GPU')
    retired = alias.with_name(checkpoint + '.retired-alias-' + secrets.token_hex(8))
    alias.rename(retired)
    for n in files:
        (retired/n).unlink()
    retired.rmdir()
    return dict(alias_removed=True, physical_bytes_freed=0, source_preserved=True,
                original_jobs_and_reports_preserved=True, checkpoint=checkpoint)
