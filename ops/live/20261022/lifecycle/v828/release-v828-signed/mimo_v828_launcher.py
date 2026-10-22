#!/usr/bin/env python3
"""MiMo v8.2.8 host launcher = the signed v8.2.7 composition (uid launcher + capture launcher v3 + v8.2.4 overrides, all
byte-identical and pinned) + three ROOT-admitted, host-process-only changes (authorized by Jacob 2026-10-21):

 1. Audit population OFF the critical path. The original Lifecycle._audit_population (unchanged, pinned) runs once per
    epoch per process in a background thread; the tick goes straight on to the trainer job. FINALIZED still waits for it
    to have completed (the v9 scorer and dashboard read epochs/E/AUDIT-POPULATION.json of FINALIZED epochs), so the file,
    its bytes and its R2 copy are exactly as before - only its position on the timeline changes. Later ticks no longer
    re-run the ~110 s re-authentication on every pass (it ran once, in the background, in this process).
 2. Audit scheduling disabled. ops.mimo_lifecycle.archive_scoped_audit_scheduler returns a scheduler that admits and
    schedules nothing, so no audit jobs are queued for the verify workers. Lifecycle config (audit_scheduler.enabled=true,
    which the public-mode check requires) and ops.mimo_lifecycle.run are byte-identical; the requirement is satisfied
    by the disabled scheduler. Miner upload format/proofs unchanged (still captured, just not checked).
 3. Native grading parallelism: NativeGradingPool workers (and its in-flight bound) 8 -> GRADING_WORKERS. Ordered map
    unchanged, so every document gets the original native grade in the original order (identical DISPATCH).
Trainer, trainer job contents, weight writer, v9 scorer, policy, closure, chain: UNCHANGED. No chain calls."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import threading
import time

P = Path(__file__).resolve().parent
VERSION = 'affine-host-v828-audit-off-path-launcher-v1'
SCOPE = 'host-audit-off-critical-path-only'
# Same spawn-child import hygiene as the uid/capture launchers (native grading children re-import __main__).
_spec = importlib.util.spec_from_file_location('affine_v828_host_paths', P/'mimo_host_path_launcher.py')
_paths = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _paths
_spec.loader.exec_module(_paths)
if __name__ in ('__main__', '__mp_main__'):
    _paths.install()

PINNED = ('Lifecycle._audit_population', 'Lifecycle._training', 'Lifecycle.tick',
          'ops.mimo_lifecycle.archive_scoped_audit_scheduler',
          'subnet.mimo_native_pool.NativeGradingPool.__init__', 'subnet.mimo_native_pool.NativeGradingPool.__call__')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def msha(method):
    return hashlib.sha256(inspect.getsource(method).encode()).hexdigest()


def pinned(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact pinned host reference')
    path = Path(ref['path'])
    need(path.is_absolute() and not path.is_symlink() and path.is_file(), 'ordinary absolute host file')
    need(hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], 'host file digest differs')
    return path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked(config):
    need(set(config) == {'version', 'uid_entry', 'uid_operator', 'host_files', 'original_methods',
                         'grading_workers', 'admission'}, 'exact v8.2.8 host configuration')
    need(config['version'] == VERSION, 'v8.2.8 launcher version')
    workers = config['grading_workers']
    need(type(workers) is int and 8 <= workers <= 32, 'bounded native grading workers (original pool allows 1..32)')
    need(set(config['host_files']) == {'mimo_v828_launcher.py', 'mimo_host_path_launcher.py'}, 'exact v8.2.8 host files')
    for name, checksum in config['host_files'].items():
        pinned(dict(path=str(P/name), sha256=checksum))
    uid_path, uid_operator = pinned(config['uid_entry']), pinned(config['uid_operator'])
    uid = load(uid_path, 'affine_v827_uid_capacity_launcher')
    _paths.install()
    parent, operator, module, Lifecycle, Coordinator, methods = uid.checked(_paths.private_json(uid_operator))
    lifecycle_config = parent.paths.private_json(operator['base']['lifecycle_config']['path'])
    authority = lifecycle_config['release_policy']['signer']
    from subnet.backend_jobs import signed
    from subnet.storage import canonical, sha
    proposal = {k: v for k, v in config.items() if k != 'admission'}
    need(signed(config['admission'], authority) == dict(version=VERSION, configuration_sha256=sha(canonical(proposal)),
         source_unchanged=True, scope=SCOPE, grading_workers=workers), 'exact ROOT v8.2.8 admission')
    # The reviewed original state this override is written against.
    need(lifecycle_config.get('native_grading') == {'workers': 8}, 'reviewed lifecycle native_grading.workers=8')
    need(lifecycle_config.get('audit_scheduler', {}).get('enabled') is True, 'reviewed lifecycle audit_scheduler.enabled=true')
    import subnet.mimo_native_pool as NP
    found = {'Lifecycle._audit_population': Lifecycle._audit_population, 'Lifecycle._training': Lifecycle._training,
             'ops.mimo_lifecycle.archive_scoped_audit_scheduler': module.archive_scoped_audit_scheduler,
             'subnet.mimo_native_pool.NativeGradingPool.__init__': NP.NativeGradingPool.__init__,
             'subnet.mimo_native_pool.NativeGradingPool.__call__': NP.NativeGradingPool.__call__}
    need(set(config['original_methods']) == set(PINNED), 'exact reviewed v8.2.8 original method set')
    for name, method in found.items():
        need(msha(method) == config['original_methods'][name], 'original method differs: ' + name)
    import subnet.mimo_lifecycle as ML
    need(msha(ML.Lifecycle.tick) == config['original_methods']['Lifecycle.tick'], 'original Lifecycle.tick differs')
    need('self._audit_population(directory,coordinator,opening,dispatch,plan,current)\n            try:\n'
         '                result=self._training(directory,opening,dispatch,current,now)' in inspect.getsource(ML.Lifecycle.tick),
         'tick calls the audit population right before training')
    return uid, parent, operator, module, Lifecycle, Coordinator, methods, workers


class _AuditPopulationJob:
    def __init__(self):
        self.thread = None; self.done = False; self.error = None; self.seconds = None


class DisabledAuditScheduler:
    """v8.2.8: audits are not on the training or weight path. Nothing is admitted, scheduled or queued."""
    def __init__(self, **kwargs):
        pass

    def admit(self, population):
        return None

    def tick(self, now=None):
        return None


def install_v828(module, Lifecycle, grading_workers):
    import subnet.mimo_native_pool as NP
    original_population = Lifecycle._audit_population
    inner_training = Lifecycle._training          # the v8.2.4 (finalize clock fix) replacement already installed

    def _audit_population(self, directory, coordinator, opening, dispatch, plan, current):
        jobs = self.__dict__.setdefault('_v828_audit_jobs', {})
        key = str(directory); job = jobs.get(key)
        if job is not None and (job.done or job.thread.is_alive()):
            return None
        job = _AuditPopulationJob(); jobs[key] = job
        epoch = plan['manifest']['epoch']
        def run():
            began = time.time()
            try:
                # The thread re-reads the exact durable envelopes the tick holds (no shared mutable objects).
                o = json.loads((coordinator.root/'OPENING.json').read_bytes())
                d = json.loads((coordinator.root/'DISPATCH.json').read_bytes())
                need(o == opening and d == dispatch, 'durable opening/dispatch equal the tick objects')
                original_population(self, directory, coordinator, o, d, plan, current)
                job.done = True
            except BaseException as error:
                job.error = type(error).__name__
            job.seconds = round(time.time() - began, 1)
            print(json.dumps(dict(component='audit-population-offpath', epoch=epoch,
                                  status='done' if job.done else 'error', error_type=job.error, seconds=job.seconds)), flush=True)
        job.thread = threading.Thread(target=run, name='v828-audit-population', daemon=True)
        job.thread.start()
        print(json.dumps(dict(component='audit-population-offpath', epoch=epoch, status='started')), flush=True)
        return None
    _audit_population.__qualname__, _audit_population.__module__ = original_population.__qualname__, original_population.__module__

    def _training(self, directory, opening, dispatch, current, now):
        result = inner_training(self, directory, opening, dispatch, current, now)
        if result is None:
            return None
        # FINALIZED (and so the v9 scorer) never sees an epoch without its completed original audit population.
        job = self.__dict__.get('_v828_audit_jobs', {}).get(str(directory))
        if job is not None and job.thread.is_alive():
            job.thread.join()
        need(job is not None and job.done, 'original audit population not yet complete; finalize deferred')
        return result
    _training.__qualname__, _training.__module__ = inner_training.__qualname__, inner_training.__module__

    def archive_scoped_audit_scheduler(base, auditable_archives, *, report_name='UNAUDITABLE-BY-DESIGN.json'):
        print(json.dumps(dict(component='audit-scheduler', status='disabled-v828')), flush=True)
        return DisabledAuditScheduler

    original_init = NP.NativeGradingPool.__init__
    def __init__(self, tokenizer_directory, checkpoint_files, template_sha256, task_assets, *, workers=8):
        need(workers == 8, 'reviewed native_grading configuration (workers=8)')
        original_init(self, tokenizer_directory, checkpoint_files, template_sha256, task_assets, workers=grading_workers)
    __init__.__qualname__, __init__.__module__ = original_init.__qualname__, original_init.__module__

    Lifecycle._audit_population = _audit_population
    Lifecycle._training = _training
    module.archive_scoped_audit_scheduler = archive_scoped_audit_scheduler
    NP.NativeGradingPool.__init__ = __init__
    return dict(v828=True, audit_population='background-before-finalize', audit_scheduler='disabled', grading_workers=[8, grading_workers])


def compose(uid, parent, operator, Lifecycle, Coordinator, methods, module, workers):
    """Exactly the v8.2.7 uid launcher main() install sequence, then the v8.2.8 layer last."""
    parent.install_capture(Lifecycle, operator['workers'])
    Lifecycle._plan, Coordinator.open = methods
    helper = sys.modules['affine_host_uid_capacity']
    v824 = helper.v824_install(Lifecycle)
    return v824, install_v828(module, Lifecycle, workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-config', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = _paths.private_json(args.operator_config)
    uid, parent, operator, module, Lifecycle, Coordinator, methods, workers = checked(config)
    if args.check:
        helper = sys.modules['affine_host_uid_capacity']
        from subnet.mimo_training_worker import validate_result
        ok = (helper.method_sha(validate_result) == helper.V824_PINS['mimo_training_worker.validate_result'] and
              helper.method_sha(Lifecycle._training) == helper.V824_PINS['Lifecycle._training'] and
              helper.method_sha(Lifecycle._current) == helper.V824_PINS['Lifecycle._current'])
        # Dry composition in this check process only (nothing started): proves the full install order succeeds.
        v824, v828 = compose(uid, parent, operator, Lifecycle, Coordinator, methods, module, workers)
        print(json.dumps(dict(version=VERSION, capacity=1000, original_source_guard_passed=True, operator_admission_passed=True,
                              v824_override_pins_passed=ok, v828_dry_install=v828, production_services_started=False)))
        return
    v824, v828 = compose(uid, parent, operator, Lifecycle, Coordinator, methods, module, workers)
    print(json.dumps(dict(component='host-v824', **v824)), flush=True)
    print(json.dumps(dict(component='host-v828', **v828)), flush=True)
    sys.argv = [operator['base']['entry']['path']] + operator['base']['argv']
    module.main()


if __name__ == '__main__':
    main()
