"""Surgical drained activation TRANSACTION SIMULATOR. Never calls systemctl/SSH.

A successful simulation is not live admission. Staging evidence cannot promote
itself to a production qualification. Only the isolated fake service image below
is accepted; production service adapters deliberately do not exist in this module.
"""
import copy
from .durable import Stage, atomic, digest, read


class StageServices:
    def __init__(self, stage):
        self.stage = Stage(stage)
        self.path = self.stage.root / 'SERVICE-IMAGE.json'

    def snapshot(self):
        return read(self.path)

    def save(self, value):
        atomic(self.path, value)

    def stop(self, name):
        value = self.snapshot()
        value['units'][name] = 'stopped'
        self.save(value)

    def start(self, name):
        value = self.snapshot()
        value['units'][name] = 'active'
        self.save(value)


class Activation:
    def __init__(self, stage):
        self.stage = Stage(stage)
        self.journal = self.stage.root / 'ACTIVATION-SIMULATION.json'

    @staticmethod
    def quiet(image):
        return not any(image.get(k) for k in ('training_job', 'recovery_required', 'gate_busy', 'unacked_apply'))

    def simulate(self, expected_epoch, expected_head, pe_config, dropin, fault=lambda phase: None):
        svc = StageServices(self.stage.root)
        with self.stage.locked():
            prior = svc.snapshot()
            if prior['current'] != dict(epoch=expected_epoch, head=expected_head):
                raise ValueError('epoch/head moved before drain')
            journal = dict(version='pe-activation-simulation-v1', live=False,
                           state='prepared', before=copy.deepcopy(prior), ready_for_live=False)
            atomic(self.journal, journal)
            try:
                svc.stop('lifecycle')
                svc.stop('gate')
                fault('drained')
                drained = svc.snapshot()
                if drained['current'] != prior['current'] or not self.quiet(drained):
                    raise ValueError('boundary changed or busy after drain')
                journal['state'] = 'drained'
                atomic(self.journal, journal)
                # Only the PE key + PE drop-in, never whole source/caps overwrite.
                configured = copy.deepcopy(drained)
                configured['flag']['parallel_eval'] = pe_config
                configured['dropins']['parallel_eval'] = dropin
                svc.save(configured)
                fault('configured')
                journal['state'] = 'configured'
                atomic(self.journal, journal)
                # Recheck after mutation, before restart. Training/epoch data are
                # NEVER restored from the old snapshot if they changed.
                observed = svc.snapshot()
                if observed['current'] != prior['current'] or not self.quiet(observed):
                    raise ValueError('boundary changed before restart')
                svc.start('gate')
                svc.start('lifecycle')
                fault('restarted')
                after = svc.snapshot()
                if any(after['units'][n] != 'active' for n in ('lifecycle', 'gate')):
                    raise ValueError('composed units failed verification')
                if {k: v for k, v in after['flag'].items() if k != 'parallel_eval'} != {k: v for k, v in prior['flag'].items() if k != 'parallel_eval'} or after['caps'] != prior['caps']:
                    raise ValueError('non-PE/64K configuration changed')
                journal.update(state='verified', after_sha256=digest(after))
                atomic(self.journal, journal)
                return journal
            except BaseException:
                # Surgical rollback of ONLY our configuration keys. Preserve a
                # concurrently changed CURRENT/job; never replay a whole image.
                value = svc.snapshot()
                for field, key in (('flag', 'parallel_eval'), ('dropins', 'parallel_eval')):
                    if key in prior[field]:
                        value[field][key] = prior[field][key]
                    else:
                        value[field].pop(key, None)
                svc.save(value)
                for n in ('gate', 'lifecycle'):
                    if prior['units'][n] == 'active':
                        svc.start(n)
                journal['state'] = 'rolled-back'
                atomic(self.journal, journal)
                raise

    def recover(self):
        """Crash recovery also restores only PE keys, not epoch/job/caps state."""
        with self.stage.locked():
            j = read(self.journal)
            if j['state'] in ('verified', 'rolled-back'):
                return j
            svc = StageServices(self.stage.root)
            value, prior = svc.snapshot(), j['before']
            for field, key in (('flag', 'parallel_eval'), ('dropins', 'parallel_eval')):
                if key in prior[field]:
                    value[field][key] = prior[field][key]
                else:
                    value[field].pop(key, None)
            svc.save(value)
            for n in ('gate', 'lifecycle'):
                if prior['units'][n] == 'active':
                    svc.start(n)
            j['state'] = 'rolled-back'
            atomic(self.journal, j)
            return j
