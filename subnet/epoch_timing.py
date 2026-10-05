"""Measured controller timings; no inferred training or chain completion."""
import time

def transition(active,phase,now=None):
    now=time.time() if now is None else now
    start=active.get('phase_started_at')
    if start is not None:
        active.setdefault('phase_timings',[]).append(dict(phase=active['phase'],started_at=start,completed_at=now,seconds=max(0,now-start)))
    active['phase']=phase;active['phase_started_at']=now

def completion(active,manifest,previous_steps,now=None):
    now=time.time() if now is None else now
    started=active.get('started_at',manifest.get('start'))
    timings=list(active.get('phase_timings',[]))
    if active.get('phase_started_at') is not None:
        timings.append(dict(phase=active['phase'],started_at=active['phase_started_at'],completed_at=now,seconds=max(0,now-active['phase_started_at'])))
    steps=active['next_steps']-previous_steps
    return dict(version='measured-epoch-controller-v1',epoch=manifest['epoch'],
                started_at=started,controller_completed_at=now,
                controller_seconds=now-started if started is not None else None,
                phase_timings=timings,training_updates=steps,
                nonempty_training_update=steps>0,
                inference_checkpoint=active['next_checkpoint']['id'],
                public_optimizer_steps=active.get('next_trainer_state',{}).get('optimizer_steps'),
                chain_weights_observed=False,
                full_hourly_epoch_confirmed=False)
