"""Offline stub of subnet.environments for the probe_worker grade-supervisor test.
Task index behaviour (step() of the stub session): 1 -> hangs forever, 2 -> os._exit (worker dies),
3 -> raises a BaseException like math_verify.errors.TimeoutException, even -> positive, odd -> negative."""
import os, time
class TimeoutException(BaseException): pass
class EnvironmentSpec:
    def __init__(self, d): self.id = d['id']; self.source_hash = d['source_hash']
    @classmethod
    def from_dict(cls, d): return cls(d)
def _source_hash(spec): return spec.source_hash
class _S:
    def reset(self, index, seed):
        self.i = index; return dict(messages=[dict(role='user', content='task %d' % index)], task_hash='h%d' % index)
    def close_runtime(self): pass
    def close(self): pass
    def step(self, act):
        if self.i == 1 and os.environ.get('STUB_HANG', '1') == '1': time.sleep(10 ** 6)
        if self.i == 2 and os.environ.get('STUB_CRASH', '1') == '1': os._exit(9)
        if self.i == 3 and os.environ.get('STUB_BASEEXC', '1') == '1': raise TimeoutException('Timeout during comparison')
        return dict(done=True, classification='positive' if self.i % 2 == 0 else 'negative')
def create_session(spec): return _S()
