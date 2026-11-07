"""Actual file-lock ownership fence for isolated legacy-gate/PE composition.

Not installed in the live gate. Every legacy mutation (including late apply)
MUST enter guard('legacy'); checking only whether an epoch was claimed is racy.
A mode transfer requires an authenticated external drain/current proof. Missing
ownership configuration fails closed; it never silently grants legacy access.
"""
import contextlib
import fcntl
from .durable import Stage, atomic, digest, read


class GateOwnership:
    def __init__(self, root, verifier):
        self.stage, self.verify = Stage(root), verifier
        if not callable(verifier):
            raise TypeError('explicit drain-proof verifier required')
        self.path = self.stage.root/'GATE-OWNERSHIP.json'
        self.lock_path = self.stage.root/'.gate-ownership.lock'

    @contextlib.contextmanager
    def _locked(self):
        if self.lock_path.is_symlink():
            raise ValueError('symlink ownership lock')
        with self.lock_path.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            yield

    def initialize(self, owner, epoch, head):
        if owner not in {'legacy','pe'} or type(epoch) is not int or epoch < 0:
            raise ValueError('explicit gate owner/epoch')
        with self._locked():
            if self.path.exists():
                raise ValueError('ownership already initialized')
            atomic(self.path,dict(version='pe-gate-ownership-v1',owner=owner,
                                  generation=0,boundary_epoch=epoch,boundary_head=head))

    @contextlib.contextmanager
    def guard(self, owner, *, generation=None):
        """Hold through the short mutation/commit, not the remote GPU probe."""
        with self._locked():
            value=read(self.path)
            if owner != value['owner'] or (generation is not None and generation != value['generation']):
                raise ValueError('gate mutation fenced by owner/generation')
            yield value

    def transfer(self, expected, owner, epoch, head, proof):
        if expected not in {'legacy','pe'} or owner not in {'legacy','pe'} or expected == owner:
            raise ValueError('explicit different gate owners')
        if type(epoch) is not int or epoch < 0:
            raise ValueError('boundary epoch')
        before=read(self.path)
        binding=dict(before=before,owner=owner,epoch=epoch,head=head,drained_required=True)
        if not self.verify(binding,proof):
            raise ValueError('authenticated real drain/current proof required')
        with self._locked():
            if read(self.path) != before or before['owner'] != expected:
                raise ValueError('ownership moved during proof verification')
            value=dict(version='pe-gate-ownership-v1',owner=owner,generation=before['generation']+1,
                       boundary_epoch=epoch,boundary_head=head,drain_proof_sha256=digest(proof))
            atomic(self.path,value)
            return value
