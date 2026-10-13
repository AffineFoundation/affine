"""Prospective host scheduling cache; no installation or production side effects.

Only records returned by original Coordinator.collect in this process are reused.
New/changed/restarted cases go through that original method. The warm cache
is an optimization of already durable snapshots, never authority for new data.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os

from subnet.backend_jobs import signed
from subnet.storage import canonical, sha
from subnet.complete_group_capture import _submission_get, SubmissionRejected
from subnet.complete_group_codec import limits
from subnet.complete_group_transport import paths
from subnet.mimo_release_policy import need


class CapturePass:
    def __init__(self, *, workers=8):
        need(type(workers) is int and 1 <= workers <= 16, 'bounded capture probe concurrency')
        self.workers = workers
        self.scope = None
        self.accepted = {}
        self.last_metrics = {}

    @staticmethod
    def _record(coordinator, manifest, miner, record):
        value = signed(record, coordinator.identity.id)
        need(value.get('epoch') == manifest['epoch'] and value.get('miner') == miner
             and value.get('manifest_sha256') == sha(canonical(manifest))
             and value.get('assurance') == 'structurally-eligible-unaudited'
             and value.get('fully_audited') is False,
             'original authenticated structural capture only')
        return value

    def __call__(self, lifecycle, coordinator, manifest, now):
        scope = (str(coordinator.root.absolute()), sha(canonical(manifest)))
        if scope != self.scope:
            self.scope = scope
            self.accepted.clear()
        miners = lifecycle.services.committed_miners(manifest)
        need(type(miners) is list and len(miners) <= len(manifest['capabilities'])
             and len(set(miners)) == len(miners) and set(miners) <= set(manifest['capabilities']),
             'complete registered commitment listing')
        miners = sorted(miners)
        bound = limits(manifest)
        counters = dict(listed=len(miners), reused=0, full_collect=0, rejected=0)

        def probe(miner):
            previous = self.accepted.get(miner)
            if previous is None:
                return False
            # Read and hash exact current bytes, not LIST/ETag/size claims. The
            # same original upload-window rule is preserved, including now.
            try:
                data, received = _submission_get(coordinator.bucket, paths(manifest, miner),
                                                bound.commitment_bytes)
            except SubmissionRejected:
                return False  # Original collect supplies the canonical rejection.
            return (manifest['start'] <= received < manifest['deadline'] and received <= now
                    and sha(data) == previous['commitment_sha256'])

        rejected = {}
        # Threads only read small commitments. Native resolution, ledger writes,
        # signing and capture publication remain in original sorted serial order.
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='commitment-probe') as pool:
            for miner, unchanged in zip(miners, pool.map(probe, miners), strict=True):
                if unchanged:
                    previous = self.accepted[miner]
                    fd = coordinator._lock()
                    try:
                        need(not (coordinator.root/'POPULATION.json').exists()
                             and not (coordinator.root/'DISPATCH.json').exists(),
                             'epoch population already sealed')
                        path = coordinator.root/'captured'/(miner+'.json')
                        if (path.is_file() and not path.is_symlink()
                                and path.stat().st_size <= 4*1024**2
                                and sha(path.read_bytes()) == previous['record_sha256']):
                            counters['reused'] += 1
                            continue
                    finally:
                        os.close(fd)
                try:
                    counters['full_collect'] += 1
                    # A changed/rejected observation invalidates the optimization
                    # marker. Restoring old bytes must earn a fresh original collect.
                    self.accepted.pop(miner, None)
                    result = coordinator.collect(miner, now=now)
                    value = self._record(coordinator, manifest, miner, result)
                    path = coordinator.root/'captured'/(miner+'.json')
                    raw = path.read_bytes()
                    need(json.loads(raw) == result, 'original capture durable local acknowledgement')
                    self.accepted[miner] = dict(commitment_sha256=value['commitment_sha256'],
                                               record_sha256=sha(raw))
                except SubmissionRejected as error:
                    rejected[miner] = str(error)[:1024]
                    counters['rejected'] += 1
        self.last_metrics = counters
        return rejected
