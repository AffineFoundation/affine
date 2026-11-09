"""PIT box runner (BRIEF §1.3) — provision → train → seal → serve → probe → result.

Fail-closed on fp32 expected-master mismatch. Never passes
``--fp32-bootstrap-override``. M(C) upload during probe when configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from . import GATE_K_CAND, GATE_MAX_TOKENS, GATE_N_DRAWN
from .config import GATE_DTYPE, PitConfig
from .pe_lane import LaneService
from .signer import AdmissionCert, EphemeralBoxKey, generate_ephemeral_box_key, verify_admission_cert
from .state import ItemRecord, ItemState, TransitionError, apply_transition
from .training_report import (
    TrainingReportError,
    copy_training_report_to_host,
    extract_report_summary,
    gpu_type_of,
)


class Fp32MasterError(RuntimeError):
    def __init__(self, reason: str, **detail: Any):
        self.reason = reason
        self.detail = detail
        super().__init__(reason)


class BoxIO(Protocol):
    def download_head(self, head: str) -> str: ...
    def download_master(self, head: str, expected_digest: str) -> str: ...
    def train_v834e(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def upload_candidate(self, *, candidate_id: str, candidate_path: str) -> Mapping[str, Any]: ...
    def upload_master_during_probe(self, *, candidate_id: str, master_path: str) -> Mapping[str, Any]: ...
    def probe_result_fixture(self) -> Mapping[str, Any]: ...


@dataclass
class ProbeBundle:
    item_id: str
    bound_head: str
    bound_master_digest: str
    candidate_id: str
    probe_result: Dict[str, Any]
    master_upload: Dict[str, Any]
    train_receipt: Dict[str, Any]
    box_public_key_hex: str
    training_report: Dict[str, Any] = field(default_factory=dict)
    train_gpu: Optional[str] = None
    eval_gpu: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoxRunner:
    config: PitConfig
    io: BoxIO
    lane: Optional[LaneService] = None
    root_pubkey_hex: Optional[str] = None

    def start_box_identity(self) -> EphemeralBoxKey:
        return generate_ephemeral_box_key()

    def admit(self, box_key: EphemeralBoxKey, cert: AdmissionCert, *, now: Optional[float] = None) -> Dict[str, Any]:
        if not self.root_pubkey_hex:
            raise ValueError('root_pubkey_hex required')
        return verify_admission_cert(
            cert, expected_root_pubkey_hex=self.root_pubkey_hex, now=now,
            pubkey=box_key.public_key_hex,
        )

    def run(
        self,
        record: ItemRecord,
        *,
        batch_handle: str,
        plan_handle: str,
        box_key: Optional[EphemeralBoxKey] = None,
        admission_cert: Optional[AdmissionCert] = None,
        now: Optional[float] = None,
        indices_count: int = GATE_N_DRAWN,
        indices: Optional[Any] = None,
        train_gpu: Optional[str] = None,
        eval_gpu: Optional[str] = None,
    ) -> tuple[ItemRecord, ProbeBundle]:
        if record.state is not ItemState.PROVISIONING:
            raise TransitionError('box.run requires PROVISIONING, got %s' % record.state.value)
        if not record.bound_head or not record.bound_master_digest:
            raise TransitionError('missing binding')
        if not batch_handle or 'presign' in batch_handle.lower() or batch_handle.startswith('secret:'):
            raise ValueError('batch_handle must be opaque non-secret')
        if 'seed=' in (plan_handle or '').lower():
            raise ValueError('plan_handle must not embed seeds')

        box_key = box_key or self.start_box_identity()
        if admission_cert is not None:
            self.admit(box_key, admission_cert, now=now)

        head_path = self.io.download_head(record.bound_head)
        try:
            master_path = self.io.download_master(record.bound_head, record.bound_master_digest)
        except Exception as exc:
            msg = str(exc)
            if 'expected-master' in msg or 'Fp32MasterError' in type(exc).__name__:
                raise Fp32MasterError('expected-master-mismatch', detail=msg[:300]) from exc
            raise

        record = apply_transition(record, ItemState.TRAINING, reason='provisioned')
        train_gpu = gpu_type_of(train_gpu) or gpu_type_of((record.meta or {}).get('train_gpu'))
        pod = (record.meta or {}).get('pod')
        if pod is None and (self.lane and (self.lane.require_host or self.lane.host_run_probe is not None)):
            raise TransitionError('pod handle required for host box train/probe')

        train_receipt = dict(self.io.train_v834e(
            head_path=head_path,
            master_path=master_path,
            expected_master_digest=record.bound_master_digest,
            batch_handle=batch_handle,
            item_id=record.item_id,
            hotkey=record.hotkey_prefix or None,
            train_gpu=train_gpu,
            pod=pod,
            epoch=int(record.epoch),
            epoch_dir=(record.meta or {}).get('epoch_dir'),
        ))
        if train_receipt.get('used_fp32_bootstrap_override'):
            raise Fp32MasterError('bootstrap-override-refused-by-pit')
        candidate_id = train_receipt.get('candidate_id')
        candidate_path = train_receipt.get('candidate_path')
        candidate_master_path = train_receipt.get('candidate_master_path')
        if not isinstance(candidate_id, str) or len(candidate_id) != 64:
            raise ValueError('trainer must return candidate_id hex64')
        if not candidate_path or not candidate_master_path:
            raise ValueError('trainer must return candidate paths')
        train_gpu = gpu_type_of(train_receipt.get('train_gpu')) or train_gpu

        # Copy training report to host BEFORE seal/release (Jacob 2026-11-09).
        report_doc = train_receipt.get('training_report') or train_receipt.get('report_doc')
        report_path = train_receipt.get('report') if isinstance(train_receipt.get('report'), str) else None
        try:
            training_report = copy_training_report_to_host(
                report_path=report_path,
                item_id=record.item_id,
                epoch=int(record.epoch),
                candidate_id=candidate_id,
                report_doc=report_doc if isinstance(report_doc, Mapping) else None,
            )
        except TrainingReportError:
            # Scripted/synthetic trainers may embed a minimal summary directly.
            if isinstance(report_doc, Mapping):
                training_report = extract_report_summary(report_doc)
            elif isinstance(train_receipt.get('training_report_summary'), Mapping):
                training_report = dict(train_receipt['training_report_summary'])
            else:
                raise

        # Keep ~18.8 GB candidate on the box until GREEN. Only upload on
        # green_fresh apply (presigned PUT); RED releases without transfer.
        if not train_receipt.get('defer_candidate_upload'):
            self.io.upload_candidate(candidate_id=candidate_id, candidate_path=candidate_path)
        record = apply_transition(
            record, ItemState.SEALED, reason='sealed',
            updates={'candidate_id': candidate_id, 'sealed': True},
        )
        record = apply_transition(record, ItemState.SERVING, reason='serve-start')

        lane = self.lane or LaneService()
        # Live pe_lane.serve/run_probe need the pod handle (object/dict), not a box-id string.
        pod_handle = pod or record.box_id or 'box'
        if lane.require_host or lane.host_boot is not None:
            lane.boot(pod_handle)
        lane.serve(pod_handle, candidate_id, max_num_seqs=self.config.max_num_seqs)
        record = apply_transition(record, ItemState.PROBING, reason='probe-start')

        master_upload = {}
        if self.config.master_upload == 'during_probe':
            master_upload = dict(self.io.upload_master_during_probe(
                candidate_id=candidate_id, master_path=candidate_master_path,
            ))

        idx_list = list(indices) if indices is not None else None
        # Live / require_host lanes need the secret draw's task indices. Never
        # fabricate them here — host pe_round_seed.indices_for_epoch supplies them.
        if lane.require_host or lane.host_run_probe is not None:
            if idx_list is None:
                raise TransitionError('indices list required for host run_probe')
            probe_result = lane.run_probe(
                pod_handle,
                candidate_id,
                indices=idx_list,
                plan_seed=plan_handle,
                k=self.config.k_cand,
                max_tokens=self.config.max_output_tokens,
            )
        else:
            probe_fixture = dict(self.io.probe_result_fixture())
            probe_result = lane.run_probe(
                pod_handle,
                candidate_id,
                indices=idx_list,
                indices_count=indices_count if idx_list is None else None,
                plan_seed_handle=plan_handle,
                k=self.config.k_cand,
                max_tokens=self.config.max_output_tokens,
                result=probe_fixture,
            )
        eval_gpu = (
            gpu_type_of(eval_gpu)
            or gpu_type_of(probe_result.get('eval_gpu') or probe_result.get('gpu'))
            or gpu_type_of((record.meta or {}).get('eval_gpu'))
            or train_gpu
        )
        probe_result = dict(probe_result)
        probe_result['train_gpu'] = train_gpu
        probe_result['eval_gpu'] = eval_gpu

        record = apply_transition(record, ItemState.RESULT, reason='probe-complete', updates={'result_stored': True})
        safe_receipt = {
            k: v for k, v in train_receipt.items()
            if k not in ('seed', 'presigned_url', 'private_key', 'secret', 'report_doc', 'training_report')
        }
        safe_receipt['train_gpu'] = train_gpu
        safe_receipt['training_report_summary'] = dict(training_report)
        bundle = ProbeBundle(
            item_id=record.item_id,
            bound_head=record.bound_head,
            bound_master_digest=record.bound_master_digest,
            candidate_id=candidate_id,
            probe_result=probe_result,
            master_upload=master_upload,
            train_receipt=safe_receipt,
            box_public_key_hex=box_key.public_key_hex,
            training_report=dict(training_report),
            train_gpu=train_gpu,
            eval_gpu=eval_gpu,
            meta={'shadow': record.shadow, 'train_gpu': train_gpu, 'eval_gpu': eval_gpu},
        )
        return record, bundle


@dataclass
class ScriptedBoxIO:
    head_path: str = '/tmp/pit-fixture/head'
    master_path: str = '/tmp/pit-fixture/master'
    train_result: Dict[str, Any] = field(default_factory=lambda: {
        'candidate_id': 'c' * 64,
        'candidate_path': '/tmp/pit-fixture/cand',
        'candidate_master_path': '/tmp/pit-fixture/cand-master',
        'used_fp32_bootstrap_override': False,
        'train_gpu': 'B300',
        'training_report': {
            'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
            'batching': {'tasks': 125, 'rollouts': 1000},
            'tokens': {'prompt_tokens': 100, 'output_tokens': 200, 'loss_tokens': 150},
            'update_size': 1.0,
            'gradient_norm': 1.0,
            'engine': 'packed',
        },
    })
    probe_result: Dict[str, Any] = field(default_factory=lambda: {
        'k': GATE_K_CAND,
        'dtype': GATE_DTYPE,
        'partial': False,
        'samples': GATE_N_DRAWN * GATE_K_CAND,
        'planned_samples': GATE_N_DRAWN * GATE_K_CAND,
        'score': {'decision': 'green', 'z': 1.5},
        'eval_gpu': 'B300',
    })
    upload_result: Dict[str, Any] = field(default_factory=lambda: {'ok': True, 'receipt': 'mc-upload-fixture'})
    fail_master_digest: bool = False
    on_train: Optional[Callable[..., Mapping[str, Any]]] = None

    def download_head(self, head: str) -> str:
        return self.head_path

    def download_master(self, head: str, expected_digest: str) -> str:
        if self.fail_master_digest:
            raise Fp32MasterError('expected-master-mismatch', expected=expected_digest)
        return self.master_path

    def train_v834e(self, **kwargs: Any) -> Mapping[str, Any]:
        if kwargs.get('fp32_bootstrap_override') or kwargs.get('bootstrap_override'):
            raise Fp32MasterError('bootstrap-override-refused-by-pit')
        if self.on_train:
            out = dict(self.on_train(**kwargs))
        else:
            out = dict(self.train_result)
        out.setdefault('train_gpu', kwargs.get('train_gpu') or 'B300')
        out.setdefault('training_report', {
            'source_file_hashes': {'trainer': 'a' * 64, 'batch': 'b' * 64},
            'batching': {'tasks': 125, 'rollouts': 1000},
            'tokens': {'prompt_tokens': 100, 'output_tokens': 200, 'loss_tokens': 150},
            'update_size': 1.0,
            'gradient_norm': 1.0,
            'engine': 'packed',
        })
        # Scripted path ignores pod; real HostBoxIO requires it.
        out.setdefault('pod_id', (kwargs.get('pod') or {}).get('id') if isinstance(kwargs.get('pod'), Mapping) else None)
        return out

    def upload_candidate(self, **kwargs: Any) -> Mapping[str, Any]:
        return {'ok': True}

    def upload_master_during_probe(self, **kwargs: Any) -> Mapping[str, Any]:
        return dict(self.upload_result)

    def probe_result_fixture(self) -> Mapping[str, Any]:
        return dict(self.probe_result)

    # Older BoxIO name used by prior draft
    def probe_live_gate(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.probe_result_fixture()
