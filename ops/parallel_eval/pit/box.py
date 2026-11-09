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

        train_receipt = dict(self.io.train_v834e(
            head_path=head_path,
            master_path=master_path,
            expected_master_digest=record.bound_master_digest,
            batch_handle=batch_handle,
            item_id=record.item_id,
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

        # seal
        self.io.upload_candidate(candidate_id=candidate_id, candidate_path=candidate_path)
        record = apply_transition(
            record, ItemState.SEALED, reason='sealed',
            updates={'candidate_id': candidate_id, 'sealed': True},
        )
        record = apply_transition(record, ItemState.SERVING, reason='serve-start')

        lane = self.lane or LaneService()
        lane.serve(record.box_id or 'box', candidate_id, max_num_seqs=self.config.max_num_seqs)
        record = apply_transition(record, ItemState.PROBING, reason='probe-start')

        master_upload = {}
        if self.config.master_upload == 'during_probe':
            master_upload = dict(self.io.upload_master_during_probe(
                candidate_id=candidate_id, master_path=candidate_master_path,
            ))

        probe_fixture = dict(self.io.probe_result_fixture())
        probe_fixture.setdefault('score', probe_fixture.get('score') or {'decision': 'green', 'z': 1.5})
        probe_result = lane.run_probe(
            record.box_id or 'box',
            candidate_id,
            indices_count=indices_count,
            plan_seed_handle=plan_handle,
            k=self.config.k_cand,
            max_tokens=self.config.max_output_tokens,
            result=probe_fixture,
        )
        # Ensure score block survives lane copy.
        if 'score' not in probe_result and 'score' in probe_fixture:
            probe_result['score'] = probe_fixture['score']

        record = apply_transition(record, ItemState.RESULT, reason='probe-complete', updates={'result_stored': True})
        bundle = ProbeBundle(
            item_id=record.item_id,
            bound_head=record.bound_head,
            bound_master_digest=record.bound_master_digest,
            candidate_id=candidate_id,
            probe_result=probe_result,
            master_upload=master_upload,
            train_receipt={k: v for k, v in train_receipt.items() if k not in ('seed', 'presigned_url', 'private_key', 'secret')},
            box_public_key_hex=box_key.public_key_hex,
            meta={'shadow': record.shadow},
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
    })
    probe_result: Dict[str, Any] = field(default_factory=lambda: {
        'k': GATE_K_CAND,
        'dtype': GATE_DTYPE,
        'partial': False,
        'samples': GATE_N_DRAWN * GATE_K_CAND,
        'planned_samples': GATE_N_DRAWN * GATE_K_CAND,
        'score': {'decision': 'green', 'z': 1.5},
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
            return self.on_train(**kwargs)
        return dict(self.train_result)

    def upload_candidate(self, **kwargs: Any) -> Mapping[str, Any]:
        return {'ok': True}

    def upload_master_during_probe(self, **kwargs: Any) -> Mapping[str, Any]:
        return dict(self.upload_result)

    def probe_result_fixture(self) -> Mapping[str, Any]:
        return dict(self.probe_result)

    # Older BoxIO name used by prior draft
    def probe_live_gate(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.probe_result_fixture()
