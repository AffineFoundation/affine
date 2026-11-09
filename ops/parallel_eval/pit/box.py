"""PIT box runner — train then probe on one rented GPU box.

The box:

1. Receives a bound item (head id + expected fp32 master digest + batch handle)
2. Downloads the head bf16 checkpoint and the fp32 master
3. Fail-closed checks the master digest (never ``--fp32-bootstrap-override``)
4. Runs trainer-v834e on the miner's batch
5. Runs the live-gate probe on the resulting candidate (same box)
6. Uploads M(C) (candidate fp32 master) during the probe phase
7. Returns a ProbeBundle for the host scorer

This module is deliberately free of secrets, seeds, and presigned URLs. Callers
pass opaque handles; IO goes through injected ``BoxIO`` ports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from .admission import AdmissionCert, EphemeralBoxKey, generate_ephemeral_box_key, verify_admission_cert
from .config import (
    GATE_DTYPE,
    GATE_K_CAND,
    GATE_MAX_OUTPUT_TOKENS,
    GATE_N_DRAWN,
    PitConfig,
)
from .state import ItemRecord, ItemState, TransitionError, apply_transition


class Fp32MasterError(RuntimeError):
    """Box-local mirror of the trainer fail-closed error.

    We deliberately do not import the trainer module here (heavy). The box
    runner raises this type when the expected-master digest check fails so
    host/controller code can treat it the same way as trainer Fp32MasterError.
    """

    def __init__(self, reason: str, **detail: Any):
        self.reason = reason
        self.detail = detail
        super().__init__(reason)


class BoxIO(Protocol):
    """Ports the box uses for download / train / probe / upload."""

    def download_head(self, head: str) -> str:
        """Return a local path handle for the bf16 head. No URLs in logs."""
        ...

    def download_master(self, head: str, expected_digest: str) -> str:
        """Return a local path handle for the fp32 master; must match digest."""
        ...

    def train_v834e(
        self,
        *,
        head_path: str,
        master_path: str,
        expected_master_digest: str,
        batch_handle: str,
        item_id: str,
    ) -> Mapping[str, Any]:
        """Run trainer-v834e. Must raise on fp32 fail-closed. No bootstrap override."""
        ...

    def probe_live_gate(
        self,
        *,
        candidate_path: str,
        candidate_id: str,
        plan_handle: str,
        k_cand: int,
        max_tokens: int,
        dtype: str,
    ) -> Mapping[str, Any]:
        """Run the live-gate probe replica on this box."""
        ...

    def upload_master_during_probe(self, *, candidate_id: str, master_path: str) -> Mapping[str, Any]:
        """Upload M(C) during probe (DESIGN §12). Returns opaque receipt handle."""
        ...


@dataclass
class ProbeBundle:
    """Host-facing result of a finished box run (no secrets)."""

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
    """Orchestrates one item on one box."""

    config: PitConfig
    io: BoxIO
    root_pubkey_hex: Optional[str] = None  # required to verify a real/fixture cert

    def __post_init__(self) -> None:
        if self.config.k_cand != GATE_K_CAND or self.config.max_output_tokens != GATE_MAX_OUTPUT_TOKENS:
            raise ValueError('box runner refuses non-live gate math')
        if self.config.dtype != GATE_DTYPE:
            raise ValueError('box runner refuses non-bf16 dtype')
        if self.config.n_drawn != GATE_N_DRAWN:
            raise ValueError('box runner refuses non-live N')

    def start_box_identity(self) -> EphemeralBoxKey:
        return generate_ephemeral_box_key()

    def admit(
        self,
        box_key: EphemeralBoxKey,
        cert: AdmissionCert,
        *,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.root_pubkey_hex:
            raise ValueError('root_pubkey_hex required to verify admission')
        return verify_admission_cert(
            cert,
            expected_root_pubkey_hex=self.root_pubkey_hex,
            now=now,
            box_public_key_hex=box_key.public_key_hex,
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
    ) -> tuple[ItemRecord, ProbeBundle]:
        """Drive DOWNLOADING -> TRAINING -> PROBING. Returns updated record + bundle.

        Caller owns RENTING / SCORING transitions on the host side.
        """
        if record.state is not ItemState.DOWNLOADING:
            raise TransitionError('box.run requires DOWNLOADING, got %s' % record.state.value)
        if not record.bound_head or not record.bound_master_digest:
            raise TransitionError('box.run requires bound head + master digest')
        if not batch_handle or batch_handle.startswith('secret:') or 'presign' in batch_handle.lower():
            raise ValueError('batch_handle must be an opaque non-secret placeholder')
        if not plan_handle or 'seed=' in plan_handle.lower():
            raise ValueError('plan_handle must not embed seeds')

        box_key = box_key or self.start_box_identity()
        if admission_cert is not None:
            self.admit(box_key, admission_cert, now=now)

        # --- download (fail-closed on master digest) ---
        head_path = self.io.download_head(record.bound_head)
        try:
            master_path = self.io.download_master(record.bound_head, record.bound_master_digest)
        except Exception as exc:
            # Normalize expected-master failures.
            msg = str(exc)
            if 'expected-master' in msg or 'Fp32MasterError' in type(exc).__name__:
                raise Fp32MasterError('expected-master-mismatch', detail=msg[:300]) from exc
            raise

        record = apply_transition(record, ItemState.TRAINING, reason='downloads-ready')

        # --- train (v834e; never pass fp32-bootstrap-override) ---
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
            raise ValueError('trainer must return candidate_path and candidate_master_path')

        record = apply_transition(
            record,
            ItemState.PROBING,
            reason='train-complete',
            updates={'candidate_id': candidate_id},
        )

        # --- probe + M(C) upload during probe ---
        master_upload = dict(self.io.upload_master_during_probe(
            candidate_id=candidate_id,
            master_path=candidate_master_path,
        ))
        probe_result = dict(self.io.probe_live_gate(
            candidate_path=candidate_path,
            candidate_id=candidate_id,
            plan_handle=plan_handle,
            k_cand=self.config.k_cand,
            max_tokens=self.config.max_output_tokens,
            dtype=self.config.dtype,
        ))
        if int(probe_result.get('k') or 0) != GATE_K_CAND:
            raise ValueError('probe k mismatch with live gate')
        if str(probe_result.get('dtype') or '') != GATE_DTYPE:
            raise ValueError('probe dtype mismatch')

        record = apply_transition(record, ItemState.SCORING, reason='probe-complete')
        bundle = ProbeBundle(
            item_id=record.item_id,
            bound_head=record.bound_head,
            bound_master_digest=record.bound_master_digest,
            candidate_id=candidate_id,
            probe_result=probe_result,
            master_upload=master_upload,
            train_receipt={
                k: v for k, v in train_receipt.items()
                if k not in ('seed', 'presigned_url', 'private_key', 'secret')
            },
            box_public_key_hex=box_key.public_key_hex,
            meta={'shadow': record.shadow},
        )
        return record, bundle


@dataclass
class ScriptedBoxIO:
    """Test double for BoxIO."""

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
        'z_inputs_ready': True,
    })
    upload_result: Dict[str, Any] = field(default_factory=lambda: {
        'ok': True,
        'receipt': 'mc-upload-fixture',
    })
    fail_master_digest: bool = False
    on_train: Optional[Callable[..., Mapping[str, Any]]] = None

    def download_head(self, head: str) -> str:
        return self.head_path

    def download_master(self, head: str, expected_digest: str) -> str:
        if self.fail_master_digest:
            raise Fp32MasterError('expected-master-mismatch', expected=expected_digest)
        return self.master_path

    def train_v834e(self, **kwargs: Any) -> Mapping[str, Any]:
        if 'fp32_bootstrap_override' in kwargs or kwargs.get('bootstrap_override'):
            raise Fp32MasterError('bootstrap-override-refused-by-pit')
        if self.on_train:
            return self.on_train(**kwargs)
        return dict(self.train_result)

    def probe_live_gate(self, **kwargs: Any) -> Mapping[str, Any]:
        return dict(self.probe_result)

    def upload_master_during_probe(self, **kwargs: Any) -> Mapping[str, Any]:
        return dict(self.upload_result)
