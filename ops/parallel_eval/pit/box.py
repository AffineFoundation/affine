"""PIT box runner (BRIEF §1.3) — provision → train → seal → serve → probe → result.

Fail-closed on fp32 expected-master mismatch. Never passes
``--fp32-bootstrap-override``. M(C) upload during probe when configured.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from ops.parallel_eval.live.pe_box import BOX_ROOT, _as_pod, detached_eval_command
from ops.parallel_eval.live.pe_probe_plan import build_probe_plan

from . import GATE_K_CAND, GATE_MAX_TOKENS, GATE_N_DRAWN
from .box_job import STATUS_RUNNING
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

        train_gpu = gpu_type_of(train_gpu) or gpu_type_of((record.meta or {}).get('train_gpu'))
        # Item records / pods.json persist a plain rent dict (no .run). Normalize
        # to a Pod before box-job controller / lane / train — otherwise detached
        # eval is skipped and the host takes the inert sync-serve path.
        raw_pod = (record.meta or {}).get('pod')
        pod = normalize_meta_pod(raw_pod)
        if pod is None and (self.lane and (self.lane.require_host or self.lane.host_run_probe is not None)):
            raise TransitionError('pod handle required for host box train/probe')

        # Re-adopt after host restart:
        # 1) RESULT.json → fetch bundle, never re-train / re-eval / release.
        # 2) train.done (+ optional live job.pid in eval) → skip train, resume eval
        #    via markers/pid (run_stage_once polls; never double-launch or kill).
        box_job_ctrl = _try_box_job_controller(pod, record.item_id)
        if box_job_ctrl is not None:
            prior = box_job_ctrl.resume_or_fetch()
            if prior and prior.get('probe_result') and prior.get('candidate_id'):
                return _bundle_from_box_result(
                    record, prior, box_key=box_key, train_gpu=train_gpu, eval_gpu=eval_gpu,
                )

        train_receipt: Dict[str, Any]
        training_report: Dict[str, Any]
        candidate_id: str
        candidate_path: Any
        candidate_master_path: Any
        master_upload: Dict[str, Any] = {}
        resumed_mid_eval = False

        train_marker = _train_marker_payload(box_job_ctrl) if box_job_ctrl is not None else None
        if train_marker and _train_marker_usable(train_marker):
            # Mid-eval / post-train re-adopt: do not re-download head/master or re-train.
            train_receipt = dict(train_marker)
            candidate_id = str(train_receipt.get('candidate_id') or train_receipt.get('checkpoint') or '')
            candidate_path = train_receipt.get('candidate_path')
            candidate_master_path = (
                train_receipt.get('candidate_master_path')
                or train_receipt.get('master_path')
            )
            if not isinstance(candidate_id, str) or len(candidate_id) != 64:
                raise ValueError('train.done marker missing candidate_id hex64')
            if not candidate_path or not candidate_master_path:
                raise ValueError('train.done marker missing candidate paths')
            train_gpu = gpu_type_of(train_receipt.get('train_gpu')) or train_gpu
            report_doc = train_receipt.get('training_report') or train_receipt.get('report_doc')
            if isinstance(train_receipt.get('training_report_summary'), Mapping):
                training_report = dict(train_receipt['training_report_summary'])
            elif isinstance(report_doc, Mapping):
                training_report = extract_report_summary(report_doc)
            else:
                training_report = {
                    'source_file_hashes': {'trainer': '0' * 64, 'batch': '0' * 64},
                    'batching': {'tasks': 0, 'rollouts': 0},
                    'tokens': {'prompt_tokens': 0, 'output_tokens': 0, 'loss_tokens': 0},
                    'update_size': 0.0,
                    'gradient_norm': 0.0,
                    'engine': 'packed',
                    'recovered': 'train-marker',
                }
            if record.state is ItemState.PROVISIONING:
                record = apply_transition(record, ItemState.TRAINING, reason='readopt-train-marker')
            if record.state is ItemState.TRAINING:
                record = apply_transition(
                    record, ItemState.SEALED, reason='readopt-train-marker',
                    updates={'candidate_id': candidate_id, 'sealed': True},
                )
            else:
                # Already past SEALED (SERVING/PROBING) on a previous tick — keep C.
                record.sealed = True
                record.candidate_id = candidate_id
            resumed_mid_eval = True
        else:
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

        lane = self.lane or LaneService()
        pod_handle = pod or record.box_id or 'box'
        if self.config.master_upload == 'during_probe' and not resumed_mid_eval:
            master_upload = dict(self.io.upload_master_during_probe(
                candidate_id=candidate_id, master_path=candidate_master_path,
            ))

        idx_list = list(indices) if indices is not None else None
        probe_result, eval_gpu, record = self._run_eval_stage(
            record,
            box_job_ctrl=box_job_ctrl,
            lane=lane,
            pod_handle=pod_handle,
            candidate_id=candidate_id,
            plan_handle=plan_handle,
            idx_list=idx_list,
            indices_count=indices_count,
            train_gpu=train_gpu,
            eval_gpu=eval_gpu,
            train_receipt=train_receipt,
            training_report=training_report,
            box_key=box_key,
            master_upload=master_upload,
        )
        return record, ProbeBundle(
            item_id=record.item_id,
            bound_head=record.bound_head,
            bound_master_digest=record.bound_master_digest,
            candidate_id=candidate_id,
            probe_result=probe_result,
            master_upload=master_upload,
            train_receipt={
                **{
                    k: v for k, v in train_receipt.items()
                    if k not in ('seed', 'presigned_url', 'private_key', 'secret', 'report_doc', 'training_report')
                },
                'train_gpu': train_gpu,
                'training_report_summary': dict(training_report),
            },
            box_public_key_hex=box_key.public_key_hex,
            training_report=dict(training_report),
            train_gpu=train_gpu,
            eval_gpu=eval_gpu,
            meta={'shadow': record.shadow, 'train_gpu': train_gpu, 'eval_gpu': eval_gpu},
        )

    def _run_eval_stage(
        self,
        record: ItemRecord,
        *,
        box_job_ctrl: Any,
        lane: LaneService,
        pod_handle: Any,
        candidate_id: str,
        plan_handle: str,
        idx_list: Optional[list],
        indices_count: int,
        train_gpu: Optional[str],
        eval_gpu: Optional[str],
        train_receipt: Mapping[str, Any],
        training_report: Mapping[str, Any],
        box_key: EphemeralBoxKey,
        master_upload: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Optional[str], ItemRecord]:
        """Detached eval (serve+probe): resume via eval.done / RESULT; never host-thread-only.

        Prefer resume over re-run. Re-run kills recorded PIDs / vLLM first.
        """
        record = apply_transition(record, ItemState.SERVING, reason='serve-start')

        def _host_eval_runner() -> Mapping[str, Any]:
            if lane.require_host or lane.host_boot is not None:
                lane.boot(pod_handle)
            lane.serve(pod_handle, candidate_id, max_num_seqs=self.config.max_num_seqs)
            if lane.require_host or lane.host_run_probe is not None:
                if idx_list is None:
                    raise TransitionError('indices list required for host run_probe')
                probe = lane.run_probe(
                    pod_handle, candidate_id,
                    indices=idx_list, plan_seed=plan_handle,
                    k=self.config.k_cand, max_tokens=self.config.max_output_tokens,
                )
            else:
                probe = lane.run_probe(
                    pod_handle, candidate_id,
                    indices=idx_list,
                    indices_count=indices_count if idx_list is None else None,
                    plan_seed_handle=plan_handle,
                    k=self.config.k_cand, max_tokens=self.config.max_output_tokens,
                    result=dict(self.io.probe_result_fixture()),
                )
            return {
                'ok': True,
                'candidate_id': candidate_id,
                'probe_result': dict(probe),
            }

        eval_marker: Dict[str, Any] = {}
        if box_job_ctrl is not None:
            # Prefer resume: if RESULT already written by a prior detached eval, use it.
            prior = box_job_ctrl.resume_or_fetch()
            if prior and prior.get('probe_result') and prior.get('candidate_id'):
                probe_result = dict(prior['probe_result'])
                eg = (
                    gpu_type_of(eval_gpu)
                    or gpu_type_of(probe_result.get('eval_gpu') or probe_result.get('gpu'))
                    or gpu_type_of((record.meta or {}).get('eval_gpu'))
                    or train_gpu
                )
                probe_result['train_gpu'] = train_gpu
                probe_result['eval_gpu'] = eg
                record = apply_transition(
                    record, ItemState.RESULT, reason='eval-resume-result',
                    updates={'result_stored': True},
                )
                return probe_result, eg, record

            detached_cmd = None
            # Real pod: ship plan + launch box-side detached eval (setsid).
            if hasattr(getattr(box_job_ctrl, 'store', None), 'pod') and (
                lane.require_host or lane.host_run_probe is not None
            ):
                if idx_list is None:
                    raise TransitionError('indices list required for host run_probe')
                detached_cmd = self._arm_detached_eval(
                    box_job_ctrl, candidate_id=candidate_id,
                    plan_handle=plan_handle, indices=idx_list,
                )

            # Host restart mid-eval: job.pid already live → poll, do not kill/relaunch.
            st0 = box_job_ctrl.status()
            already_running_eval = (
                st0.state == STATUS_RUNNING and st0.next_stage == 'eval'
            )

            def _host_forbidden() -> Mapping[str, Any]:
                raise RuntimeError('host must not sync-run eval when detached path is armed')

            eval_marker = dict(box_job_ctrl.run_stage_once(
                'eval',
                _host_eval_runner if detached_cmd is None else _host_forbidden,
                detached_command=detached_cmd,
                wait_for_marker=bool(detached_cmd) or already_running_eval,
                poll_s=float(os.environ.get('PIT_BOX_JOB_POLL_S') or 2.0),
                timeout_s=float(os.environ.get('PIT_EVAL_TIMEOUT_S') or 7200),
                extra_kill_patterns=('vllm serve', 'podctl.py serve', 'pit-box-job-eval'),
            ))
        else:
            eval_marker = dict(_host_eval_runner())

        probe_raw = eval_marker.get('probe_result') or eval_marker
        if not isinstance(probe_raw, Mapping):
            raise TransitionError('eval marker missing probe_result')
        probe_result = dict(probe_raw)
        eg = (
            gpu_type_of(eval_gpu)
            or gpu_type_of(probe_result.get('eval_gpu') or probe_result.get('gpu'))
            or gpu_type_of((record.meta or {}).get('eval_gpu'))
            or train_gpu
        )
        probe_result['train_gpu'] = train_gpu
        probe_result['eval_gpu'] = eg
        if record.state is ItemState.SERVING:
            record = apply_transition(record, ItemState.PROBING, reason='probe-start')
        record = apply_transition(
            record, ItemState.RESULT, reason='eval-complete', updates={'result_stored': True},
        )
        if box_job_ctrl is not None:
            try:
                bundle_tmp = ProbeBundle(
                    item_id=record.item_id,
                    bound_head=record.bound_head or '',
                    bound_master_digest=record.bound_master_digest or '',
                    candidate_id=candidate_id,
                    probe_result=probe_result,
                    master_upload=dict(master_upload),
                    train_receipt=dict(train_receipt),
                    box_public_key_hex=box_key.public_key_hex,
                    training_report=dict(training_report),
                    train_gpu=train_gpu,
                    eval_gpu=eg,
                )
                if 'eval' not in box_job_ctrl.status().markers:
                    box_job_ctrl.mark_stage('eval', {
                        'ok': True,
                        'candidate_id': candidate_id,
                        'samples': probe_result.get('samples'),
                        'probe_result': probe_result,
                    })
                box_job_ctrl.publish_result(_box_result_doc(bundle_tmp, train_receipt=train_receipt))
            except Exception:
                pass
        return probe_result, eg, record

    def _arm_detached_eval(
        self,
        box_job_ctrl: Any,
        *,
        candidate_id: str,
        plan_handle: str,
        indices: list,
    ) -> str:
        """Ship live-identical eval plan JSON onto the box; return detached command.

        Plan is built by ``pe_probe_plan.build_probe_plan`` — the same builder
        ``pe_lane.run_probe`` uses — so harness / manifest_projection cannot drift.
        """
        harness = _load_public_harness()
        plan = build_probe_plan(
            candidate_id,
            list(indices),
            plan_handle,
            k=int(self.config.k_cand),
            max_tokens=int(self.config.max_output_tokens),
            max_num_seqs=int(self.config.max_num_seqs),
            harness=harness,
        )
        plan_path = '%s/eval-plan.json' % box_job_ctrl.job_dir.rstrip('/')
        box_job_ctrl.store.write_text(plan_path, json.dumps(plan, sort_keys=True) + '\n')
        return detached_eval_command(
            job_dir=box_job_ctrl.job_dir,
            candidate_id=candidate_id,
            plan_path=plan_path,
            podd=BOX_ROOT,
            max_num_seqs=int(self.config.max_num_seqs),
        )


def _load_public_harness() -> dict:
    """Best-effort public_harness.json (same source pe_lane.HARNESS uses)."""
    candidates = []
    env = os.environ.get('PIT_PUBLIC_HARNESS') or os.environ.get('PE_PUBLIC_HARNESS')
    if env:
        candidates.append(Path(env))
    # Live pe_common PRIV/public_harness.json when pe_lane is importable.
    try:
        live = Path(__file__).resolve().parents[1] / 'live'
        if str(live) not in sys.path:
            sys.path.insert(0, str(live))
        import pe_lane as PL  # type: ignore
        h = getattr(PL, 'HARNESS', None)
        if isinstance(h, dict):
            return dict(h)
    except Exception:
        pass
    for p in candidates:
        try:
            doc = json.loads(Path(p).read_text())
            if isinstance(doc, dict):
                return doc
        except Exception:
            continue
    return {}


def normalize_meta_pod(pod: Any) -> Any:
    """Turn a persisted rent dict / Pod into an object with ``.run``.

    Controller writes ``meta['pod']`` as a plain dict (id/ip/port/…) into the
    item record and pods.json. Callers that need SSH must normalize via
    ``pe_box._as_pod`` — never gate on ``hasattr(pod, 'run')`` of the raw dict.
    """
    if pod is None:
        return None
    try:
        return _as_pod(pod)
    except Exception:
        # Already a Pod-like, or unusable handle — surface None so callers
        # fall back / fail closed rather than crashing the host tick.
        if hasattr(pod, 'run') and callable(getattr(pod, 'run')):
            return pod
        return None


def _try_box_job_controller(pod: Any, item_id: str):
    """Return a BoxJobController when the pod can run remote shell commands.

    Accepts a live Pod *or* the persisted rent dict from ``meta['pod']``.
    """
    P = normalize_meta_pod(pod)
    if P is None or not hasattr(P, 'run'):
        return None
    try:
        from .box_job import controller_for_pod
        return controller_for_pod(P, item_id)
    except Exception:
        return None


def _train_marker_payload(box_job_ctrl: Any) -> Optional[Dict[str, Any]]:
    """Read markers/train.done payload when present (host restart mid-eval)."""
    try:
        st = box_job_ctrl.status()
        if 'train' not in (st.markers or set()):
            return None
        return dict(box_job_ctrl._read_stage_payload('train') or {})
    except Exception:
        return None


def _train_marker_usable(payload: Mapping[str, Any]) -> bool:
    cid = payload.get('candidate_id') or payload.get('checkpoint')
    path = payload.get('candidate_path')
    master = payload.get('candidate_master_path') or payload.get('master_path')
    return (
        isinstance(cid, str) and len(cid) == 64
        and bool(path) and bool(master)
    )


def _box_result_doc(bundle: ProbeBundle, *, train_receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Durable on-box RESULT.json — enough to rebuild ProbeBundle after host restart."""
    safe_receipt = {
        k: v for k, v in dict(train_receipt).items()
        if k not in ('seed', 'presigned_url', 'private_key', 'secret', 'pod')
    }
    return {
        'item_id': bundle.item_id,
        'bound_head': bundle.bound_head,
        'bound_master_digest': bundle.bound_master_digest,
        'candidate_id': bundle.candidate_id,
        'candidate_path': safe_receipt.get('candidate_path'),
        'candidate_master_path': safe_receipt.get('candidate_master_path'),
        'candidate_on_box': safe_receipt.get('candidate_on_box', True),
        'box_candidate_dir': safe_receipt.get('box_candidate_dir'),
        'candidate_files': safe_receipt.get('candidate_files'),
        'candidate_master_digest': safe_receipt.get('candidate_master_digest'),
        'defer_candidate_upload': safe_receipt.get('defer_candidate_upload', True),
        'probe_result': dict(bundle.probe_result),
        'master_upload': dict(bundle.master_upload or {}),
        'train_receipt': safe_receipt,
        'training_report': dict(bundle.training_report or {}),
        'train_gpu': bundle.train_gpu,
        'eval_gpu': bundle.eval_gpu,
        'box_public_key_hex': bundle.box_public_key_hex,
        'report': safe_receipt.get('report'),
        'report_doc': safe_receipt.get('report_doc'),
        'ok': True,
    }


def _bundle_from_box_result(
    record: ItemRecord,
    prior: Mapping[str, Any],
    *,
    box_key: EphemeralBoxKey,
    train_gpu: Optional[str],
    eval_gpu: Optional[str],
) -> tuple[ItemRecord, ProbeBundle]:
    """Rebuild state + ProbeBundle from on-box RESULT.json after re-adopt."""
    candidate_id = str(prior.get('candidate_id') or '')
    if len(candidate_id) != 64:
        raise ValueError('box RESULT missing candidate_id')
    probe_result = dict(prior.get('probe_result') or {})
    train_receipt = dict(prior.get('train_receipt') or prior)
    training_report = dict(prior.get('training_report') or {})
    if not training_report and isinstance(train_receipt.get('training_report_summary'), Mapping):
        training_report = dict(train_receipt['training_report_summary'])
    train_gpu = (
        gpu_type_of(train_gpu)
        or gpu_type_of(prior.get('train_gpu'))
        or gpu_type_of(train_receipt.get('train_gpu'))
    )
    eval_gpu = (
        gpu_type_of(eval_gpu)
        or gpu_type_of(prior.get('eval_gpu'))
        or gpu_type_of(probe_result.get('eval_gpu') or probe_result.get('gpu'))
        or train_gpu
    )
    # Walk legal edges from PROVISIONING to RESULT.
    record = apply_transition(record, ItemState.TRAINING, reason='readopt-result')
    record = apply_transition(
        record, ItemState.SEALED, reason='readopt-result',
        updates={'candidate_id': candidate_id, 'sealed': True},
    )
    record = apply_transition(record, ItemState.SERVING, reason='readopt-result')
    record = apply_transition(record, ItemState.PROBING, reason='readopt-result')
    record = apply_transition(
        record, ItemState.RESULT, reason='readopt-result',
        updates={'result_stored': True},
    )
    safe_receipt = {
        k: v for k, v in train_receipt.items()
        if k not in ('seed', 'presigned_url', 'private_key', 'secret', 'report_doc', 'training_report')
    }
    safe_receipt['train_gpu'] = train_gpu
    if training_report:
        safe_receipt['training_report_summary'] = dict(training_report)
    bundle = ProbeBundle(
        item_id=record.item_id,
        bound_head=record.bound_head or str(prior.get('bound_head') or ''),
        bound_master_digest=record.bound_master_digest or str(prior.get('bound_master_digest') or ''),
        candidate_id=candidate_id,
        probe_result=probe_result,
        master_upload=dict(prior.get('master_upload') or {}),
        train_receipt=safe_receipt,
        box_public_key_hex=str(prior.get('box_public_key_hex') or box_key.public_key_hex),
        training_report=training_report,
        train_gpu=train_gpu,
        eval_gpu=eval_gpu,
        meta={'shadow': record.shadow, 'train_gpu': train_gpu, 'eval_gpu': eval_gpu, 'readopt_result': True},
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
