"""Read-only recovery checks for uncertain weight submissions.

This module neither signs nor submits transactions and
never edits a cursor. The caller must hold the existing global writer lock when
journaling or applying a decision, then advance the resolved window monotonically.

Journal *before* execute. expected_weights must be the SDK's final clipped,
max-upscaled u16 vector, not scores or sum-normalized floats. A pre-read account
nonce is only a hint. A separate SDK plan is NOT the executed call: execute plans
again and generates fresh timelock ciphertext. Actual signed-call identity may be
added as a separate immutable observation only if captured from that exact call.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Protocol


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _integer(value: Any, name: str, low: int = 0, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"invalid {name}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc
    return value.lower()


def _vector(rows: Any, *, permit_zero: bool = False) -> list[list[int]]:
    if not isinstance(rows, (list, tuple)):
        raise ValueError("weight vector must be explicit pairs")
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("invalid weight pair")
        uid = _integer(row[0], "uid", high=65535)
        weight = _integer(row[1], "u16 weight", low=0 if permit_zero else 1, high=65535)
        if uid in seen:
            raise ValueError("duplicate uid")
        seen.add(uid)
        # Zeroed rows from deregistration are semantically absent, never positive.
        if weight:
            result.append([uid, weight])
    return sorted(result)


def validate_era(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {'period', 'birth', 'death', 'block_hash'}:
        raise ValueError('exact prepared mortal era metadata required')
    if type(value['period']) is not int or value['period'] != 64:
        raise ValueError('only actual mortal64 is qualified')
    birth = _integer(value['birth'], 'era birth', low=1)
    if type(value['death']) is not int or value['death'] != birth + 64:
        raise ValueError('actual era death mismatch')
    block_hash = value['block_hash']
    if not isinstance(block_hash, str) or not block_hash.startswith('0x'):
        raise ValueError('prepared era block anchor required')
    _digest(block_hash[2:], 'era block hash')
    return dict(value)


def make_intent(*, owner: str, owner_uid: int, netuid: int, window_end: int,
                expected_weights: list[list[int]], assessment_sha256: str,
                policy_sha256: str, registrations_sha256: str,
                attempt_start_block: int, nonce_hint: int | None = None,
                mecid: int = 0) -> dict[str, Any]:
    """Freeze one attempt's public inputs; this function does not normalize them."""
    if not isinstance(owner, str) or not owner:
        raise ValueError("owner is required")
    _integer(owner_uid, "owner_uid", high=65535)
    _integer(netuid, "netuid", high=65535)
    _integer(mecid, "mecid", high=255)
    _integer(window_end, "window_end", low=3600)
    if window_end % 3600:
        raise ValueError("window_end must be an hourly boundary")
    _integer(attempt_start_block, "attempt_start_block", low=1)
    if nonce_hint is not None:
        _integer(nonce_hint, "nonce_hint")
    weights = _vector(expected_weights)
    if not weights or max(weight for _, weight in weights) != 65535:
        raise ValueError("expected SDK vector must be nonempty and max-upscaled")
    if any(uid == owner_uid for uid, _ in weights):
        raise ValueError("owner weights are forbidden")
    payload = dict(version="weight-submission-intent-v1", owner=owner,
                   owner_uid=owner_uid, netuid=netuid, mecid=mecid,
                   window_end=window_end, expected_weights=weights,
                   expected_weights_sha256=_sha(weights),
                   assessment_sha256=_digest(assessment_sha256, "assessment_sha256"),
                   policy_sha256=_digest(policy_sha256, "policy_sha256"),
                   registrations_sha256=_digest(registrations_sha256, "registrations_sha256"),
                   attempt_start_block=attempt_start_block, nonce_hint=nonce_hint)
    return {"payload": payload, "sha256": _sha(payload)}


def validate_intent(document: Mapping[str, Any]) -> dict[str, Any]:
    """Reject altered, ambiguous, or unsupported journals before chain reads."""
    payload = document["payload"]
    fields = {key: value for key, value in payload.items()
              if key not in ("version", "expected_weights_sha256")}
    rebuilt = make_intent(**fields)
    if dict(document) != rebuilt:
        raise ValueError("intent integrity mismatch")
    return rebuilt["payload"]


def journal_intent(path: str | Path, document: Mapping[str, Any]) -> None:
    """Durably create once. Existing or partial journals are never overwritten.

    Any exception prevents execute. A crash during this write can leave a partial
    file; it must remain fenced for inspection because this code never repairs it.
    """
    validate_intent(document)
    target = Path(path)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_bytes(document) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class ReadonlyChainReader(Protocol):
    """Adapter must use bounded RPCs and canonical finalized block hashes.

    finalized_head -> {number, hash}; owner_state at that exact hash ->
    {block, block_hash, last_update, weights, owner_pending: bool|None}.
    block -> {number, hash, commits}, with each commit containing signer, netuid,
    mecid, nonce, extrinsic_hash, extrinsic_index, call_function and success.
    success requires System.ExtrinsicSuccess for that exact extrinsic index.
    Pending must inspect all supported modern timelock queues; an unavailable or
    unrecognized queue is unknown (None), not empty. Never mix head snapshots.
    """

    def finalized_head(self) -> dict[str, Any]: ...
    def owner_state(self, intent: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]: ...
    def block(self, number: int) -> dict[str, Any]: ...


class BittensorReadonlyReader:
    """Thin adapter for the locally installed synchronous SDK; never loads keys.

    Pass an already constructed ``bt.subtensor`` with bounded transport timeouts.
    This uses read-only internals whose shape must be rechecked on SDK upgrades.
    It supports the current direct CR4/mecid0 path, refusing unknown variants.
    """

    def __init__(self, chain: Any):
        self.chain = chain
        self.substrate = chain._client._substrate
        self._finalized: dict[str, Any] | None = None

    @staticmethod
    def _unwrap_rpc(value: Any) -> Any:
        return value.get("result", value) if isinstance(value, dict) else value

    def finalized_head(self) -> dict[str, Any]:
        # Open the SDK connection with an ordinary read before accessing raw RPC.
        _ = self.chain.block
        raw = self.substrate.raw
        block_hash = self._unwrap_rpc(self.chain._call(raw.rpc_request("chain_getFinalizedHead", [])))
        header = self._unwrap_rpc(self.chain._call(raw.rpc_request("chain_getHeader", [block_hash])))
        number = header["number"]
        number = int(number, 16) if isinstance(number, str) else int(number)
        self._finalized = {"number": number, "hash": block_hash}
        return dict(self._finalized)

    def _query(self, name: str, params: list[Any], block_hash: str) -> Any:
        return self.chain._call(self.substrate.query("SubtensorModule", name, params,
                                                    block_hash=block_hash))

    def owner_state(self, intent: Mapping[str, Any], head: Mapping[str, Any]) -> dict[str, Any]:
        if self._finalized != dict(head):
            raise ValueError("head was not read by this adapter")
        block_hash, netuid, owner = head["hash"], intent["netuid"], intent["owner"]
        if intent["mecid"] != 0 or self._query("CommitRevealWeightsVersion", [], block_hash) != 4:
            raise ValueError("unsupported commit variant")
        if self._query("Uids", [netuid, owner], block_hash) != intent["owner_uid"]:
            raise ValueError("owner UID changed")
        if self._query("Keys", [netuid, intent["owner_uid"]], block_hash) != owner:
            raise ValueError("owner UID reverse mapping changed")
        pending = False
        for name, width in (("TimelockedWeightCommits", 4),
                            ("CRV3WeightCommits", 3), ("CRV3WeightCommitsV2", 4)):
            rows = self.chain._call(self.substrate.query_map("SubtensorModule", name,
                                                            [netuid], block_hash=block_hash))
            for _, queue in rows:
                if not isinstance(queue, (list, tuple)):
                    raise ValueError("unknown pending queue shape")
                for entry in queue:
                    if (not isinstance(entry, (list, tuple)) or len(entry) != width
                            or not isinstance(entry[0], str)):
                        raise ValueError("unknown pending commit shape")
                    pending = pending or entry[0] == owner
        updates = self._query("LastUpdate", [netuid], block_hash)
        weights = self._query("Weights", [netuid, intent["owner_uid"]], block_hash)
        return dict(block=head["number"], block_hash=block_hash,
                    last_update=updates[intent["owner_uid"]],
                    weights=weights, owner_pending=pending)

    def block(self, number: int) -> dict[str, Any]:
        if self._finalized is None or not 0 < number <= self._finalized["number"]:
            raise ValueError("requested block is not finalized")
        block = self.chain._call(self.substrate.get_block(block_number=number))
        if block is None or block["header"]["number"] != number:
            raise ValueError("missing canonical block")
        block_hash = block["header"]["hash"]
        events = self.chain._call(self.substrate.events(block_hash=block_hash))
        commits = []
        for index, item in enumerate(block["extrinsics"]):
            extrinsic = getattr(item, "value", item)
            if extrinsic is None:
                raise ValueError("undecoded extrinsic cannot be skipped")
            call = extrinsic["call"]
            if (call["call_module"] != "SubtensorModule" or
                    call["call_function"] != "commit_timelocked_mechanism_weights"):
                continue
            args = {arg["name"]: arg["value"] for arg in call["call_args"]}
            matching_events = [event for event in events if event.get("extrinsic_idx") == index]
            dispatch_ok = any(event.get("module_id") == "System" and
                              event.get("event_id") == "ExtrinsicSuccess" for event in matching_events)
            failed = any(event.get("module_id") == "System" and
                         event.get("event_id") == "ExtrinsicFailed" for event in matching_events)
            committed = any(event.get("module_id") == "SubtensorModule" and
                            event.get("event_id") == "TimelockedWeightsCommitted"
                            for event in matching_events)
            commits.append(dict(signer=extrinsic["address"], netuid=args["netuid"],
                                mecid=args["mecid"], nonce=extrinsic["nonce"],
                                extrinsic_hash=extrinsic["extrinsic_hash"],
                                extrinsic_index=index, call_function=call["call_function"],
                                reveal_round=args["reveal_round"],
                                success=dispatch_ok and committed,
                                dispatch_failed=failed and not dispatch_ok and not committed))
        # Every extrinsic was strictly decoded above. Full-era absence needs
        # all hashes, not merely the subset that looked like weight commits.
        hashes = [getattr(item, 'value', item)['extrinsic_hash'] for item in block['extrinsics']]
        return dict(number=number, hash=block_hash, commits=commits, extrinsic_hashes=hashes)


def collect_evidence(reader: ReadonlyChainReader, document: Mapping[str, Any], *,
                     max_blocks: int = 128, budget_seconds: float = 60,
                     monotonic=time.monotonic,
                     actual_signed_observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read a bounded contiguous prefix; partial scans cannot imply absence.

    RPC timeouts remain the reader's responsibility: the time budget is checked
    between calls and cannot cancel a stuck RPC. Store the receipt before applying
    a decision. Legacy vector-only recovery is LastUpdate-scoped. Actual signed
    identity recovery inspects the prepared mortal era independently of LastUpdate;
    full finalized-era coverage is mandatory before claiming expiration/absence.
    """
    payload = validate_intent(document)
    _integer(max_blocks, "max_blocks", low=1, high=512)
    if isinstance(budget_seconds, bool) or not 0 < budget_seconds <= 180:
        raise ValueError("budget_seconds must be in (0,180]")
    deadline = monotonic() + budget_seconds
    evidence: dict[str, Any] = {"version": "weight-reconciliation-evidence-v1",
                                "intent_sha256": document["sha256"],
                                "blocks": [], "read_error": None}
    try:
        head = reader.finalized_head()
        evidence["finalized_head"] = head
        if monotonic() >= deadline:
            raise TimeoutError("read budget exhausted")
        state = reader.owner_state(payload, head)
        evidence["owner_state"] = state
        start = payload["attempt_start_block"]
        if actual_signed_observation is not None:
            if actual_signed_observation['intent_sha256'] != document['sha256']:
                raise ValueError('signed observation intent mismatch')
            era = actual_signed_observation.get('era')
            if era is not None:
                era = validate_era(era)
                start = era['birth']
                stop = min(int(head['number']), era['death'] - 1, start + max_blocks - 1)
                evidence['scan_kind'] = 'actual-signed-mortal-era-v1'
            else:
                stop = min(int(head['number']), start + max_blocks - 1)
                evidence['scan_kind'] = 'actual-signed-prefix-v1'
        else:
            stop = min(int(head["number"]), int(state["last_update"]), start + max_blocks - 1)
        evidence["scan_start"] = start
        evidence["scan_stop"] = stop
        for number in range(start, stop + 1):
            if monotonic() >= deadline:
                raise TimeoutError("read budget exhausted")
            block = reader.block(number)
            if block["number"] != number:
                raise ValueError("noncontiguous block read")
            evidence["blocks"].append(block)
            if actual_signed_observation is not None and any(
                commit.get('extrinsic_hash') == actual_signed_observation['extrinsic_hash']
                and commit.get('nonce') == actual_signed_observation['nonce']
                for commit in block['commits']):
                # Positive inclusion can stop early. Absence requires the full era.
                evidence['scan_stop'] = number
                break
    except Exception as exc:
        # Do not persist provider exception messages: they may contain RPC tokens.
        evidence["read_error"] = type(exc).__name__
    return evidence


def reconcile(document: Mapping[str, Any], evidence: Mapping[str, Any], *,
              actual_signed_observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve proven finality or complete mortal-era absence; never replay an old call.

    An optional signed observation is caller-captured public metadata from the
    actual signed extrinsic, binding intent_sha256, extrinsic_hash and nonce. It
    must not be synthesized from a nonce pre-read or a separate SDK plan.
    """
    result: dict[str, Any] = dict(status="unresolved", reason="invalid_evidence",
                                  preserve_fence=True, old_window_replay_allowed=False,
                                  allow_current_window_submission=False,
                                  intent_sha256=document.get("sha256"))

    def fence(reason: str) -> dict[str, Any]:
        return dict(result, reason=reason)

    try:
        intent = validate_intent(document)
        if evidence.get("intent_sha256") != document["sha256"]:
            return fence("wrong_intent")
        if evidence.get("read_error") is not None:
            return fence("incomplete_read")
        head, state = evidence["finalized_head"], evidence["owner_state"]
        head_number = _integer(head["number"], "finalized_head", low=1)
        if state["block"] != head_number or state["block_hash"] != head["hash"]:
            return fence("mixed_chain_snapshots")
        pending = state["owner_pending"]
        if type(pending) is not bool:
            return fence("unknown_pending_state")
        blocks = evidence["blocks"]
        start, stop = evidence["scan_start"], evidence["scan_stop"]
        era = None
        expected_start = intent['attempt_start_block']
        if actual_signed_observation is not None and actual_signed_observation.get('era') is not None:
            era = validate_era(actual_signed_observation['era'])
            expected_start = era['birth']
            if evidence.get('scan_kind') != 'actual-signed-mortal-era-v1':
                return fence('wrong_era_scan')
        if start != expected_start or stop > head_number:
            return fence("invalid_scan_bounds")
        if [block["number"] for block in blocks] != list(range(start, stop + 1)):
            return fence("incomplete_scan")
        if era is not None and (not blocks or blocks[0]['hash'] != era['block_hash']):
            return fence('prepared_era_anchor_mismatch')
        commits = []
        for block in blocks:
            for commit in block["commits"]:
                if (commit["signer"] == intent["owner"] and
                    commit["netuid"] == intent["netuid"] and
                    commit["mecid"] == intent["mecid"] and
                    commit["call_function"] == "commit_timelocked_mechanism_weights"):

                    commits.append(dict(commit, block=block["number"], block_hash=block["hash"]))
        if actual_signed_observation is not None:
            observation = actual_signed_observation
            if observation["intent_sha256"] != document["sha256"]:
                return fence("wrong_signed_observation")
            commits = [commit for commit in commits
                       if commit["extrinsic_hash"] == observation["extrinsic_hash"]
                       and commit["nonce"] == observation["nonce"]]
        else:
            # A pre-read nonce is deliberately not used as signed identity.
            # LastUpdate + exact revealed vector can independently bind success.
            commits = [commit for commit in commits if commit["block"] == state["last_update"]
                       and commit['success'] is True]
        if len(commits) == 0 and era is not None and head_number >= era['death']:
            if stop != era['death'] - 1 or len(blocks) != era['period']:
                return fence('incomplete_finalized_era_absence')
            actual_hash = actual_signed_observation['extrinsic_hash']
            if any(not isinstance(block.get('extrinsic_hashes'), list) for block in blocks):
                return fence('incomplete_block_extrinsic_inventory')
            if any(actual_hash in block['extrinsic_hashes'] for block in blocks):
                return fence('signed_hash_present_without_bound_dispatch')
            return dict(result, status='expired_unincluded', reason='complete_finalized_era_absence',
                        preserve_fence=False, allow_current_window_submission=True,
                        resolved_attempt_window_end=intent['window_end'], submission_succeeded=False,
                        evidence_sha256=_sha(evidence))
        if len(commits) != 1:
            return fence("missing_or_ambiguous_finalized_commit")
        commit = commits[0]
        if actual_signed_observation is not None:
            if commit.get('dispatch_failed') is True and commit['success'] is False:
                return dict(result, status='failed_finalized', reason='exact_signed_finalized_failure',
                            preserve_fence=False, allow_current_window_submission=True,
                            resolved_attempt_window_end=intent['window_end'], submission_succeeded=False,
                            commit=commit, evidence_sha256=_sha(evidence))
            if commit['success'] is not True:
                return fence('unknown_finalized_dispatch_outcome')
            # The actual SDK-produced signed hash and nonce were durably recorded
            # before its sole send. Canonical finalized success proves that exact
            # submission, even while drand reveal is pending or a later writer
            # has advanced LastUpdate. No guessed preflight identity is accepted.
            return dict(result, status="submitted_finalized", reason="exact_signed_finalized_commit",
                        preserve_fence=False, allow_current_window_submission=True,
                        resolved_window_end=intent["window_end"], commit=commit,
                        owner_timelock_pending=pending, revealed_weights_asserted=False,
                        expected_weights_sha256=intent["expected_weights_sha256"],
                        evidence_sha256=_sha(evidence))
        if pending:
            return dict(result, status="pending", reason="owner_timelock_pending",
                        commit=commit, finalized_commit_bound=actual_signed_observation is not None)
        if state["last_update"] != commit["block"]:
            return fence("last_update_mismatch")
        if _vector(state["weights"], permit_zero=True) != intent["expected_weights"]:
            return fence("revealed_vector_mismatch")
        return dict(result, status="submitted_revealed", reason="exact_finalized_reveal",
                    preserve_fence=False, allow_current_window_submission=True,
                    resolved_window_end=intent["window_end"], commit=commit,
                    expected_weights_sha256=intent["expected_weights_sha256"],
                    evidence_sha256=_sha(evidence))
    except (KeyError, TypeError, ValueError, OverflowError):
        return fence("invalid_evidence")
