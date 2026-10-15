"""Signed, append-only succession of qualified MiMo release policies.

A single ``ReleasePolicy`` binds exactly one qualified release, and every
historical artifact (audit population, evaluator opening, trainer private state,
writer population) is re-validated against that one policy. That makes an
in-place release switch impossible: a running lifecycle cannot adopt a successor
release while its durable history was governed by the predecessor.

This module adds the minimal, fail-closed primitive required for a run to switch
releases without a fresh start: an operator-signed, append-only chain of release
policies, ordered by the first global round each one governs. Each historical
manifest already embeds the exact signed policy that governed it (``POLICY_KEY``);
the chain says which signed policies are authorized members of this run's
succession and in what order. A ``ReleaseChain`` is therefore a drop-in guard for
the single-policy call sites: it selects the chain member a manifest names,
re-authenticates it, and validates the manifest against *that* member, while the
newest member governs new epochs.

There is no network or chain side effect here. An unsigned or broken chain, a
member signed by a different authority, a non-increasing (non-append-only) order,
or a manifest whose embedded policy is not an exact chain member all reject
exactly as the single-policy gate did.
"""
from __future__ import annotations

import copy
import re

from .backend_jobs import signed
from .mimo_release_policy import ReleasePolicy, POLICY_KEY, digest, need

SUCCESSION_VERSION = 'mimo-release-succession-v1'


class ReleaseChain:
    """An ordered, authority-signed set of qualified releases for one run.

    ``entries`` is ``[{policy: <signed release policy envelope>, first_round: int}]``
    in strictly increasing ``first_round`` order (append-only). Every member is a
    fully validated :class:`ReleasePolicy` over the same ``run_id``; the last
    member is the active release that governs new epochs.
    """

    def __init__(self, envelope, authority, *, allow_public=False):
        need(type(allow_public) is bool, 'explicit local release execution boolean')
        payload = copy.deepcopy(signed(envelope, authority))
        need(type(payload) is dict and set(payload) == {'version', 'run_id', 'entries'},
             'exact release succession fields')
        need(payload['version'] == SUCCESSION_VERSION, 'explicit release succession version')
        run_id = payload['run_id']
        need(type(run_id) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id),
             'safe release succession run namespace')
        entries = payload['entries']
        need(type(entries) is list and 1 <= len(entries) <= 64, 'bounded release succession chain')
        self.members = []          # list[(first_round, ReleasePolicy)] in chain order
        self._by_sha = {}          # policy_sha256 -> (position, first_round, ReleasePolicy)
        previous_round = -1
        for position, entry in enumerate(entries):
            need(type(entry) is dict and set(entry) == {'policy', 'first_round'},
                 'exact release succession entry')
            first_round = entry['first_round']
            need(type(first_round) is int and first_round > previous_round,
                 'strictly increasing append-only release succession rounds')
            previous_round = first_round
            policy = ReleasePolicy(entry['policy'], authority, allow_public=allow_public)
            need(policy.run_id == run_id, 'release succession members share one run')
            need(policy.policy_sha256 not in self._by_sha, 'release succession members are distinct')
            self.members.append((first_round, policy))
            self._by_sha[policy.policy_sha256] = (position, first_round, policy)
        self.authority = authority
        self.run_id = run_id
        self.envelope = copy.deepcopy(envelope)
        self.chain_sha256 = digest(envelope)
        self.active = self.members[-1][1]
        self.baseline_checkpoint_id = self.members[0][1].scope['initial_checkpoint']['id']

    # --- active-release proxies (the newest member governs new work) ----------
    @property
    def policy_sha256(self):
        return self.active.policy_sha256

    @property
    def mode(self):
        return self.active.mode

    @property
    def scope(self):
        return self.active.scope

    def namespace(self, manifest=None):
        # Namespace is a per-run value shared across the succession (members share
        # run_id). Validate against the member the manifest actually names.
        if manifest is not None:
            self.policy_for_manifest(manifest)
        return self.active.namespace()

    # --- membership and ordering ---------------------------------------------
    def member(self, policy_sha256):
        """Return the ``ReleasePolicy`` for a member sha, or ``None``."""
        entry = self._by_sha.get(policy_sha256)
        return None if entry is None else entry[2]

    def position(self, policy_sha256):
        """Return the chain index of a member sha, or ``None``."""
        entry = self._by_sha.get(policy_sha256)
        return None if entry is None else entry[0]

    def succeeds(self, predecessor_sha256, successor_sha256):
        """True iff both are members and the predecessor is strictly earlier."""
        a = self._by_sha.get(predecessor_sha256)
        b = self._by_sha.get(successor_sha256)
        return a is not None and b is not None and a[0] < b[0]

    # --- the single-policy drop-in guard -------------------------------------
    def policy_for_manifest(self, manifest):
        """Select, re-authenticate and validate against the named chain member."""
        need(type(manifest) is dict, 'original manifest for release selection')
        envelope = manifest.get(POLICY_KEY)
        need(envelope is not None, 'manifest carries no release policy')
        entry = self._by_sha.get(digest(envelope))
        need(entry is not None, 'manifest policy is not a signed release chain member')
        policy = entry[2]
        # Byte-exact identity, not only a digest match, before trusting it.
        need(envelope == policy.envelope, 'manifest policy differs from signed chain member')
        policy.validate_manifest(manifest)
        return policy

    def validate_manifest(self, manifest):
        self.policy_for_manifest(manifest)
        return manifest


def load_release(envelope, authority, *, allow_public=False):
    """Build a :class:`ReleaseChain` from a succession envelope, or a single
    :class:`ReleasePolicy` from a release-policy envelope.

    Both results expose ``validate_manifest``, ``policy_sha256``, ``namespace``,
    ``mode``, ``run_id``, ``authority`` and ``scope``, so a caller that only
    needs the single-policy surface can accept either transparently. The kind is
    decided by the signed payload's ``version`` after authority verification, so
    an unsigned or wrong-authority envelope rejects before any dispatch.
    """
    payload = signed(envelope, authority)
    if type(payload) is dict and payload.get('version') == SUCCESSION_VERSION:
        return ReleaseChain(envelope, authority, allow_public=allow_public)
    return ReleasePolicy(envelope, authority, allow_public=allow_public)
