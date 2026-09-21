"""wvk 19 staging — per-duel confirmation slice for crowns.

Knob `[duel].confirmation_required` (default False = pre-wvk-19: a duel that
clears max(k_sigma·SE, δ) + gates crowns at once). When on, a first-slice
pass is NOT a crown yet: the validator runs one more independent n_turns
slice against the same king (seed = blake2b(block_hash ‖ hotkey ‖ "|slice1"),
turns disjoint from the first slice, fresh teacher references, warm
engines) and crowns only if (a) that slice's own paired margin is > 0 and
(b) the POOLED margin over both slices clears max(k_sigma·SE_pooled, δ).
Otherwise the verdict is a loss with rejection_reason "confirmation_failed"
(slot consumed like any loss, no re-queue) and the king stands.

Reuses the wvk-15 confirmation machinery (pod: `confirm` request,
confirmation_stamp / pooled_margin_stats; validator: run_duel(confirm=)).
Semantics differ from wvk 15 (`confirm.rule = "per_duel"`): the pass test
is (a)+(b) above, not "pooled > 0".

Edits (anchored, idempotent):
  evalsrv/dueling.py     confirmation_stamp learns rule="per_duel";
                         duel_params.confirmation_required stamped
  affine/validator.py    _confirm_crown(); called after the crown bar in
                         crown_mode "duel" when the knob is on
  affine/config.py       DuelCfg.confirmation_required
  affine/dash/readers.py + dashboard.py pass verdict.confirmation to the site
  website/app.js         "confirmation slice" card on the duel view
"""

from pathlib import Path

REPO = Path("/home/const/subnet120")


def sub(path: Path, old: str, new: str, marker: str) -> None:
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already has {marker!r}")
        return
    if s.count(old) != 1:
        raise SystemExit(f"{path.name}: anchor not unique/missing: {old[:70]!r}")
    path.write_text(s.replace(old, new))
    print(f"{path.name}: patched ({marker})")


du = REPO / "affine/evalsrv/dueling.py"
sub(du, '''    stamp = {"challenge_id": confirm.get("challenge_id"),
             "slice_index": int(confirm.get("slice_index", 1)),
             "base": {"n": n1, "margin": m1, "se": se1},
             "slice": own, "pooled": None, "passed": False}
    ok = (n1 > 0 and isinstance(m1, (int, float)) and isinstance(se1, (int, float))
          and own["margin"] is not None and own["se"] is not None
          and result.n_paired_turns > 0 and own["rejection_reason"] is None)
    if ok:
        N, M, se, z = pooled_margin_stats(n1, float(m1), float(se1),
                                          result.n_paired_turns, result.margin,
                                          result.se)
        stamp["pooled"] = {"n": N, "margin": M, "se": se,
                           "z": z if math.isfinite(z) else None}
        stamp["passed"] = bool(M > 0.0)
    return stamp
''', '''    rule = str(confirm.get("rule") or "window_best")
    stamp = {"challenge_id": confirm.get("challenge_id"),
             "slice_index": int(confirm.get("slice_index", 1)),
             "rule": rule,
             "base": {"n": n1, "margin": m1, "se": se1},
             "slice": own, "pooled": None, "passed": False}
    ok = (n1 > 0 and isinstance(m1, (int, float)) and isinstance(se1, (int, float))
          and own["margin"] is not None and own["se"] is not None
          and result.n_paired_turns > 0 and own["rejection_reason"] is None)
    if ok:
        N, M, se, z = pooled_margin_stats(n1, float(m1), float(se1),
                                          result.n_paired_turns, result.margin,
                                          result.se)
        stamp["pooled"] = {"n": N, "margin": M, "se": se,
                           "z": z if math.isfinite(z) else None}
        if rule == "per_duel":
            # wvk 19 (2026-09-16): a crown must win twice — this slice's own
            # margin > 0 AND the pooled margin clears the same bar the first
            # slice cleared, max(k_sigma·SE_pooled, δ).
            k_sigma = float(confirm.get("k_sigma", 2.0))
            delta = float(confirm.get("min_margin", 0.0))
            bar = max(k_sigma * se, delta)
            stamp["bar"] = bar
            stamp["passed"] = bool(result.margin > 0.0 and M > bar)
        else:
            stamp["passed"] = bool(M > 0.0)
    return stamp
''', 'rule == "per_duel"')
sub(du, '''            # wvk 18: prose reply at a tool_call turn = `text` action.
            "text_fallback_at_tool_turns": text_fallback,
''', '''            # wvk 18: prose reply at a tool_call turn = `text` action.
            "text_fallback_at_tool_turns": text_fallback,
            # wvk 19: a first-slice pass needs a confirmation slice to crown.
            "confirmation_required": bool(duel_cfg.get("confirmation_required", False)),
''', '"confirmation_required": bool(duel_cfg.get(')

cfg = REPO / "affine/affine/config.py"
sub(cfg, '''    text_fallback_at_tool_turns: bool = False
''', '''    text_fallback_at_tool_turns: bool = False
    # wvk 19 (2026-09-16): a duel that clears the crown bar is confirmed on a
    # second independent n_turns slice (own margin > 0 AND pooled margin >
    # max(k_sigma·SE_pooled, δ)) before it crowns; a failed confirmation is
    # a loss ("confirmation_failed"). False = pre-wvk-19 (crown at once).
    confirmation_required: bool = False
''', "confirmation_required: bool = False")
sub(cfg, '''        text_fallback_at_tool_turns=bool(d.get("text_fallback_at_tool_turns", False)),
''', '''        text_fallback_at_tool_turns=bool(d.get("text_fallback_at_tool_turns", False)),
        confirmation_required=bool(d.get("confirmation_required", False)),
''', 'confirmation_required=bool(d.get(')

va = REPO / "affine/affine/validator.py"
sub(va, '''        accepted = bool(verdict.get("challenger_wins"))
        crowned_entry = entry
        if accepted and is_r2_ref(entry.repo):
''', '''        if verdict.get("challenger_wins") and self.cfg.duel.confirmation_required:
            # wvk 19: a first-slice pass is a candidate, not a crown. One more
            # independent slice must agree (own margin > 0, pooled margin over
            # the bar). A failed confirmation is a plain loss: the slot is
            # consumed, the king stands, nothing is re-queued.
            conf = await self._confirm_crown(entry, king, verdict, block_hash,
                                             margin, info)
            verdict["confirmation"] = conf
            if not conf.get("passed"):
                verdict["challenger_wins"] = False
                verdict["rejection_reason"] = "confirmation_failed"
        accepted = bool(verdict.get("challenger_wins"))
        crowned_entry = entry
        if accepted and is_r2_ref(entry.repo):
''', "await self._confirm_crown(")
sub(va, '''    def _repromote_if_private(self, member: dict) -> None:
''', '''    async def _confirm_crown(self, entry: QueueEntry, king: King, first: dict,
                             block_hash: str, margin: dict, info) -> dict:
        """wvk 19 confirmation slice for a first-slice crown pass.

        Same king and challenger (engines are warm), seed
        blake2b(block_hash ‖ hotkey ‖ "|slice<k>") with k = number of slices
        the first verdict scored (1; 2 if the near-miss rule pooled), turns
        disjoint from those, fresh teacher references. The pod pools the two
        samples exactly (pooled_margin_stats) and applies rule "per_duel":
        passed = own margin > 0 AND pooled margin > max(k_sigma·SE_pooled, δ).
        Returns the flat stamp {seed, n, margin, se, z, pooled_margin,
        pooled_se, pooled_z, bar, passed, ...} plus the pod's sub-blocks.
        Infra faults propagate (the whole challenge is requeued as infra)."""
        cid = entry.challenge_id
        nm = first.get("near_miss") or {}
        n_slices = len(nm.get("slices") or []) or 1
        confirm = {"challenge_id": cid, "slice_index": n_slices, "rule": "per_duel",
                   "k_sigma": float(self.cfg.duel.k_sigma),
                   "min_margin": float(margin["min_margin_effective"]),
                   "base": {"n": int(first.get("n_paired_turns") or 0),
                            "margin": first.get("margin"), "se": first.get("se")}}
        self.state.current_eval = {
            "challenge_id": f"{cid} (confirmation slice)", "repo": entry.repo,
            "hotkey": entry.hotkey, "stage": "dispatching", "progress": {},
            "started_at": now_iso(),
        }
        self.state.set_phase("confirmation", challenge_id=cid)
        self.dashboard.flush(force=True)
        log.info("%s: first slice cleared the bar (margin=%s z=%s) — running the "
                 "confirmation slice", cid, first.get("margin"), first.get("z"))

        def on_progress(data: dict) -> None:
            self.watchdog.beat()
            if self.state.current_eval is not None:
                self.state.current_eval["stage"] = data.get("phase", "scoring")
                self.state.current_eval["progress"] = data
            self.dashboard.flush()

        verdict = await self.eval_client.run_duel(
            king_repo=king.repo, king_revision=king.revision,
            challenger_repo=entry.repo, challenger_revision=entry.revision,
            challenger_hotkey=entry.hotkey, block_hash=block_hash,
            challenger_weight_bytes=info.total_safetensors_bytes,
            margin=margin, confirm=confirm, on_progress=on_progress)
        self.state.current_eval = None
        pod = dict(verdict.get("confirmation") or {})
        sl = pod.get("slice") or {}
        pooled = pod.get("pooled") or {}
        conf = {
            "required": True, "rule": "per_duel",
            "seed": sl.get("seed"), "n": sl.get("n_paired_turns"),
            "n_forfeit_turns": sl.get("n_forfeit_turns"), "digest": sl.get("digest"),
            "margin": sl.get("margin"), "se": sl.get("se"), "z": sl.get("z"),
            "pooled_n": pooled.get("n"), "pooled_margin": pooled.get("margin"),
            "pooled_se": pooled.get("se"), "pooled_z": pooled.get("z"),
            "bar": pod.get("bar"), "k_sigma": confirm["k_sigma"],
            "min_margin": confirm["min_margin"],
            "passed": bool(pod.get("passed")),
            "rejection_reason_on_slice": (sl.get("rejection_reason")
                                          or verdict.get("rejection_reason")),
            "job_id": verdict.get("job_id"), "base": pod.get("base"),
        }
        if not pod:
            conf["passed"] = False
            conf["error"] = ("pod returned no confirmation stamp (stale eval pod? "
                             "redeploy scripts/redeploy_pods.py)")
            log.error("confirmation of %s: %s", cid, conf["error"])
        if conf["rejection_reason_on_slice"]:
            conf["passed"] = False
        art = QueueEntry(challenge_id=f"{cid}-confirm", hotkey=entry.hotkey,
                         repo=entry.repo, revision=entry.revision,
                         block=entry.block, queued_at="")
        verdict["confirmation_of"] = cid
        await self._publish_eval_artifact(art, verdict)
        log.info("confirmation %s: slice margin=%s z=%s | pooled margin=%s se=%s "
                 "z=%s bar=%s -> passed=%s", cid, conf["margin"], conf["z"],
                 conf["pooled_margin"], conf["pooled_se"], conf["pooled_z"],
                 conf["bar"], conf["passed"])
        return conf

    def _repromote_if_private(self, member: dict) -> None:
''', "async def _confirm_crown(")

rd = REPO / "affine/affine/dash/readers.py"
sub(rd, '''        "revoked_by": r.get("revoked_by"),
''', '''        "revoked_by": r.get("revoked_by"),
        # wvk 19: the confirmation slice of a first-slice crown pass.
        "confirmation": v.get("confirmation"),
''', '"confirmation": v.get("confirmation"),')
db = REPO / "affine/affine/dashboard.py"
sub(db, '''                "near_miss": v.get("near_miss"),
                "rejection_reason": v.get("rejection_reason"),
''', '''                "near_miss": v.get("near_miss"),
                # wvk 19 confirmation slice (flat stamp).
                "confirmation": v.get("confirmation"),
                "rejection_reason": v.get("rejection_reason"),
''', '"confirmation": v.get("confirmation"),')

app = REPO / "affine/website/app.js"
sub(app, '''      ${card("king Reason", esc(fine(kgR)), "same slice, same teacher")}
''', '''      ${card("king Reason", esc(fine(kgR)), "same slice, same teacher")}
      ${(() => {
        // wvk 19: a first-slice pass must be confirmed on a second slice.
        const c = duel.confirmation;
        if (!c || c.rule !== "per_duel") return "";
        const ok = Boolean(c.passed);
        return card("confirmation slice", esc(fine(c.margin)),
          `z = ${esc(fmtZ(c.z))} · ${esc(String(c.n ?? "—"))} paired turns · must be > 0`, passCls(c.margin != null ? Number(c.margin) > 0 : null))
          + card("pooled (both slices)", esc(fine(c.pooled_margin)),
            `z = ${esc(fmtZ(c.pooled_z))} · bar max(${esc(String(c.k_sigma ?? 2))}·SE, δ) = ${esc(fine(c.bar))} · ${ok ? "confirmed" : "not confirmed"}`, passCls(ok));
      })()}
''', "confirmation slice")
print("done")
