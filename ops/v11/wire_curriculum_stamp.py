"""Additive verdict stamp for the adaptive curriculum (plan §3.4). Idempotent.

  evalsrv/corpus.py   CorpusSync.info() returns the manifest's `curriculum`
                      block (None when the manifest has none).
  evalsrv/dueling.py  draw_slice stamps slice.curriculum_version =
                      "v<rule_version>@<weights_sha256[:12]>" and
                      slice.curriculum = {rule_version, mode, ledger_sha256,
                      weights_sha256, manifest_sha256} when the block exists.
No scoring change, no wvk. Pre-curriculum manifests stamp nothing new.
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


co = REPO / "affine/evalsrv/corpus.py"
sub(co, '''            "synced_at": self.synced_at,
            "stale": self.stale,
            "ready": self.ready,
        }
''', '''            "synced_at": self.synced_at,
            "stale": self.stale,
            "ready": self.ready,
            # Adaptive curriculum (2026-09-14, telemetry): the manifest's
            # `curriculum` block — {rule_version, mode, ledger_sha256,
            # weights_sha256, knobs} — or None for a manifest without one.
            "curriculum": self.curriculum_block(),
        }

    def curriculum_block(self) -> dict | None:
        """The corpus manifest's `curriculum` block, or None. Read-only: the
        fold writes it; the pod stamps it on every verdict slice so a
        reader can tell which curriculum weights drew the slice."""
        block = (self.manifest or {}).get("curriculum")
        return dict(block) if isinstance(block, dict) and block else None
''', "def curriculum_block(")

du = REPO / "affine/evalsrv/dueling.py"
sub(du, '''        if corpus_info.get("view_spec"):
            info["view_spec"] = str(corpus_info["view_spec"])
            info["corpus_base_url"] = str(corpus_info.get("corpus_base_url", ""))
        return turns, info
''', '''        if corpus_info.get("view_spec"):
            info["view_spec"] = str(corpus_info["view_spec"])
            info["corpus_base_url"] = str(corpus_info.get("corpus_base_url", ""))
        # Adaptive curriculum stamp (2026-09-14, additive telemetry): which
        # curriculum rule / weights the manifest was folded under. Absent
        # on manifests without a `curriculum` block (every epoch so far).
        cur = corpus_info.get("curriculum")
        if isinstance(cur, dict) and cur:
            wsha = str(cur.get("weights_sha256") or "")
            info["curriculum_version"] = f"v{cur.get('rule_version', 0)}@{wsha[:12]}"
            info["curriculum"] = {
                "rule_version": cur.get("rule_version"),
                "mode": cur.get("mode"),
                "ledger_sha256": cur.get("ledger_sha256"),
                "weights_sha256": cur.get("weights_sha256") or None,
                "manifest_sha256": info["manifest_sha256"],
            }
        return turns, info
''', 'info["curriculum_version"]')
print("done")
