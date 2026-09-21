"""Label wvk-15 (12 h window rule, retired) rows on the website history tab.

- affine/affine/dash/readers.py history_row_from_raw: pass crown_mode,
  window_id, outcome, via, revoked_reason through to the API payload.
- affine/website/app.js: outcomeBadge tags window_close / crown_revoked /
  window-era verdict rows as "retired rule (wvk 15)"; the history meta line
  explains the tag when such rows are shown.
Idempotent (refuses if already applied).
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

p = REPO / "affine/affine/dash/readers.py"
s = p.read_text()
if '"crown_mode": v.get("crown_mode")' not in s:
    old = '''        "challenger_wins": v.get("challenger_wins"),
    }


def _side_score('''
    new = '''        "challenger_wins": v.get("challenger_wins"),
        # wvk-15 era (2026-09-12 17:01 -> 2026-09-13 13:01 UTC, retired):
        # window stamps on verdicts, and the window_close / crown_revoked
        # rows themselves. The site labels these as the retired rule.
        "crown_mode": v.get("crown_mode") or r.get("crown_mode"),
        "window_id": v.get("window_id", r.get("window_id")),
        "outcome": r.get("outcome"),
        "via": r.get("via") or v.get("via"),
        "revoked_reason": r.get("revoked_reason"),
    }


def _side_score('''
    assert s.count(old) == 1, "readers anchor"
    p.write_text(s.replace(old, new))
    print("readers.py patched")

p = REPO / "affine/website/app.js"
s = p.read_text()
if "RETIRED_WINDOW_TAG" not in s:
    old = '''function outcomeBadge(r) {
  if (r.event === "crowned") return badge("crowned", `crowned #${r.reign_number ?? "?"}`);
  if (r.event === "failed") return badge("failed", r.error_code || "failed");
  if (r.accepted) return badge("accepted", "accepted");
  if (r.accepted === false) return badge("rejected", "rejected");
  return badge("queued", r.event || "event");
}
'''
    new = '''// wvk 15 (2026-09-12 17:01 -> 2026-09-13 13:01 UTC) crowned per 12 h window;
// retired by wvk 16. Its rows stay in the history as audit trail and are
// labelled so they do not read as a live rule.
const RETIRED_WINDOW_TAG = "retired rule (wvk 15)";
function isRetiredWindowRow(r) {
  return r.crown_mode === "window_best" || r.event === "window_close"
    || r.event === "crown_revoked" || r.via === "window_best";
}
function retiredTag(title) {
  return ` <span class="dim" title="${esc(title)}">${esc(RETIRED_WINDOW_TAG)}</span>`;
}

function outcomeBadge(r) {
  if (r.event === "crowned") {
    return badge("crowned", `crowned #${r.reign_number ?? "?"}`)
      + (isRetiredWindowRow(r) ? retiredTag("crowned by the 12 h window rule, retired 2026-09-13 13:01 UTC (wvk 16)") : "");
  }
  if (r.event === "crown_revoked") {
    return badge("failed", `crown revoked #${r.reign_number ?? "?"} - model copy`)
      + retiredTag(r.revoked_reason || "crown revoked");
  }
  if (r.event === "window_close") {
    return badge("queued", `window ${r.window_id ?? "?"} closed - ${r.outcome || "-"}`)
      + retiredTag("12 h window rule: every window close wrote one row. Retired 2026-09-13 13:01 UTC (wvk 16); no window runs now.");
  }
  if (r.event === "failed") return badge("failed", r.error_code || "failed");
  if (r.accepted) return badge("accepted", "accepted");
  if (r.accepted === false) {
    return badge("rejected", "rejected")
      + (isRetiredWindowRow(r) ? retiredTag("judged as a window candidate under the 12 h window rule (wvk 15), retired 2026-09-13 13:01 UTC") : "");
  }
  return badge("queued", r.event || "event");
}
'''
    assert s.count(old) == 1, "outcomeBadge anchor"
    s = s.replace(old, new)
    old = '''  const rule = liveCrownShort(cache.contract);
  const meta = $("history-meta");
  const shown = rows.length + audits.length;
  meta.textContent = rule
    ? `${rule} · ${shown} shown`
    : `${shown} shown`;
'''
    new = '''  const rule = liveCrownShort(cache.contract);
  const meta = $("history-meta");
  const shown = rows.length + audits.length;
  const retired = rows.some(isRetiredWindowRow);
  meta.textContent = (rule ? `${rule} · ${shown} shown` : `${shown} shown`)
    + (retired ? ` · rows tagged "${RETIRED_WINDOW_TAG}" were judged under the 12 h window rule (2026-09-12 17:01 -> 2026-09-13 13:01 UTC), kept for audit; no window runs now` : "");
'''
    assert s.count(old) == 1, "history meta anchor"
    p.write_text(s.replace(old, new))
    print("app.js patched")
