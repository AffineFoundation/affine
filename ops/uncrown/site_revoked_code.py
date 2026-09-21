"""Website: render `crown_revoked` rows with the row's own code/reason
instead of the hard-coded "model copy" (the reign-12 revocation is an
operator decision, not a copy). Idempotent."""

from pathlib import Path

REPO = Path("/home/const/subnet120")

p = REPO / "affine/affine/dash/readers.py"
s = p.read_text()
if '"revoked_code"' not in s:
    old = '        "revoked_reason": r.get("revoked_reason"),\n'
    assert s.count(old) == 1
    p.write_text(s.replace(old, old + '        "revoked_code": r.get("revoked_code"),\n'))
    print("readers.py patched")

p = REPO / "affine/website/app.js"
s = p.read_text()
old = '''  if (r.event === "crown_revoked") {
    return badge("failed", `crown revoked #${r.reign_number ?? "?"} - model copy`)
      + retiredTag(r.revoked_reason || "crown revoked");
  }
'''
new = '''  if (r.event === "crown_revoked") {
    const code = r.revoked_code || String(r.revoked_reason || "").split(":")[0] || "revoked";
    const label = code.replace(/^revoked_/, "").replace(/_/g, " ");
    const tag = isRetiredWindowRow(r) ? retiredTag(r.revoked_reason || "crown revoked")
      : ` <span class="dim" title="${esc(r.revoked_reason || "")}">${esc(r.revoked_by || "operator decision")}</span>`;
    return badge("failed", `crown revoked #${r.reign_number ?? "?"} - ${label}`) + tag;
  }
'''
if old in s:
    s = s.replace(old, new)
    s = s.replace('''function isRetiredWindowRow(r) {
  return r.crown_mode === "window_best" || r.event === "window_close"
    || r.event === "crown_revoked" || r.via === "window_best";
}''', '''function isRetiredWindowRow(r) {
  return r.crown_mode === "window_best" || r.event === "window_close"
    || r.via === "window_best";
}''')
    p.write_text(s)
    print("app.js patched")
else:
    print("app.js already patched or anchor missing")

p = REPO / "affine/affine/dash/readers.py"
s = p.read_text()
if '"revoked_by"' not in s:
    old = '        "revoked_code": r.get("revoked_code"),\n'
    p.write_text(s.replace(old, old + '        "revoked_by": r.get("revoked_by"),\n'))
    print("readers.py: revoked_by added")
