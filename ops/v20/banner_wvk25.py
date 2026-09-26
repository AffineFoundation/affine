"""Dashboard banner for the wvk-25 notice (#fork-notice; --remove at T0). Idempotent."""
from __future__ import annotations
import argparse, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
HTML = REPO / "affine" / "website" / "index.html"
BANNER = '''    <!-- wvk 25 fork notice (posted 2026-09-26; REMOVE AT T0 2026-09-30 14:00 UTC:
         python ops/v20/banner_wvk25.py --remove). Static so the notice shows even
         when the dashboard API is down. -->
    <div class="fork-notice" id="fork-notice" role="note">
      <div class="fork-notice-inner">
        <span class="fork-notice-tag">FORK NOTICE</span>
        <span class="fork-notice-text">
          Upcoming fork wvk 25 — <b>Wed 2026-09-30 14:00 UTC</b>: teacher →
          <code>GLM-5.3-Flash</code>, 262k context, miner empty-thought rule,
          sequential stopping, R cap. Reign 21 stands. Miners: serve
          <code>--max-model-len 262144</code>.
        </span>
        <a class="fork-notice-link" href="./llms.txt">spec: llms.txt § Upcoming fork: wvk 25 →</a>
      </div>
    </div>
'''
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--remove", action="store_true"); a = ap.parse_args()
    s = HTML.read_text()
    if a.remove:
        s2 = re.sub(r"    <!-- wvk 25 fork notice.*?</div>\n    </div>\n", "", s, flags=re.S)
        if s2 == s: print("no banner to remove"); return 0
        HTML.write_text(s2); print("banner removed"); return 0
    if 'id="fork-notice"' in s: print("already"); return 0
    anchor = "  </header>\n"
    if s.count(anchor) != 1: raise SystemExit("header anchor")
    HTML.write_text(s.replace(anchor, BANNER + anchor)); print("banner added"); return 0
if __name__ == "__main__":
    raise SystemExit(main())
