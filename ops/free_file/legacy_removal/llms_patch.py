"""RA/20261031-legacy-removal: drop old-format/old-client text from llms.txt. In-place anchored edits (other jobs patch other
sections of the same file, so the file is never replaced whole). Idempotent: an edit already applied is skipped.
usage: llms_patch.py PATH"""
import os, sys
p = sys.argv[1]; s = open(p).read()
REPS = [
("Updated 2026-10-30 (free-file protocol, Jacob). This replaces the old client guide; the previous\nversion is archived at https://affine.io/llms-legacy.txt (history only, not the contract).\n",
 "Updated 2026-10-31 (free-file protocol, Jacob). Batch files are the only mining format.\n"),
("(`batch_file`, `old_format_cutoff`, `training`)", "(`batch_file`, `training`)"),
("There is no mining client, no epochs or commitments, no TOPLOC, no log-probs, no proofs and no\nregrading. When your turn comes the validator copies your file, checks its format only, trains\n",
 "No mining software is needed and your rewards are used as given. When your turn comes the\nvalidator copies your file, checks its format only, trains\n"),
("## Old-format cutoff\n\nOld-format (client) batches are still trained until 2026-10-31 03:00 CR (UTC-6) = 2026-10-31\n09:00 UTC (unix 1793437200; mining.json `old_format_cutoff`). Until then, a queue member without\na valid batch file but with a ready old-format batch is trained the old way. After it only batch\nfiles are trained; old-format batches are ignored. The old client was removed from GitHub.\n\n", ""),
("mode 0600 (the same file the\n  old client used). Never share it", "mode 0600. Never share it"),
("takes it (until the cutoff also a member with\n  a ready old-format batch). Members", "takes it. Members"),
(" (also k1.json -> queue.grants). Old-format batches never use it.", " (also k1.json -> queue.grants)."),
("used exactly as given (no regrading).", "used exactly as given."),
]
done = 0
for a, b in REPS:
    if a in s:
        assert s.count(a) == 1, a[:60]; s = s.replace(a, b); done += 1
    elif b and b not in s:
        sys.exit('anchor missing and edit not present: ' + a[:70])
t = p + '.tmp'; open(t, 'w').write(s); os.chmod(t, 0o644); os.replace(t, p)
print('llms.txt edits applied:', done)
