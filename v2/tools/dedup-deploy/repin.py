#!/usr/bin/env python3
"""Re-pin this deploy kit to the exact stacked production checkout.

Run from the final stacked worktree, not from a production checkout:

    python -I v2/tools/dedup-deploy/repin.py FINAL_COMMIT_SHA

The checkout must be at that commit and contain the complete crown-gate and
prompt-template changes. Committed Params defaults are the source of their
values; no gate/template value is guessed or overwritten by this tool.
Only the reviewed dedup release overrides and admission floors below are
required. The committed release's crown thresholds are never replaced.

The command updates new-params.json, expect.json, the exact historical
params/code pairs in rollback-compat/params.py, the preserved raw incident
params record, and manifest.json. It then compiles the kit, checks the old-code
translation and manifest, and runs the offline frozen-input tests.
Use --python for the same reviewed test environment as the release checks.
Frozen-input tests use the local copied journal and fake transports; this tool
does not access production, R2, chain state or any service.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

KIT = Path(__file__).resolve().parent
V2 = KIT.parents[1]
ROOT = V2.parent
PARAMS = KIT / "rollback-compat/params.py"
DEDUP_FIELDS = {
    "train_seed_scope": "round",
    "dedup_enabled": True,
    "dedup_rollout_overlap": 0.5,
    "dedup_cos_shared_seed": 0.135,
    "dedup_cos_own_seed": 0.002,
    "min_signal_rollouts": 1,
    "min_step_norm": 0.0,
    "eval_merged": True,
}
REQUIRED = {"batch_cap": 6, "dedup_enabled": True, "eval_merged": True,
            "min_step_norm": 0.0, "min_signal_rollouts": 1, "train_seed_scope": "round"}
OLD_COMMIT = "c525286580c84d795b3ca7cbdf2302e96fce31ba"
OLD_PIN = "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a"
OLD_DIGEST = "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"
FULL_FOLD_COMMIT = "71d287f61338596ee24ad1dca654f8cf2cafe6e7"
REPAIRED_COMMIT = "14ae80f538448a203481de6f5a3304965cd3f334"
REPAIRED_PIN = "8caddd9c6fc276a9669dff9e01e1bb7e437d98bed5b12a605141e6025b9f9ec4"
REPAIRED_DIGEST = "2f7fd4ecaa8fa26800d8a3f54063d6286d614e309fcdffd9af59bfffeb7824fa"
COMBINED_PIN = "4c83d52f7489f8eae22e550e226edc17fb6f2f0d9f6dce5a4d333369e5b1a3be"
COMBINED_DIGEST = "07e946950889a1dfc9d8e59ce6b98fd955e2729b6ecf3b25912bf5660f556fda"
INCIDENT_COMMIT = "50afaca32ea982796728a0b07cb6e3c579ef378b"
INCIDENT_PIN = "d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c"
INCIDENT_DIGEST = "28f1fb8bc7bd4e415a4f7a62c5876d99a4404bee7b4afd156a0290b81eb6510a"
SHA = re.compile(r"[0-9a-f]{40}")
ROUND = re.compile(r"r[0-9]{6}")
PIN_NAME = re.compile(r"(?:HISTORICAL|REVIEWED|FINAL|STACKED_[1-9][0-9]*)_NEW_PARAMS_HASH")


def load_source(name: str, path: Path, *, package: bool = False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest_files(directory: Path, canonical) -> str:
    files = {path.name: canonical.file_sha256(path) for path in sorted(directory.glob("*.py"))}
    if not files:
        raise ValueError(f"no Python files in {directory}")
    return canonical.manifest_digest(files)


def read_pins(text: str) -> dict[str, tuple[str, str]]:
    names, digests = {}, {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if PIN_NAME.fullmatch(name):
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError(f"invalid accepted params pin {name}")
                names[value] = name
            elif re.fullmatch(r"(?:HISTORICAL|REVIEWED|FINAL|STACKED_[1-9][0-9]*)_NEW_CODE_DIGEST", name):
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError(f"invalid accepted code digest {name}")
                digests[name.removesuffix("_CODE_DIGEST") + "_PARAMS_HASH"] = value
    if set(names.values()) != set(digests):
        raise ValueError("every accepted params pin must have exactly one named paired code digest")
    return {pin: (name, digests[name]) for pin, name in names.items()}


def render_pins(text: str, pins: dict[str, tuple[str, str]], added: set[str]) -> str:
    lines = ["# BEGIN GENERATED RELEASE PINS (repin.py)"]
    lines.append("DEDUP_FIELDS = {")
    lines.extend(f"    {name!r}: {value!r}," for name, value in DEDUP_FIELDS.items())
    lines.append("}")
    lines.append('REVIEWED_OLD_PARAMS_HASH = "bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a"')
    lines.append('REVIEWED_OLD_CODE_DIGEST = "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"')
    lines.append("RELEASE_FIELDS = DEDUP_FIELDS | {'z_bar': 1.5}")
    for pin, (name, code_digest) in pins.items():
        prefix = name.removesuffix("_NEW_PARAMS_HASH")
        lines.append(f'{name} = "{pin}"')
        lines.append(f'{prefix}_NEW_CODE_DIGEST = "{code_digest}"')
    lines.append("REVIEWED_NEW_PARAMS_HASHES = frozenset((")
    lines.extend(f"    {name}," for name, _ in pins.values())
    lines.append("))")
    lines.append("REVIEWED_NEW_CODE_DIGESTS = {")
    lines.extend(f"    {name}: {name.removesuffix('_PARAMS_HASH')}_CODE_DIGEST," for name, _ in pins.values())
    lines.append("}")
    lines.append(f"NEW_PARAMS_ONLY_FIELDS = frozenset({tuple(sorted(added - set(DEDUP_FIELDS)))!r})"
                 if added - set(DEDUP_FIELDS) else "NEW_PARAMS_ONLY_FIELDS = frozenset()")
    lines.append("# END GENERATED RELEASE PINS (repin.py)")
    pattern = re.compile(
        r"# BEGIN GENERATED RELEASE PINS \(repin\.py\)\n.*?"
        r"# END GENERATED RELEASE PINS \(repin\.py\)", re.DOTALL,
    )
    replaced, count = pattern.subn("\n".join(lines), text)
    if count != 1:
        raise ValueError("cannot locate exactly one generated accepted-pin block")
    return replaced


def incident_params(canonical) -> tuple[dict, str]:
    """Read the preserved incident record; never infer its historical fields."""
    path = KIT / "incident_new-params.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("recipe"), dict):
        raise ValueError("incident_new-params.json must contain a full Params object and recipe")
    if data["recipe"].get("code_digest") != INCIDENT_DIGEST or canonical.digest(data) != INCIDENT_PIN:
        raise ValueError("incident_new-params.json is not the exact historical d5979c30/code pin")
    return data, canonical.digest(data)


def repaired_params(canonical) -> dict:
    """Recover the exact first repaired release from its immutable Git record."""
    raw = subprocess.run(
        ["git", "show", f"{REPAIRED_COMMIT}:v2/tools/dedup-deploy/new-params.json"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(raw)
    preserved = KIT / "repaired_new-params.json"
    if preserved.exists() and json.loads(preserved.read_text(encoding="utf-8")) != data:
        raise ValueError("preserved repaired params differ from the reviewed 14ae80f5 record")
    if (not isinstance(data, dict) or not isinstance(data.get("recipe"), dict)
            or data["recipe"].get("code_digest") != REPAIRED_DIGEST
            or canonical.digest(data) != REPAIRED_PIN):
        raise ValueError("the exact 14ae80f5 repaired params/code pin changed")
    return data


def git_digest(commit: str, canonical) -> str:
    files = {}
    names = subprocess.run(["git", "ls-tree", "--name-only", f"{commit}:v2/sn120_v2/gpu"],
                           cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    for name in names:
        if name.endswith(".py"):
            raw = subprocess.run(["git", "show", f"{commit}:v2/sn120_v2/gpu/{name}"],
                                 cwd=ROOT, check=True, capture_output=True).stdout
            files[name] = hashlib.sha256(raw).hexdigest()
    return canonical.manifest_digest(files)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_sha", help="complete 40-hex SHA of HEAD, normally the stacked production-code commit")
    parser.add_argument("--code-sha",
                        help="production code identity when final_sha is a kit-only commit; its source tree must match HEAD")
    parser.add_argument("--gate-round", default="r000093")
    parser.add_argument("--first-health-round", default="r000094")
    parser.add_argument("--python", default=sys.executable,
                        help="reviewed test interpreter; default is the interpreter running repin.py")
    parser.add_argument("--accept-new-pin", action="append", default=[], metavar="PARAMS_HASH:GPU_DIGEST",
                        help="another actually applied/reviewed new pair to retain; repeatable")
    parser.add_argument("--skip-frozen-inputs", action="store_true",
                        help="explicitly skip the default 14 offline frozen-input tests (report this)")
    args = parser.parse_args(argv)
    if not SHA.fullmatch(args.final_sha):
        parser.error("final_sha must be a complete lowercase 40-hex commit SHA")
    if args.code_sha is not None and not SHA.fullmatch(args.code_sha):
        parser.error("--code-sha must be a complete lowercase 40-hex commit SHA")
    if not ROUND.fullmatch(args.gate_round) or not ROUND.fullmatch(args.first_health_round):
        parser.error("rounds must be rNNNNNN")
    if int(args.first_health_round[1:]) != int(args.gate_round[1:]) + 1:
        parser.error("first-health-round must be gate-round + 1")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != args.final_sha:
        raise ValueError(f"checkout HEAD is {head}, not requested final SHA {args.final_sha}")
    code_sha = args.code_sha or args.final_sha
    for required in (INCIDENT_COMMIT, REPAIRED_COMMIT, FULL_FOLD_COMMIT):
        if subprocess.run(["git", "merge-base", "--is-ancestor", required, code_sha], cwd=ROOT).returncode:
            raise ValueError(f"production code {code_sha} omits required safety lineage {required}")
    for label, directory in (("production code", "v2/sn120_v2"),):
        result = subprocess.run(["git", "diff", "--exit-code", code_sha, head, "--", directory],
                                cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"{label} at HEAD differs from production code {code_sha}")
        result = subprocess.run(["git", "diff", "--exit-code", code_sha, "--", directory],
                                cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"{label} working tree differs from production code {code_sha}")
    overlay_dir = "v2/tools/dedup-deploy/rollback-compat"
    result = subprocess.run(["git", "diff", "--exit-code", code_sha, head, "--", overlay_dir],
                            cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"rollback overlay at kit HEAD differs from production code {code_sha}")
    changed = subprocess.run(["git", "diff", "--name-only", code_sha, "--", overlay_dir],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    if set(changed) - {f"{overlay_dir}/params.py"}:
        raise ValueError(f"uncommitted handwritten rollback-overlay changes are refused: {sorted(set(changed) - {f'{overlay_dir}/params.py'})}")
    overlay_path = f"{overlay_dir}/params.py"
    committed_overlay = subprocess.run(["git", "show", f"{head}:{overlay_path}"],
                                       cwd=ROOT, check=True, capture_output=True, text=True).stdout
    generated_block = re.compile(
        r"# BEGIN GENERATED RELEASE PINS \(repin\.py\)\n.*?"
        r"# END GENERATED RELEASE PINS \(repin\.py\)", re.DOTALL,
    )
    if generated_block.sub("", PARAMS.read_text(encoding="utf-8")) != generated_block.sub("", committed_overlay):
        raise ValueError("only the generated rollback-pin block may be dirty before re-pinning")
    overlay = (V2 / "sn120_v2/gpu/folder.py").read_text(encoding="utf-8")
    if "folded = in_children([FoldJob(" not in overlay:
        raise ValueError("full-mode crowned install fold is not isolated with in_children")

    # The new code and its exact old schema are loaded directly from the final
    # checkout/git object. No installed package or ambient PYTHONPATH is used.
    load_source("repin_new_package", V2 / "sn120_v2/__init__.py", package=True)
    canonical = load_source("repin_new_package.canonical", V2 / "sn120_v2/canonical.py")
    old_package = load_source("repin_old_package", KIT / "rollback-compat/__init__.py", package=True)
    # from_dict resolves these through absolute imports in the overlay source.
    sys.modules["repin_old_package.errors"] = load_source(
        "repin_old_package.errors", V2 / "sn120_v2/errors.py",
    )
    sys.modules["repin_old_package.canonical"] = load_source(
        "repin_old_package.canonical", KIT / "rollback-compat/canonical.py",
    )
    old_params_module = load_source("repin_old_package.params", PARAMS)
    old_fields = {f.name for f in dataclasses.fields(old_params_module.Params) if not f.name.startswith("_")}

    Params = load_source("repin_new_package.params", V2 / "sn120_v2/params.py").Params

    new_fields = {f.name for f in dataclasses.fields(Params) if not f.name.startswith("_")}
    missing = old_fields - new_fields
    if missing:
        raise ValueError(f"final Params removed old-schema fields; reviewed migration required: {sorted(missing)}")
    added = new_fields - old_fields
    gpu_digest = digest_files(V2 / "sn120_v2/gpu", canonical)
    # Read the historical GPU trees from git, not the final checkout.
    for commit, expected_digest in ((INCIDENT_COMMIT, INCIDENT_DIGEST), (REPAIRED_COMMIT, REPAIRED_DIGEST)):
        if git_digest(commit, canonical) != expected_digest:
            raise ValueError(f"the exact historical GPU digest changed for {commit}")

    full = json.loads((KIT / "new-params.json").read_text(encoding="utf-8"))
    if not isinstance(full, dict) or not isinstance(full.get("recipe"), dict):
        raise ValueError("new-params.json must contain a full Params object and recipe")
    unknown = set(full) - new_fields
    if unknown:
        raise ValueError(f"stale params fields absent from final Params: {sorted(unknown)}")
    full |= DEDUP_FIELDS
    for field in dataclasses.fields(Params):
        if field.name in added and field.name not in full:
            if field.default is not dataclasses.MISSING:
                full[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                full[field.name] = field.default_factory()
            else:
                raise ValueError(f"new Params field {field.name} has no committed default; set it explicitly")
    for name, value in REQUIRED.items():
        if full.get(name) != value:
            raise ValueError(f"reviewed release setting {name} must be {value!r}, got {full.get(name)!r}")
    full["recipe"] = dict(full["recipe"], code_digest=gpu_digest)
    params = Params.from_json(canonical.canonical_json(full))
    params_hash = params.params_hash
    if canonical.digest(full) != params_hash:
        raise ValueError("the full file and decoded Params canonical hashes differ")
    old_text = PARAMS.read_text(encoding="utf-8")
    # Bootstrap from the last committed overlay even on the first invocation;
    # subsequent invocations read the complete generated block below.
    bootstrap_text = subprocess.run(
        ["git", "show", "40fe054e6444a575fc9ad0a11d01631928488c81:v2/tools/dedup-deploy/rollback-compat/params.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    expected = json.loads((KIT / "expect.json").read_text(encoding="utf-8"))
    if (expected["old_commit"], expected["old_digest"], expected["old_params_hash"]) != (OLD_COMMIT, OLD_DIGEST, OLD_PIN):
        raise ValueError("re-pin supports only the reviewed c5252865 rollback identity")
    pins = read_pins(bootstrap_text if "# BEGIN GENERATED RELEASE PINS (repin.py)" not in old_text else old_text)
    for name, value in {
        "DEDUP_FIELDS": DEDUP_FIELDS,
        "RELEASE_FIELDS": DEDUP_FIELDS | {"z_bar": 1.5},
        "NEW_PARAMS_ONLY_FIELDS": frozenset(added - set(DEDUP_FIELDS)),
        "REVIEWED_NEW_PARAMS_HASHES": frozenset(pins),
        "REVIEWED_NEW_CODE_DIGESTS": {pin: code for pin, (_, code) in pins.items()},
        "REVIEWED_OLD_PARAMS_HASH": OLD_PIN,
        "REVIEWED_OLD_CODE_DIGEST": OLD_DIGEST,
    }.items():
        setattr(old_params_module, name, value)
    for pair in args.accept_new_pin:
        try:
            pin, code_digest = pair.split(":")
        except ValueError:
            parser.error("--accept-new-pin requires PARAMS_HASH:GPU_DIGEST")
        if not re.fullmatch(r"[0-9a-f]{64}", pin) or not re.fullmatch(r"[0-9a-f]{64}", code_digest):
            parser.error("--accept-new-pin requires two lowercase SHA-256 values")
        if pin in pins and pins[pin][1] != code_digest:
            raise ValueError(f"accepted params pin {pin} has conflicting GPU digests")
        if pin not in pins:
            used = {name for name, _ in pins.values()}
            number = 1
            while f"STACKED_{number}_NEW_PARAMS_HASH" in used:
                number += 1
            pins[pin] = (f"STACKED_{number}_NEW_PARAMS_HASH", code_digest)
    if expected["new_params_hash"] not in pins:
        raise ValueError("the previous final pin is missing; refusing to drop an accepted release")
    if params_hash in pins and pins[params_hash][1] != gpu_digest:
        raise ValueError(f"params pin {params_hash} is already paired with another GPU digest")
    if params_hash not in pins:
        used = {name for name, _ in pins.values()}
        number = 1
        while f"STACKED_{number}_NEW_PARAMS_HASH" in used:
            number += 1
        pins[params_hash] = (f"STACKED_{number}_NEW_PARAMS_HASH", gpu_digest)
    incident_data, incident_pin = incident_params(canonical)
    repaired_data = repaired_params(canonical)
    combined_data = json.loads(subprocess.run(
        ["git", "show", "40fe054e6444a575fc9ad0a11d01631928488c81:v2/tools/dedup-deploy/new-params.json"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout)
    required_pins = {
        INCIDENT_PIN: INCIDENT_DIGEST,
        REPAIRED_PIN: REPAIRED_DIGEST,
        COMBINED_PIN: COMBINED_DIGEST,
    }
    for pin, code_digest in required_pins.items():
        if pin not in pins or pins[pin][1] != code_digest:
            raise ValueError(f"required historical params/code pair {pin} is missing or changed")
    historical_pairs = {
        INCIDENT_PIN: incident_data,
        REPAIRED_PIN: repaired_data,
        COMBINED_PIN: combined_data,
    }
    for pin, data in historical_pairs.items():
        if canonical.digest(data) != pin:
            raise ValueError(f"historical raw params derivation changed: {pin}")
        if Params.from_json(canonical.canonical_json(data)).params_hash != pin:
            raise ValueError(
                "the final Params schema does not preserve historical hashes; "
                "new fields must use its reviewed legacy/default omission rule",
            )

    # Check the exact complete new record against the old schema before writing.
    translated = dict(full)
    for name in (*DEDUP_FIELDS, "z_bar", *added):
        translated.pop(name, None)
    translated["recipe"] = dict(full["recipe"], code_digest=expected["old_digest"])
    translated_params = old_params_module.Params.from_json(canonical.canonical_json(translated))
    if translated_params.params_hash != expected["old_params_hash"]:
        raise ValueError(
            f"old-code translation is {translated_params.params_hash}, not {expected['old_params_hash']}; "
            "do not ship a different legacy behavior by omission",
        )

    expect = dict(expected, gate_round=args.gate_round, first_health_round=args.first_health_round,
                  new_commit=code_sha, kit_commit=args.final_sha, new_digest=gpu_digest, new_params_hash=params_hash,
                  incident_commit=INCIDENT_COMMIT, incident_digest=INCIDENT_DIGEST,
                  incident_params_hash=INCIDENT_PIN, repaired_commit=REPAIRED_COMMIT,
                  repaired_digest=REPAIRED_DIGEST, repaired_params_hash=REPAIRED_PIN)
    new_text = render_pins(old_text, pins, added)
    ast.parse(new_text, filename=str(PARAMS))
    PARAMS.write_text(new_text, encoding="utf-8")
    # Load the generated decoder fresh; do not validate patched constants in
    # the pre-repin module. This is the same overlay file the old CLI installs.
    sys.modules.pop("repin_old_package.generated_params", None)
    generated_params = load_source("repin_old_package.generated_params", PARAMS)
    for data in (full, *historical_pairs.values()):
        decoded = generated_params.Params.from_dict(data)
        if decoded.params_hash != OLD_PIN or decoded.z_bar != 1.0:
            raise ValueError("an accepted release does not translate to the exact old params behavior")
        if "crown_gate" in data:
            if (data.get("crown_gate"), data.get("z_field_bar"), data.get("z_bar")) != ("field_median", 1.0, 0.0):
                raise ValueError("the field-gate release lost its committed gate parameters")
        elif data.get("z_bar") != 1.5 or "z_field_bar" in data:
            raise ValueError("a pre-field-gate release record was modified")
    for field in added - set(DEDUP_FIELDS):
        if field in full:
            changed = dict(full)
            changed[field] = {"field_median": "king", "king": "field_median"}.get(full[field], object())
            try:
                generated_params.Params.from_dict(changed)
            except Exception:
                pass
            else:
                raise ValueError(f"changed new field {field} was not refused by the old decoder")
    for damaged in (dict(full, unreviewed_field=True),
                    dict(full, recipe=dict(full["recipe"], code_digest=OLD_DIGEST))):
        try:
            generated_params.Params.from_dict(damaged)
        except Exception:
            pass
        else:
            raise ValueError("unknown fields or an unpaired code digest were accepted")
    write_json(KIT / "new-params.json", full)
    write_json(KIT / "incident_new-params.json", incident_data)
    write_json(KIT / "repaired_new-params.json", repaired_data)
    write_json(KIT / "expect.json", expect)
    manifest = json.loads((KIT / "manifest.json").read_text(encoding="utf-8"))
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((KIT / name).read_bytes()).hexdigest()
    write_json(KIT / "manifest.json", manifest)

    compile_root = KIT / ".repin-compile"
    if compile_root.exists():
        raise ValueError(f"stale compile directory {compile_root}")
    compile_root.mkdir()
    try:
        for path in KIT.rglob("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        subprocess.run([args.python, "-I", "-m", "compileall", "-q", "-f", "-b",
                        "-j", "0", str(KIT)], cwd=ROOT, check=True)
    finally:
        for path in KIT.rglob("*.pyc"):
            path.unlink()
        compile_root.rmdir()
    for name in ("autodeploy", "dry-run", "rollback-loop.sh", "restore_validator.sh", "rollback-python", "rollback-validator-weights"):
        subprocess.run(["bash", "-n", str(KIT / name)], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--check", "--", "v2/tools/dedup-deploy"], cwd=ROOT, check=True)
    manifest_check = subprocess.run(
        [args.python, "-I", "-c",
         "import importlib.util,sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location('repin_manifest_check',p+'/rollback_cli.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m._manifest(__import__('pathlib').Path(p)); print('manifest OK')",
         str(KIT)], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    if not args.skip_frozen_inputs:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([args.python, "-I", str(KIT / "test_frozen_inputs.py")], cwd=ROOT, env=env, check=True)
    print(json.dumps({
        "status": "REPIN STATIC CHECKS PASS" if args.skip_frozen_inputs else "REPIN STATIC/FROZEN-INPUT CHECKS PASS",
        "commit": head, "production_code_commit": code_sha, "gpu_digest": gpu_digest, "params_hash": params_hash,
        "new_params_sha256": hashlib.sha256((KIT / "new-params.json").read_bytes()).hexdigest(),
        "gate_round": args.gate_round, "first_health_round": args.first_health_round,
        "new_params_only_fields_dropped_for_old_code": sorted(added),
        "accepted_new_pins": {pin: code for pin, (_, code) in pins.items()},
        "manifest": manifest_check.stdout.strip(),
        "frozen_input_tests": "skipped explicitly" if args.skip_frozen_inputs else "14 passed",
        "production_or_chain_writes": 0,
        "note": "Commit the repinned kit and rerun final committed-kit qualification before deployment.",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"REPIN REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2)
