from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

import typer
from nacl import signing

from .config import settings
from .system import (
    ChallengeCommitment,
    ChallengeOutcome,
    ChutesClient,
    RatioSchedule,
    Sample,
    ValidatorSampler,
    build_block,
    compute_vtrust,
    duel_many_envs,
    get_env,
    load_block,
    scoreboard,
    set_weights,
    verify_samples,
)

app = typer.Typer(help="Affine subnet management CLI.")
env_app = typer.Typer(help="Environment utilities.")
app.add_typer(env_app, name="env")


def _split_envs(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_signing_key(hex_key: str) -> signing.SigningKey:
    material = bytes.fromhex(hex_key)
    if len(material) == 32:
        return signing.SigningKey(material)
    if len(material) == 64:
        return signing.SigningKey(material[:32])
    raise ValueError("Signing key must be 32 or 64 bytes in hex.")


@env_app.command("run")
def run_env(
    env: str = typer.Argument(..., help="Environment identifier."),
    challenge_id: Optional[str] = typer.Option(None, help="Override deterministic challenge id."),
    seed: Optional[int] = typer.Option(None, help="Seed used when deriving challenge id."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    env_cls = get_env(env)
    instance = env_cls()
    options = {"challenge_id": challenge_id} if challenge_id else {}
    observation, info = instance.reset(seed=seed, options=options)
    payload = {
        "env_id": env,
        "challenge_id": info["challenge_id"],
        "observation": observation.tolist() if hasattr(observation, 'tolist') else observation,
        "info": info,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        typer.echo(f"env: {env}")
        typer.echo(f"challenge_id: {info['challenge_id']}")
        typer.echo(f"observation: {observation}")
        typer.echo(f"info: {json.dumps(info, ensure_ascii=False)}")


def _duel_stream_factory(
    sampler: ValidatorSampler,
    outcomes: Mapping[str, List[ChallengeOutcome]],
    champion_uid: int,
    contender_uid: int,
    *,
    timeout: float,
) -> callable:
    def factory(env_id: str) -> Iterable[Optional[bool]]:
        def generator() -> Iterator[Optional[bool]]:
            while True:
                outcome = sampler.sample(env_id, champion_uid=champion_uid, contender_uid=contender_uid, timeout=timeout)
                outcomes[env_id].append(outcome)
                if outcome.winner == "contender":
                    yield True
                elif outcome.winner == "champion":
                    yield False
                else:
                    yield None

        return generator()

    return factory


@app.command()
def duel(
    contender: int = typer.Option(..., "--cont", help="Contender miner UID."),
    champion: int = typer.Option(..., "--champ", help="Champion miner UID."),
    envs: str = typer.Option("tictactoe-v0,mult8-v0", help="Comma-separated environment ids."),
    max_trials: int = typer.Option(settings.max_trials, help="Wilson max trials per environment."),
    epsilon: float = typer.Option(settings.epsilon, help="Per-env ratio to beat (0.5 + epsilon)."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    env_list = _split_envs(envs)
    chutes = ChutesClient(settings.chutes_url, api_key=settings.chutes_api_key, default_timeout=settings.chutes_timeout)
    sampler = ValidatorSampler(
        validator_hotkey=settings.validator_hotkey or "validator",
        chutes=chutes,
        epoch_anchor=settings.epoch_anchor,
        env_ids=env_list,
    )
    outcomes: Dict[str, List[ChallengeOutcome]] = {env_id: [] for env_id in env_list}
    ratio_schedule = RatioSchedule(initial=0.5 + epsilon, half_life_seconds=settings.ratio_decay_seconds)

    factory = _duel_stream_factory(sampler, outcomes, champion_uid=champion, contender_uid=contender, timeout=settings.chutes_timeout)
    result = duel_many_envs(
        contender,
        champion,
        env_list,
        stream_factory=factory,
        ratio_to_beat_global=ratio_schedule.current(),
        per_env_ratio=0.5 + epsilon,
        max_budget=max_trials,
        ratio_schedule=ratio_schedule,
    )
    chutes.close()

    if json_output:
        serializable = dict(result)
        serializable["per_env"] = {env_id: asdict(res) for env_id, res in result["per_env"].items()}  # type: ignore[index]
        typer.echo(json.dumps(serializable, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"winner_uid: {result['winner_uid']}")
        typer.echo(f"ratio_to_beat: {result['ratio_to_beat']:.4f}")
        typer.echo(f"stopped_early: {result['stopped_early']}")
        for env_id, res in result["per_env"].items():  # type: ignore[index]
            typer.echo(f"- {env_id}: outcome={res.outcome}, trials={res.trials}, ci={res.ci}")


@app.command()
def validate(
    contender: int = typer.Option(..., "--cont", help="Contender miner UID."),
    champion: int = typer.Option(..., "--champ", help="Champion miner UID."),
    envs: str = typer.Option("tictactoe-v0,mult8-v0", help="Comma-separated environment ids."),
    signing_key_hex: str = typer.Option(..., "--signing-key", help="Validator ed25519 private key (hex)."),
    epoch: int = typer.Option(0, help="Epoch identifier for commit-reveal scheduling."),
    secret_seed_hex: Optional[str] = typer.Option(None, "--secret-seed", help="Secret seed used for challenge commitments (hex)."),
    prev_hash: str = typer.Option("0" * 64, help="Previous block hash."),
    block_size: int = typer.Option(settings.block_size, help="Samples per block."),
    max_trials: int = typer.Option(settings.max_trials, help="Max trials per environment."),
    output_path: Optional[Path] = typer.Option(None, "--out", help="Write blocks as JSON to this file."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON summary."),
) -> None:
    env_list = _split_envs(envs)
    chutes = ChutesClient(settings.chutes_url, api_key=settings.chutes_api_key, default_timeout=settings.chutes_timeout)
    commitment = None
    commit_digest = None
    if secret_seed_hex:
        seed_bytes = bytes.fromhex(secret_seed_hex)
        commitment = ChallengeCommitment(settings.validator_hotkey or "validator", settings.epoch_anchor)
        commit_digest = commitment.commit(epoch, seed_bytes)
        commitment.reveal(epoch, seed_bytes)
    sampler = ValidatorSampler(
        validator_hotkey=settings.validator_hotkey or "validator",
        chutes=chutes,
        epoch_anchor=settings.epoch_anchor,
        env_ids=env_list,
        epoch=epoch,
        commitment=commitment,
    )
    outcomes: Dict[str, List[ChallengeOutcome]] = {env_id: [] for env_id in env_list}
    factory = _duel_stream_factory(sampler, outcomes, champion_uid=champion, contender_uid=contender, timeout=settings.chutes_timeout)
    signing_key = _load_signing_key(signing_key_hex)
    ratio_schedule = RatioSchedule(initial=settings.ratio_to_beat, half_life_seconds=settings.ratio_decay_seconds)

    result = duel_many_envs(
        contender,
        champion,
        env_list,
        stream_factory=factory,
        ratio_to_beat_global=ratio_schedule.current(),
        per_env_ratio=settings.ratio_to_beat,
        max_budget=max_trials,
        ratio_schedule=ratio_schedule,
    )
    chutes.close()

    # Gather samples
    all_samples: List[Sample] = []
    for env_id in env_list:
        for outcome in outcomes[env_id]:
            all_samples.extend(outcome.as_samples(settings.validator_hotkey or "validator"))

    blocks: List[Dict[str, object]] = []
    block_index = 0
    cursor = 0
    env_spec_versions = sampler.env_spec_versions()
    while cursor < len(all_samples):
        chunk = all_samples[cursor : cursor + block_size]
        block, block_digest = build_block(
            chunk,
            validator=settings.validator_hotkey or "validator",
            block_index=block_index,
            prev_hash=prev_hash,
            env_spec_versions=env_spec_versions,
            signing_key=signing_key,
        )
        prev_hash = block_digest
        block_index += 1
        cursor += len(chunk)
        blocks.append({"hash": block_digest, "block": block.canonical_dict()})

    if output_path:
        output_path.write_text(json.dumps(blocks, indent=2, ensure_ascii=False))

    verified_groups, validator_stats = verify_samples(all_samples)
    vtrust_scores = compute_vtrust(validator_stats)
    weight_scores = scoreboard(verified_groups, vtrust_scores)

    summary = {
        "duel": {
            "winner_uid": result["winner_uid"],
            "ratio_to_beat": result["ratio_to_beat"],
            "env_wins": result["env_wins"],
        },
        "blocks": blocks,
        "vtrust": vtrust_scores,
        "scores": weight_scores,
    }
    if commit_digest:
        summary["commitment"] = {"epoch": epoch, "digest": commit_digest}

    if json_output:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"winner_uid: {summary['duel']['winner_uid']}")
        typer.echo(f"blocks_emitted: {len(blocks)}")
        typer.echo(f"vtrust: {summary['vtrust']}")
        typer.echo(f"scores: {summary['scores']}")


@app.command("set-weights")
def set_weights_cmd(
    block_files: List[Path] = typer.Argument(..., help="Path(s) to block JSON files."),
    netuid: int = typer.Option(settings.netuid, help="Subnet netuid."),
    dry_run: bool = typer.Option(True, help="Dry-run weight submission."),
) -> None:
    blocks = []
    for path in block_files:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            for entry in data:
                blocks.append(load_block(entry["block"] if "block" in entry else entry))
        else:
            blocks.append(load_block(data["block"] if "block" in data else data))

    samples: List[Sample] = []
    for block in blocks:
        for sample in block.samples:
            if isinstance(sample, Sample):
                samples.append(sample)

    verified_groups, validator_stats = verify_samples(samples)
    vtrust_scores = compute_vtrust(validator_stats)
    score_map = scoreboard(verified_groups, vtrust_scores)
    result = set_weights(netuid, score_map, dry_run=dry_run)
    typer.echo(json.dumps({"scores": score_map, "vtrust": vtrust_scores, "result": result}, indent=2, ensure_ascii=False))


__all__ = ["app"]
