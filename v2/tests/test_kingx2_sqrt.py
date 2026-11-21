"""King x2 sampling and sqrt-Fisher draw, without GPU or network."""

from __future__ import annotations

import math
import random
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from sn120_v2.gpu.evaluator import samples_for_model
from sn120_v2.gpu.solo import SoloScheduler, SoloSettings, SoloWorker
from sn120_v2.irt import MASS_SCALE, SQRT_FISHER_MASS_RULE, IRTSampler, draw_from_masses, fisher_masses
from sn120_v2.judging import cap_hit_ok, check_result, decide, score_entry
from sn120_v2.params import Params
from sn120_v2.sampling import rollout_seed
from sn120_v2.sim import SimEvaluator, SimTrainer
from sn120_v2.store import LocalStore
from sn120_v2.types import CAP, FAIL, NOT_RUN, NULL_MODEL, PASS, SCORED, EntryResult
from test_stream_eval import compare  # sn120-test-stream-helper
from sn120_v2.worker import LocalRunner


def king_params(build, **changes):
    return build.params(null_samples_per_task=2, irt_draw_rule="sqrt_fisher", **changes)


def test_king_gets_two_strings_loras_one_in_both_runner_paths(build):
    n = 16
    effects = {0: [PASS] * n, 1: [PASS if i % 3 else FAIL for i in range(n)]}
    params = king_params(build, n_tasks=n, eval_chunk_tasks=8, looks=(0.5, 1.0), min_scored_tasks=2)
    ticket, streamed, baseline, seam = compare(build, effects, params=params, n_tasks=n, n_loras=1)
    assert len(streamed.null_outcomes) == 2 and len(streamed.entries[0].outcomes) == 1
    assert streamed == baseline
    check_result(ticket, streamed)


def test_simulator_uses_two_king_strings_and_one_lora_string(build, tmp_path):
    params = king_params(build, n_tasks=4)
    ticket = build.ticket(1, 4, params)
    lora = build.artifact(ticket.head, ticket.entries[0], params)
    store = LocalStore(tmp_path / "store")
    store.put(f"{lora.prefix}/adapter_config.json", b'{"quality": 0.0, "cap_rate": 0.0}')
    from sn120_v2.types import EvalRequest
    request = EvalRequest(ticket.round_id, ticket.head, (lora,), ticket.draw.task_ids, ticket.draw.round_seed, 0, params, 0, 1)
    outcomes = SimEvaluator(store).evaluate(request).outcomes
    assert len(outcomes[NULL_MODEL]) == 2 and len(outcomes[lora.lora_hash]) == 1
    assert all(len(text) == 4 for strings in outcomes.values() for text in strings)


def test_solo_scheduler_runs_both_king_samples_and_averages_early_looks(build):
    params = king_params(build, n_tasks=8, n_gpus=2, eval_chunk_tasks=4, looks=(0.5, 1.0))
    ticket = build.ticket(1, 8, params)
    runner = SimpleNamespace(settings=SoloSettings(), evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 5000)
    scheduler.add_lora(0, build.artifact(ticket.head, ticket.entries[0], params))
    scheduler.training_finished()
    for _ in range(8):
        claim = scheduler.steal(0, NULL_MODEL)
        if claim is None:
            break
        model, start, stop = claim
        for index in range(start, stop):
            for sample in range(params.null_samples if model == NULL_MODEL else params.samples_per_task):
                scheduler.result(model, sample, index, PASS)
    null, scored, dropped, stopped = scheduler.finish()
    assert len(null) == 2 and len(scored[0]) == 1 and not dropped and not stopped
    assert samples_for_model(params, NULL_MODEL) == 2 and samples_for_model(params, ticket.entries[0].batch.batch_sha256) == 1


def test_cap_rate_and_gate_use_each_models_own_denominator():
    params = Params(null_samples_per_task=2, cap_hit_floor=0.33, cap_hit_margin=0.02)
    # A 50% king cap rate is 2 / (2 tasks * 2 samples), not 2 / 2.
    assert cap_hit_ok(lora_caps=1, null_caps=2, rollouts=2, params=params, null_rollouts=4)
    score = score_entry((1.0, 1.0), (PASS + CAP, FAIL + CAP), (PASS + CAP,), params)
    assert score.null_cap_rate == score.cap_rate == 0.5
    assert not any("cap-hit" in reason for reason in score.reasons)


def test_result_shape_is_exact_for_king_and_loras(build):
    ticket = build.ticket(1, 4, king_params(build, n_tasks=4))
    result = build.result(ticket, ("PPPP", "FFFF"), {0: "PPPP"})
    check_result(ticket, result)
    for bad in (build.replace(result, null_outcomes=("PPPP",)), build.replace(result, null_outcomes=("PPPP", "FFFF", "PPPP")),
                build.replace(result, entries=(build.replace(result.entries[0], outcomes=("PPPP", "PPPP")),))):
        with pytest.raises(Exception, match="outcome string"):
            check_result(ticket, bad)


def test_king_mean_and_se_match_hand_calculation_and_early_look(build):
    params = king_params(build, n_tasks=4)
    null = ("PFPF", "FFFF")
    lora = ("PPPP",)
    # d = [.5, 1, .5, 1], delta = .75, se = sd(d)/2.
    d = [0.5, 1.0, 0.5, 1.0]
    delta = sum(d) / len(d)
    se = math.sqrt(sum((value - delta) ** 2 for value in d) / (len(d) - 1) / len(d))
    score = score_entry((1.0,) * 4, null, lora, params, n_total=4)
    assert (score.delta, score.se) == (pytest.approx(delta), pytest.approx(se))
    assert score.null_pass_rate == pytest.approx(sum(text.count(PASS) for text in null) / 8)
    ticket = build.ticket(1, 4, params)
    result = build.result(ticket, null, {0: lora})
    check_result(ticket, result)
    assert decide([(ticket, result)]).entries[0].score == score
    early = king_params(build, n_tasks=4, looks=(0.5, 1.0), min_scored_tasks=2)
    ticket = build.ticket(1, 4, early)
    # Both king strings matter at the exact look prefix, not only at settle.
    partial = build.result(ticket, ("PP--", "PF--"), {0: build.dropped(ticket, 0, "FF--", 0)})
    check_result(ticket, partial)


def test_sample_zero_is_paired_and_second_king_seed_is_independent():
    params = Params(null_samples_per_task=2)
    ticket_seed = "round"
    first = rollout_seed(ticket_seed, "task", 0)
    second = rollout_seed(ticket_seed, "task", 1)
    assert first == rollout_seed(ticket_seed, "task", 0)
    assert second != first and second == rollout_seed(ticket_seed, "task", 1)


def test_irt_learns_from_both_king_rollouts(build):
    head = build.head()
    pool = [f"task:{i}" for i in range(10)]
    sampler = IRTSampler(pool, king_params(build))
    draw = draw_from_masses({task: 1 for task in pool}, sampler.snapshot_id(), "seed", len(pool))
    outcomes = {NULL_MODEL: (PASS * len(pool), FAIL * len(pool)), "lora": (PASS * len(pool),)}
    sampler.update(head, draw, outcomes)
    row = sampler.public_snapshot()["heads"][head.version_id]
    assert (row["rollouts"], row["passes"]) == (2 * len(pool), len(pool))
    assert sampler.public_snapshot()["loras"]["lora"]["rollouts"] == len(pool)


def test_sqrt_fisher_masses_and_restore_migration_keep_old_fisher_unchanged(build):
    tasks = {"easy": [-3.0, 0.25, 0], "middle": [0.0, 0.25, 0], "hard": [3.0, 0.25, 0]}
    old = fisher_masses(tasks, 0.0, 0.1)
    new = fisher_masses(tasks, 0.0, 0.1, sqrt=True)
    assert old == fisher_masses(tasks, 0.0, 0.1, sqrt=False)
    assert new["easy"] > old["easy"] and new["hard"] > old["hard"] and new["middle"] < old["middle"]
    for task, row in tasks.items():
        p = 1.0 / (1.0 + math.exp(row[0]))
        q = math.sqrt(p * (1.0 - p))
        share = 0.9 * q / sum(math.sqrt(1.0 / (1.0 + math.exp(other[0])) * (1.0 - 1.0 / (1.0 + math.exp(other[0])))) for other in tasks.values()) + 0.1 / len(tasks)
        assert new[task] == max(1, round(share * MASS_SCALE))
    sampler = IRTSampler(list(tasks), Params(irt_draw_rule="fisher"), initial=tasks)
    before = sampler.public_snapshot()
    same = IRTSampler(list(tasks), Params(irt_draw_rule="fisher"), snapshot=before).public_snapshot()
    assert same == before
    migrated = IRTSampler(list(tasks), Params(irt_draw_rule="sqrt_fisher"), snapshot=before).public_snapshot()
    assert migrated["config"]["draw_rule"] == "sqrt_fisher"
    assert migrated["mass_rule"] == SQRT_FISHER_MASS_RULE
    assert {task: row[3] for task, row in migrated["tasks"].items()} == new
    assert all(migrated["tasks"][task][:3] == before["tasks"][task][:3] for task in tasks)


def test_solo_worker_submits_two_king_requests_and_one_lora_request(build):
    params = king_params(build, n_tasks=4, n_gpus=1, looks=(1.0,), eval_chunk_tasks=1)
    ticket = build.ticket(1, 4, params)
    evaluator = SimpleNamespace(http=None, _solo_clients=8, settings=SimpleNamespace(clients_per_server=1), clients_for=lambda params: 8,
                                _solo_servers={0: SimpleNamespace(alive=lambda: True)}, _serve_one=lambda *args: "http://gpu0",
                                _load_one=lambda *args: None, _grade=lambda tasks, jobs, replies, request: {job: PASS for job in jobs})
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=1), evaluator=evaluator)
    scheduler = SoloScheduler(runner, ticket, 10000)
    evaluator._solo_scheduler = scheduler
    lora = build.artifact(ticket.head, ticket.entries[0], params)
    scheduler.add_lora(0, lora, gpu=0)
    scheduler.training_finished()
    requests = []

    def rollout(request, url, job, written):
        requests.append(job)
        return PASS, False

    evaluator._rollout = rollout
    SoloWorker(evaluator, ticket, scheduler.trained, 10000, 0).run(steal=scheduler.steal, result=scheduler.result,
                                                                    judge=lambda: None, alive=lambda: None, deadline=10000)
    assert Counter((model, sample) for model, sample, _ in requests) == {**{(NULL_MODEL, sample): 4 for sample in range(2)}, (lora.lora_hash, 0): 4}
    null, scored, _, _ = scheduler.finish()
    assert null == (PASS * 4, PASS * 4) and scored[0] == (PASS * 4,)


def test_solo_cancels_blocked_king_requests_after_the_last_lora_drops(build):
    params = king_params(build, n_tasks=8, n_gpus=1, looks=(0.5, 1.0), eval_chunk_tasks=4, min_scored_tasks=2)
    ticket = build.ticket(1, 8, params)
    cancelled = []
    king_replies = threading.Event()

    def cancel(model):
        cancelled.append(model)
        if model == NULL_MODEL:
            king_replies.set()

    http = SimpleNamespace(cancel=cancel)
    evaluator = SimpleNamespace(http=http, _solo_clients=8, settings=SimpleNamespace(clients_per_server=1), clients_for=lambda params: 8,
                                _solo_servers={0: SimpleNamespace(alive=lambda: True)}, _serve_one=lambda *args: "http://gpu0",
                                _load_one=lambda *args: None, _grade=lambda tasks, jobs, replies, request: {job: reply[0] for job, reply in zip(jobs, replies)})
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=4), evaluator=evaluator)
    scheduler = SoloScheduler(runner, ticket, ticket.issued_at + 10000)
    evaluator._solo_scheduler = scheduler
    lora = build.artifact(ticket.head, ticket.entries[0], params)
    scheduler.add_lora(0, lora, gpu=0)
    for index in range(4):
        for sample in range(params.null_samples):
            scheduler.result(NULL_MODEL, sample, index, PASS)
    # Keep the known king prefix, but hold later king HTTP requests open. The
    # LoRA can reach the early look; only unnecessary future king work blocks.
    scheduler.training_finished()
    # The first king block is already known. Start real claims after it, and
    # submit only half a block so the LoRA can reach the look while two king
    # requests per sample remain blocked in HTTP.
    scheduler.remaining[NULL_MODEL] = 4

    def steal(gpu, owner):
        claim = scheduler.steal(gpu, owner)
        return (NULL_MODEL, 4, 6) if claim == (NULL_MODEL, 4, 8) else claim

    def rollout(request, url, job, written):
        model, _, index = job
        if model == NULL_MODEL:
            assert king_replies.wait(timeout=5), "king was left blocked after the final LoRA drop"
        return PASS if model == NULL_MODEL else (PASS if index < 2 else FAIL), False

    evaluator._rollout = rollout
    thread = threading.Thread(target=lambda: SoloWorker(evaluator, ticket, scheduler.trained, ticket.issued_at + 10000, 0).run(
        steal=steal, result=scheduler.result, judge=scheduler.judge, alive=lambda: None, deadline=ticket.issued_at + 10000))
    thread.start()
    wait_until = threading.Event()
    while not scheduler.dropped:
        assert thread.is_alive() and not wait_until.wait(.005), "LoRA did not reach the early drop"
    assert scheduler.dropped == {0: 0}
    thread.join(timeout=1)
    assert not thread.is_alive(), "worker still waits for unneeded king generations"
    assert NULL_MODEL in cancelled and lora.lora_hash in cancelled
    null, scored, dropped, stopped = scheduler.finish()
    assert dropped == {0: 0} and not stopped
    assert null == (PASS * 4 + NOT_RUN * 4, PASS * 4 + NOT_RUN * 4) and scored[0] == (PASS * 2 + FAIL * 2 + NOT_RUN * 4,)
    king_replies.set()  # Also release a failing implementation's worker.


def test_solo_training_finish_cancels_king_after_an_earlier_final_drop(build):
    params = king_params(build, n_tasks=8, looks=(0.5, 1.0), min_scored_tasks=2)
    ticket = build.ticket(1, 8, params)
    cancelled = []
    runner = SimpleNamespace(settings=SoloSettings(), evaluator=SimpleNamespace(http=SimpleNamespace(cancel=cancelled.append)))
    scheduler = SoloScheduler(runner, ticket, ticket.issued_at + 10000)
    lora = build.artifact(ticket.head, ticket.entries[0], params)
    scheduler.add_lora(0, lora)
    for index in range(4):
        for sample in range(ticket.params.null_samples):
            scheduler.result(NULL_MODEL, sample, index, PASS)
    for index, code in enumerate((PASS, PASS, FAIL, FAIL)):
        scheduler.result(lora.lora_hash, 0, index, code)
    assert scheduler.dropped == {0: 0} and cancelled == [lora.lora_hash]
    scheduler.training_finished()
    assert cancelled == [lora.lora_hash, NULL_MODEL]


def test_public_gate_rollouts_keep_both_king_samples(build, tmp_path):
    from sn120_v2.rollouts import published_rows
    params = king_params(build, n_tasks=4)
    ticket = build.ticket(1, 4, params)
    result = build.result(ticket, (PASS * 4, FAIL * 4), {})
    capture = tmp_path / "king.jsonl"
    with capture.open("w") as handle:
        for sample in range(2):
            for task_pos in range(4):
                handle.write(__import__("json").dumps({"sample_index": sample, "task_pos": task_pos, "outcome": result.null_outcomes[sample][task_pos]}) + "\n")
    rows = list(published_rows(capture, ticket, result, "head", NULL_MODEL, None))
    assert [(row["sample_index"], row["task_pos"], row["outcome"]) for row in rows] == [(sample, pos, result.null_outcomes[sample][pos]) for sample in range(2) for pos in range(4)]


def test_fake_evaluator_se_falls_near_the_independent_noise_ratio(build):
    rng = random.Random(119)
    old_se, new_se = [], []
    for _ in range(400):
        null0, null1, lora = [], [], []
        for _ in range(500):
            # Independent binomial noise with the same variance for king/LoRA.
            old = rng.random() < 0.5
            null0.append(PASS if old else FAIL)
            null1.append(PASS if rng.random() < 0.5 else FAIL)
            lora.append(PASS if rng.random() < 0.6 else FAIL)
        one = score_entry((1.0,) * 500, ("".join(null0),), ("".join(lora),), build.params(n_tasks=500))
        two = score_entry((1.0,) * 500, ("".join(null0), "".join(null1)), ("".join(lora),), king_params(build, n_tasks=500))
        old_se.append(one.se)
        new_se.append(two.se)
    mean = lambda values: sum(values) / len(values)
    ratio = mean(new_se) / mean(old_se)
    assert 0.82 < ratio < 0.92, ratio
