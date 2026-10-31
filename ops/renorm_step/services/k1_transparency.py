#!/usr/bin/env python3
"""SN120 K=1 transparency publisher: sidecar, read-only on lifecycle and gate state (Jacob t764u, 2026-10-28).

Publishes into the public website dir ONLY for epochs whose gate DECISION.json is final (green/red), so nothing is
revealed before the decision and nothing runs on the epoch's critical path:
  /k1/turns/<e>/turn.json        decision, turn, the turn's selected dispatch rows, training report (step/cap stats,
                                 loss aggregation, microbatch plan, per-rollout trajectories), candidate file hashes,
                                 trainer engine spec
  /k1/turns/<e>/docs/<slot>.json the miner's admitted batch documents, byte-exact from R2, sha256-verified
  /gate/reveal/<e>.json          gate seed, sampled task ids, commitment check, plan seed, per-task head/candidate cells,
                                 independent recomputation of the decision
  /k1/turns/index.json, /gate/reveal/index.json
Never writes lifecycle/gate state. Other miners' excluded/contested dispatch rows are summarised as counts only.
usage: k1_transparency.py [--once] [--from N]"""
import hashlib, json, os, sys, time, traceback
from pathlib import Path

ROOT = Path('/home/const/subnet120-rewrite/state')
LIFE = ROOT / 'mimo-mainnet-v1'
AUD = ROOT / 'root-audits'
GATE = Path(os.environ['HG_GATE_STATE']) / 'epochs' if os.environ.get('HG_GATE_STATE') else AUD / '20261026-promotion-gate/state/epochs'
GCODE = AUD / '20261026-promotion-gate/code'
FLAG = AUD / '20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json'
HOST_CFG = AUD / '20261027-k1-per-uid/trainer-v833/release-v833b-signed/trainer-host-v833/TRAINER-HOST.v8.3.3.ROOT-ADMITTED.private.json'
HOST_CACHE = AUD / '20261026-promotion-gate/state/trainer-host-params.json'
SRC = AUD / '20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
LIFE_CFG = AUD / '20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json'
WEB = Path(os.environ.get('K1T_WEB', '/home/const/subnet120/affine/website'))
TURNS = WEB / 'k1' / 'turns'
REVEAL = WEB / 'gate' / 'reveal'
FROM = 196
SCHEMA = 'sn120-k1-transparency-v1'
ENGINE_KEYS = ('batched_engine', 'dp_world', 'dp_token_budget', 'dp_max_rows', 'token_budget', 'max_rows', 'loss_agg_mode',
               'loss_scale', 'drop_unresolved', 'sgd_mode', 'sgd_lr', 'sgd_tensor_rel_cap', 'sgd_update_norm_cap', 'sgd_freeze',
               'optimizer_persistence', 'training_kernels', 'sgd_fp32_master', 'sgd_direction', 'support_renorm')
TRAIN_KEYS = ('version', 'status', 'input', 'input_checkpoint', 'output_checkpoint', 'config', 'loss_aggregation', 'microbatch',
              'optimizer', 'optimizer_step', 'optimizer_step_before', 'preclip_gradient_norm', 'objective_loss', 'loss_tokens',
              'trained_rollouts', 'trained_tokens', 'sampled_tokens', 'rollout_count', 'group_count', 'admitted_groups',
              'derailed_rollouts', 'masked_rollouts', 'unresolved_filter', 'unresolved_trained_rollouts', 'outcome_counts',
              'epoch_metrics', 'trajectories', 'data_sha256', 'input_documents_sha256', 'runtime_context',
              'maximum_old_actor_logprob_drift', 'support_renorm')
FORBID = ('seed_hex', 'r2.cloudflarestorage', 'X-Amz', 'aws_secret', 'secret_access', 'access_key')


def rj(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None


def log(**kw):
    print(json.dumps(dict(kw, at=round(time.time(), 1))), flush=True)


def wj_public(path, obj):
    data = (json.dumps(obj, indent=1, sort_keys=True) + '\n').encode()
    low = data.decode('utf-8', 'replace')
    for bad in FORBID:
        if bad in low: raise RuntimeError('publish guard: %s in %s' % (bad, path))
    write_bytes(path, data)


def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_bytes(data); os.chmod(tmp, 0o644); os.replace(tmp, path)


def final_decision(e):
    d = rj(GATE / str(e) / 'DECISION.json')
    return d if d and d.get('decision') in ('green', 'red') and d.get('epoch') == e else None


_bucket = None
def bucket():
    global _bucket
    if _bucket is None:
        if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
        from subnet.storage import Bucket
        _bucket = Bucket(json.loads(LIFE_CFG.read_text())['bucket'])
    return _bucket


def engine_spec():
    cache = rj(HOST_CACHE) or {}
    raw = HOST_CFG.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if cache.get('operator_config_sha256') and cache['operator_config_sha256'] != sha:
        return dict(status='unavailable: local trainer host config does not match the operator config hash in use')
    eng = json.loads(raw).get('engine') or {}
    return dict(status='ok', trainer_host=cache.get('trainer_host'), operator_config_sha256=sha,
                engine={k: eng[k] for k in ENGINE_KEYS if k in eng},
                note='Trainer host engine settings in force at publish time (the operator config hash is also in mining.json '
                     'training). Per-turn facts (microbatch plan, optimizer stats, loss aggregation) are in this turn\'s training block.')


def publish_turn(e, dec):
    out = TURNS / str(e) / 'turn.json'
    if out.exists(): return False
    ed = LIFE / 'epochs' / str(e)
    turn = rj(ed / 'K1-TURN.json')
    if not turn or turn.get('status') != 'turn': return False
    if turn.get('format') == 'affine-batch-v1': return publish_ff_turn(e, dec, turn, ed, out)
    env = rj(ed / 'TRAIN-DISPATCH-K1.json')
    res = rj(ed / 'TRAINING-RESULT.json')
    if not env or not res: return False
    pay = env.get('payload') or env
    rep = (res.get('payload') or res).get('report') or {}
    tr = rep.get('training') or {}
    desc = ((rep.get('publication_receipt') or {}).get('payload') or {}).get('descriptor') or {}
    if desc.get('id') and dec.get('candidate') and desc['id'] != dec['candidate']:
        log(event='candidate_mismatch', epoch=e); return False
    docs, seen = [], {}
    for row in pay.get('selected') or []:
        d = row.get('document') or {}
        key, sha = d.get('key'), d.get('sha256')
        if not key or not sha or key in seen: continue
        data = bucket().get(key)
        if hashlib.sha256(data).hexdigest() != sha: raise RuntimeError('document sha mismatch e%d slot %s' % (e, d.get('slot')))
        name = 'docs/%s-%s.json' % (d.get('slot', len(docs)), sha[:12])
        write_bytes(TURNS / str(e) / name, data)
        seen[key] = name
        docs.append(dict(slot=d.get('slot'), sha256=sha, bytes=len(data), received_at=d.get('received_at'),
                         r2_key=key, url='/k1/turns/%d/%s' % (e, name)))
    sel = []
    for row in pay.get('selected') or []:
        r = dict(row); d = dict(r.get('document') or {})
        if d.get('key') in seen: d['url'] = '/k1/turns/%d/%s' % (e, seen[d['key']])
        r['document'] = d; sel.append(r)
    ex = pay.get('excluded') or []
    doc = dict(
        schema=SCHEMA, epoch=e, published_at=round(time.time(), 1),
        decision={k: dec.get(k) for k in ('decision', 'reasons', 'head', 'candidate', 'evaluated', 'batch_overlap', 'batch_tasks',
                                           'sample_size', 'z_min', 'z_inclusive', 'cap_max', 'min_tasks', 'commitment',
                                           'decided_at', 'decided_at_cr', 'mode')},
        promoted=dec.get('decision') == 'green',
        turn={k: turn.get(k) for k in ('uid', 'hotkey', 'reg_block', 'queue_rule', 'queue_position', 'tasks', 'rollouts', 'head', 'at',
                                       'skipped_not_ready_count')},
        task_indices=sorted({int(r['index']) for r in sel if 'index' in r}),
        dispatch=dict(
            source='lifecycle TRAIN-DISPATCH-K1 (filtered, signed): the exact rows the trainer consumed',
            envelope_file_sha256=hashlib.sha256((ed / 'TRAIN-DISPATCH-K1.json').read_bytes()).hexdigest(),
            signer=env.get('signer'), signature=env.get('signature'),
            header={k: pay.get(k) for k in ('version', 'epoch', 'run_id', 'checkpoint', 'manifest_sha256', 'population_sha256',
                                            'claim_order_rule', 'selected_task_count', 'selected_rollout_count', 'selection',
                                            'training_requires_sampling_audit')},
            selected=sel,
            excluded_count=len(ex), contested_claims_count=len(pay.get('contested_claims') or []),
            note='Other miners\' excluded and contested rows are summarised as counts, so their task choices stay private. '
                 'The signature covers the full payload, so this partial view cannot be checked against it.'),
        documents=docs,
        candidate=dict(id=desc.get('id'), files_sha256=desc.get('files'),
                       r2_prefix='mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/model/' % desc.get('id'),
                       promoted=dec.get('decision') == 'green',
                       note='GREEN candidates become the next head; RED candidates are discarded but their hashes are listed here.'),
        training={k: tr.get(k) for k in TRAIN_KEYS if k in tr},
        runtime_versions=rep.get('runtime_versions'), backend_profile=rep.get('backend_profile'),
        engine=engine_spec(),
        recipe='/k1/spec/step-recipe.json', replay='/k1/replay/README.md',
        gate_reveal='/gate/reveal/%d.json' % e)
    wj_public(out, doc)
    return True


def publish_ff_turn(e, dec, turn, ed, out):
    """free-file turn (Jacob pt898u): the miner's file is published only on GREEN; on RED it stays private (no file, no
    per-rollout trajectories, no per-group hashes)."""
    env, res = rj(ed / 'TRAIN-DISPATCH-FF.json'), rj(ed / 'TRAINING-RESULT.json')
    if not env or not res: return False
    pay = env.get('payload') or env
    rep = (res.get('payload') or res).get('report') or {}
    tr = rep.get('training') or {}
    desc = ((rep.get('publication_receipt') or {}).get('payload') or {}).get('descriptor') or {}
    if desc.get('id') and dec.get('candidate') and desc['id'] != dec['candidate']:
        log(event='candidate_mismatch', epoch=e); return False
    green = dec.get('decision') == 'green'
    batch = dict(published=False, note='RED: the miner file stays private.')
    if green:
        raw = (ed / 'ff' / (turn['hotkey'] + '.json')).read_bytes()
        if hashlib.sha256(raw).hexdigest() != turn['file_sha256']: raise RuntimeError('ff snapshot sha mismatch e%d' % e)
        write_bytes(TURNS / str(e) / 'batch.json', raw)
        batch = dict(published=True, url='/k1/turns/%d/batch.json' % e, sha256=turn['file_sha256'], bytes=len(raw),
                     note='The exact file copied from the miner object at the turn (before the batch-cap trim).')
    private = ('trajectories', 'admitted_groups', 'input_documents_sha256', 'data_sha256')
    doc = dict(
        schema=SCHEMA, epoch=e, published_at=round(time.time(), 1), format=turn['format'],
        decision={k: dec.get(k) for k in ('decision', 'reasons', 'head', 'candidate', 'evaluated', 'batch_tasks', 'sample_size',
                                           'z_min', 'z_inclusive', 'cap_max', 'min_tasks', 'commitment', 'decided_at', 'decided_at_cr', 'mode')},
        promoted=green,
        turn={k: turn.get(k) for k in ('uid', 'hotkey', 'reg_block', 'queue_rule', 'queue_position', 'tasks', 'rollouts', 'dropped_groups',
                                       'capped_rollouts', 'file_sha256', 'file_bytes', 'head', 'at', 'skipped_not_ready_count')},
        batch=batch,
        dispatch=dict(source='lifecycle TRAIN-DISPATCH-FF (signed): one document per prompt group, rewards as given',
                      envelope_file_sha256=hashlib.sha256((ed / 'TRAIN-DISPATCH-FF.json').read_bytes()).hexdigest(),
                      signer=env.get('signer'),
                      header={k: pay.get(k) for k in ('version', 'epoch', 'run_id', 'checkpoint', 'manifest_sha256', 'format',
                                                      'selected_task_count', 'selected_rollout_count')},
                      **({'selected': pay.get('selected'), 'signature': env.get('signature')} if green else {})),
        candidate=dict(id=desc.get('id'), files_sha256=desc.get('files'),
                       r2_prefix='mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/model/' % desc.get('id'), promoted=green,
                       note='GREEN candidates become the next head; RED candidates are discarded but their hashes are listed here.'),
        training={k: tr.get(k) for k in TRAIN_KEYS if k in tr and (green or k not in private)},
        runtime_versions=rep.get('runtime_versions'), backend_profile=rep.get('backend_profile'), engine=engine_spec(),
        recipe='/k1/spec/step-recipe.json', gate_reveal='/gate/reveal/%d.json' % e)
    wj_public(out, doc)
    return True


def publish_reveal(e, dec):
    out = REVEAL / ('%d.json' % e)
    if out.exists(): return False
    g = GATE / str(e)
    s, c, h, cp = rj(g / 'SAMPLE.private.json'), rj(g / 'COMMIT.json'), rj(g / 'H-PROBE.private.json'), rj(g / 'C-PROBE.private.json')
    if not (s and c and h and cp): return False
    if s.get('standing') and not (GATE.parent / 'standing' / str(s.get('standing_id')) / 'ROTATED.json').exists():
        return False   # v2a: standing private sample is revealed only after it has been rotated out
    if s.get('heldout') and not (GATE.parent / 'heldout' / str(s.get('round_id')) / 'CLOSED.json').exists():
        return False   # held-out round (Jacob pt808u): secret set + per-task grades are revealed only after the round closes
    if str(GCODE) not in sys.path: sys.path.insert(0, str(GCODE))
    import gate_core as G
    flags = rj(FLAG) or {}
    pool = int(flags.get('pool', G.DEFAULTS['pool']))
    seed, idx = s['seed'], list(s['indices'])
    redraw = G.draw_sample(seed, pool, len(idx))
    commit = G.sample_commitment(seed, idx)
    P = hashlib.sha256(('%s:k1-crn-plan' % seed).encode()).hexdigest()
    if s.get('heldout'): P = dict(head_blocks=h.get('plan_seeds'), candidate=cp.get('plan_seed'))
    d = rj(LIFE / 'epochs' / str(e) / 'TRAIN-DISPATCH-K1.json') or {}
    batch = sorted({int(r['index']) for r in ((d.get('payload') or d).get('selected') or []) if 'index' in r})
    fe = flags.get('turn_batch_counted_from_epoch')
    if type(fe) is int and e >= fe: batch = []      # free-file (pt898u): the turn batch is no longer excluded from the probe
    hpt = {int(k): v for k, v in (h.get('per_task') or {}).items()}
    cpt = {int(k): v for k, v in (cp.get('per_task') or {}).items()}
    if s.get('heldout'):   # v2 held-out scoring (weighted gain on the round's kept tasks), recomputed from the round baseline
        if str(GCODE) not in sys.path: sys.path.insert(0, str(GCODE))
        import heldout as HO
        rb = rj(GATE.parent / 'heldout' / str(s.get('round_id')) / 'BASELINE.private.json') or {}
        re = HO.score_ivw(rb.get('selection') or {}, hpt, cpt, rb.get('kept') or [], batch, env='affine_math', bar_z=float(dec.get('z_min', 1.0)),
                          z_inclusive=bool(dec.get('z_inclusive')), cap_max=float(dec.get('cap_max', 0.2)), min_kept=int(dec.get('min_tasks', 100)),
                          set_size=len(idx), sigma_run=float(((rj(GATE / str(e) / 'EVALUATED.private.json') or {}).get('evaluated') or {}).get('sigma_run') or 0.0),
                          n_p_blocks=len(rb.get('plan_seeds') or [0, 0, 0, 0]) - 1)
    else:
        re = G.decide(hpt, cpt, idx, batch, z_min=float(dec.get('z_min', 0.0)), cap_max=float(dec.get('cap_max', 0.2)),
                      min_tasks=int(dec.get('min_tasks', 200)), z_inclusive=bool(dec.get('z_inclusive')))
    doc = dict(
        schema=SCHEMA + '-gate-reveal', epoch=e, published_at=round(time.time(), 1),
        revealed_after='DECISION.json (%s, %s)' % (dec.get('decision'), dec.get('decided_at_cr')),
        version=s.get('version'), sample_seed=seed, sample_drawn_at_cr=s.get('drawn_at_cr'), pool=pool,
        sample_indices=idx, commitment_published=c.get('commitment'), commitment_recomputed=commit,
        checks=dict(commitment_matches=commit == c.get('commitment') == dec.get('commitment'),
                    indices_match_seed=redraw == sorted(idx),
                    decision_recomputed=re['decision'], decision_matches=re['decision'] == dec.get('decision'),
                    z_recomputed=re['evaluated']['z'], z_published=(dec.get('evaluated') or {}).get('z')),
        plan_seed=P, plan_seed_rule=("held-out round: head baseline block b uses sha256(sample_seed + ':heldout-baseline:' + b), block 0 "
                                     "= selection on all of S, blocks 1.. = p_i on the kept tasks; every candidate uses block 1's plan seed (common random "
                                     "numbers); request seeds derive from the plan seed, env, task "
                                     "index and sample j" if s.get('heldout') else
                                     "sha256(sample_seed + ':k1-crn-plan'); head and candidate probes use the same P, "
                                     "and each request's sampling seed is derived from P, env, task index and sample j"),
        heldout_round=s.get('round_id'), round_reveal=('/gate/heldout/%s.json' % s['round_id']) if s.get('heldout') else None,
        decision_params=dict(z_min=float(dec.get('z_min', 0.0)), cap_max=float(dec.get('cap_max', 0.2)),
                             min_tasks=int(dec.get('min_tasks', 200)), z_inclusive=bool(dec.get('z_inclusive'))),
        k=h.get('k'), seeded=h.get('seeded'), head=h.get('head'), candidate=cp.get('candidate'),
        batch_tasks=batch, evaluated_tasks=[t for t in idx if t not in set(batch)],
        cell_keys=dict(n='samples graded', c='correct', b='max-token cap hits', L='per-sample output lengths',
                       tok='total output tokens', u='unresolved', w='wrong', x='errors'),
        head_per_task={str(k): hpt[k] for k in sorted(hpt)},
        candidate_per_task={str(k): cpt[k] for k in sorted(cpt)},
        decision_rule='/gate/reveal/gate_core.py decide(); verify with /gate/reveal/verify_gate_reveal.py',
        standing_id=s.get('standing_id'), head_probe_reused=bool(h.get('reused')), head_probe_from_epoch=h.get('reused_from_epoch'),
        note=('Held-out round %s: one secret set per head, reused by every turn of the round against a head baseline computed once '
              '(selection block k4 on all of S + 3 x k4 p_i blocks on the kept tasks; candidates scored by inverse-variance-weighted gain); revealed when the round closed. Full round (baseline blocks, all candidates): /gate/heldout/%s.json'
              % (s.get('round_id'), s.get('round_id'))) if s.get('heldout') else ('Standing private sample %s: reused across turns (so the head probe is reused after RED and the promoted candidate\'s '
              'own probe becomes the new head probe); revealed only after rotation (every 24 h or when the head must be re-probed). '
              'The next standing seed is a fresh os.urandom(32).' % s.get('standing_id')) if s.get('standing') else
             'The seed is os.urandom(32) drawn fresh when each epoch opens, so revealing a finished epoch tells nothing about the next sample.')
    wj_public(out, doc)
    return True


def indexes():
    t = []
    for p in sorted(TURNS.glob('*/turn.json'), key=lambda p: int(p.parent.name)):
        d = rj(p) or {}
        t.append(dict(epoch=d.get('epoch'), uid=(d.get('turn') or {}).get('uid'), decision=(d.get('decision') or {}).get('decision'),
                      candidate=(d.get('candidate') or {}).get('id'), tasks=len(d.get('task_indices') or []),
                      turn='/k1/turns/%s/turn.json' % p.parent.name, gate_reveal='/gate/reveal/%s.json' % p.parent.name))
    wj_public(TURNS / 'index.json', dict(schema=SCHEMA, turns=t, recipe='/k1/spec/step-recipe.json', replay='/k1/replay/README.md',
                                         dev_split='/k1/dev/README.md', updated_at=round(time.time(), 1)))
    r = []
    for p in sorted(REVEAL.glob('[0-9]*.json'), key=lambda p: int(p.stem)):
        d = rj(p) or {}
        r.append(dict(epoch=d.get('epoch'), url='/gate/reveal/%s' % p.name, **(d.get('checks') or {})))
    wj_public(REVEAL / 'index.json', dict(schema=SCHEMA + '-gate-reveal', reveals=r, gate_core='/gate/reveal/gate_core.py',
                                          verify='/gate/reveal/verify_gate_reveal.py', updated_at=round(time.time(), 1)))


def tick(start):
    n = 0
    for d in sorted((p for p in GATE.iterdir() if p.name.isdigit() and int(p.name) >= start), key=lambda p: int(p.name)):
        e = int(d.name); dec = final_decision(e)
        if not dec: continue
        for f in (publish_turn, publish_reveal):
            try:
                if f(e, dec): n += 1; log(event=f.__name__, epoch=e)
            except Exception as ex:
                log(event='error', fn=f.__name__, epoch=e, error=str(ex)[:300])
    if n or not (TURNS / 'index.json').exists(): indexes()
    return n


if __name__ == '__main__':
    start = int(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else FROM
    if '--once' in sys.argv:
        print(tick(start)); sys.exit(0)
    while True:
        try: tick(start)
        except Exception: log(event='tick_error', error=traceback.format_exc()[-400:])
        time.sleep(60)
