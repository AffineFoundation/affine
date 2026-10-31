"""SN120 free-file turns in the lifecycle (Jacob 2026-10-30 pt898u; RA/20261030-free-file).

Installed by ff_launcher.py over the live v8.2.9x launcher (L), between L's K1 wrapper and the promotion gate wrapper:

  Lifecycle._training (after the seal):
    queue = the unchanged K1 order (hotkey-registration-once: registration block, uid; UID 0 and served hotkeys /
    registrations out). For each queue member in order:
      * their object private/ff/v1/miners/<hotkey>/batch.json exists -> server-side copy to
        private/ff/v1/turns/<epoch>/<hotkey>.json (+ local epochs/N/ff/<hotkey>.json) and format check (ff_format);
        valid -> the turn (free-file);
      * else, before the old-format cutoff, a ready old-format admitted batch -> the unchanged K1 path (L);
      * else skipped, not charged, keeps its place.
    A free-file turn trains from the published head exactly like a K1 turn: one signed job, same trainer, same config;
    the dispatch is a signed 'sn120-ff-dispatch-v1' selection of one document per prompt group (ff_trainer.py on the
    trainer host reads it). K1-TURN.json / K1-TURNS.jsonl are written exactly as for K1 turns (queue + credit unchanged).
  L.v829t_predict (early seal): also predicts a free-file turn from the ff_service readiness file.
  Lifecycle._capture: after the cutoff (FF.json legacy_capture false) old-format commitments are no longer captured.
Config: staged/FF.json (read every call; absent / enabled false -> exact L behaviour).
"""
import hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ff_format

FF_FLAG = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/FF.json')
BATCH_CAP = FF_FLAG.parent / 'BATCH-CAP.json'
GRANTS = FF_FLAG.parent / 'K1-GRANTS.json'      # Jacob 2026-10-30 19:30 CR: one fresh turn for near-miss REDs (0 < weighted_z <= 1)
FF_STATE = Path('/home/const/subnet120-rewrite/state/root-audits/20261030-free-file/state')
FF_READY = FF_STATE / 'FF-READY.json'
L_K1_CHAIN = Path('/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/state/K1-CHAIN.json')
MINER_PREFIX = 'private/ff/v1/miners/'
TURN_PREFIX = 'private/ff/v1/turns/'
DISPATCH = 'sn120-ff-dispatch-v1'
DOCUMENT = 'sn120-ff-group-v1'
TOKENIZER_FILES = ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json')
L = None          # the live launcher module (set by install)
_TOK = {}


def say(**kw):
    print(json.dumps(dict(component='ff', **kw), default=str), flush=True)


def rj(path):
    try:
        return json.loads(Path(path).read_bytes())
    except Exception:
        return None


def wj(path, value):
    path = Path(path); tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n'); os.replace(tmp, path)


def flag():
    d = rj(FF_FLAG)
    if not (isinstance(d, dict) and d.get('enabled') is True and type(d.get('from_epoch')) is int
            and type(d.get('legacy_cutoff_unix')) is int and isinstance(d.get('tokenizer_dir'), str)):
        return None
    return d


def legacy_open(f, at=None):
    return (time.time() if at is None else at) < f['legacy_cutoff_unix']


def object_key(hotkey):
    return MINER_PREFIX + hotkey + '/batch.json'


def caps(adapter=None):
    c = rj(BATCH_CAP) or {}
    t, r = int(c.get('max_tasks_per_miner', 125)), int(c.get('max_rollouts_per_miner', 1000))
    if adapter is not None:
        t = min(t, int(adapter.trainer_config['settings']['max_tasks']))
    return t, r


def tokenizer(f, head_files):
    """Pinned tokenizer dir; its files must equal the current head's (CURRENT checkpoint file hashes)."""
    pins = {n: head_files[n] for n in TOKENIZER_FILES}
    key = (f['tokenizer_dir'], tuple(sorted(pins.items())))
    if key not in _TOK:
        _TOK.clear(); _TOK[key] = ff_format.tokenizer(f['tokenizer_dir'], pins)
    return _TOK[key]


def open_grants(turns):
    """K1-GRANTS.json (enabled): granted hotkeys whose fresh turn is still unused, oldest RED epoch first. A grant is used only
    by a real turn (a K1-TURNS row for that hotkey after its RED epoch); a skip (missing / invalid file) does not use it."""
    g = rj(GRANTS)
    if not (isinstance(g, dict) and g.get('enabled') is True):
        return []
    out = []
    for r in g.get('grants') or []:
        try:
            hk, e0 = r['hotkey'], int(r['red_epoch'])
        except Exception:
            continue
        if any(t.get('hotkey') == hk and int(t.get('epoch_index', -1)) > e0 for t in turns):
            continue
        out.append(r)
    return sorted(out, key=lambda r: int(r['red_epoch']))


def grant_rows(chain, turns):
    """[(position, uid, row + grant)] for open grants whose hotkey is registered now (UID != 0); positions -k..-1, i.e. ahead
    of position 0 (the K1 order itself is unchanged)."""
    by_hk = {row['hotkey']: (int(u), row) for u, row in (chain.get('uids') or {}).items()
             if int(u) != 0 and isinstance(row, dict) and type(row.get('miner_hex')) is str}
    rows = [(by_hk[r['hotkey']][0], dict(by_hk[r['hotkey']][1], grant=dict(red_epoch=int(r['red_epoch']), weighted_z=r.get('weighted_z'))))
            for r in open_grants(turns) if r['hotkey'] in by_hk]
    return [(i - len(rows), u, row) for i, (u, row) in enumerate(rows)]


def queue(chain, turns):
    """Open K1-GRANTS first (free-file only), then the unchanged K1 order for hotkey-registration-once: [(position, uid, row)]."""
    hks, regs = L.k1_served_hotkeys(turns), L.k1_served_registrations(turns)
    out = []
    for u, row in (chain.get('uids') or {}).items():
        u = int(u)
        if u == 0 or not isinstance(row, dict) or type(row.get('miner_hex')) is not str:
            continue
        reg = int(row.get('reg_block') or 0)
        if row['hotkey'] in hks or (reg > 0 and (u, reg) in regs):
            continue
        out.append((reg, u, row))
    out.sort(key=lambda x: (x[0], x[1]))
    return grant_rows(chain, turns) + [(i, u, row) for i, (reg, u, row) in enumerate(out)]


def legacy_ready(sealed, f):
    """miner_hex -> distinct admitted old-format tasks (k1_select's own test, >= min_tasks)."""
    out = {}
    for m, rows in L.k1_rows_by_miner(sealed).items():
        n = len({(r.get('env_id'), r.get('index')) for r in rows})
        if n >= f.get('min_tasks', 8):
            out[m] = n
    return out


def listing(bucket):
    """hotkey -> (etag, size) for every miner object."""
    out, token = {}, None
    while True:
        kw = dict(Bucket=bucket.name, Prefix=MINER_PREFIX, MaxKeys=1000)
        if token:
            kw['ContinuationToken'] = token
        r = bucket.client.list_objects_v2(**kw)
        for o in r.get('Contents') or []:
            parts = o['Key'][len(MINER_PREFIX):].split('/')
            if len(parts) == 2 and parts[1] == 'batch.json':
                out[parts[0]] = (o['ETag'].strip('"'), int(o['Size']))
        if not r.get('IsTruncated'):
            return out
        token = r['NextContinuationToken']


def snapshot(bucket, hotkey, epoch_name, local_dir):
    """Server-side copy of the miner's object at the turn -> (raw bytes, copy key). None when it vanished."""
    from botocore.exceptions import ClientError
    key = TURN_PREFIX + epoch_name + '/' + hotkey + '.json'
    try:
        bucket.client.copy_object(Bucket=bucket.name, Key=key, CopySource=dict(Bucket=bucket.name, Key=object_key(hotkey)))
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in ('NoSuchKey', '404'):
            return None
        raise
    head = bucket.client.head_object(Bucket=bucket.name, Key=key)
    if int(head['ContentLength']) > ff_format.MAX_FILE_BYTES:
        raw = b''         # too large: recorded, never read
    else:
        raw = bucket.client.get_object(Bucket=bucket.name, Key=key)['Body'].read()
    local_dir.mkdir(mode=0o700, exist_ok=True)
    (local_dir / (hotkey + '.json')).write_bytes(raw)
    return raw, key, int(head['ContentLength'])


def select(self, d, e, m, sealed, current, adapter, f, k1f):
    """-> turn dict: status turn (route ff | legacy) or skip. Pure except for the snapshot copies."""
    chain = json.loads(L.K1_CHAIN.read_bytes())
    L.need(time.time() - float(chain.get('taken_at', 0)) < k1f['chain_max_age_seconds'], 'K1 chain snapshot fresh')
    turns = L.k1_turns(before_epoch=e)
    q = queue(chain, turns)
    objects = listing(adapter.bucket)
    ready = legacy_ready(sealed, k1f) if legacy_open(f) else {}
    ready_cache = (rj(FF_READY) or {}).get('miners') or {}
    max_tasks, max_rollouts = caps(adapter)
    tok = None
    skipped = []
    base = dict(epoch_index=e, queue_rule='hotkey-registration-once', queue_length=len(q), chain_block=chain.get('block'),
                legacy_open=legacy_open(f), caps=dict(max_tasks=max_tasks, max_rollouts=max_rollouts))
    for pos, u, row in q:
        hk = row['hotkey']; who = dict(uid=u, hotkey=hk, reg_block=int(row.get('reg_block') or 0), miner=row['miner_hex'],
                                        queue_position=pos, turns_before=1 if row.get('grant') else 0,
                                        **({'grant': row['grant']} if row.get('grant') else {}))
        reason = None
        if hk in objects:
            etag, size = objects[hk]
            known = ready_cache.get(hk) or {}
            if known.get('etag') == etag and known.get('valid') is False:
                reason = 'invalid-file: ' + str(known.get('reason'))[:200]
            else:
                got = snapshot(adapter.bucket, hk, m['epoch'], d / 'ff')
                if got is None:
                    reason = 'no-file'
                else:
                    raw, key, n = got
                    if tok is None:
                        tok = tokenizer(f, current['checkpoint']['files'])
                    ok, info, groups = ff_format.check(raw, hk, tok, max_tasks, max_rollouts) if raw else (False, 'file larger than %d bytes' % ff_format.MAX_FILE_BYTES, None)
                    if ok:
                        return dict(base, status='turn', route='ff', format=ff_format.FORMAT, **who, snapshot_key=key,
                                    file_sha256=info['file_sha256'], file_bytes=n, tasks=info['groups'], rollouts=info['rollouts'],
                                    dropped_groups=info['dropped_groups'], capped_rollouts=info['capped_rollouts'],
                                    skipped_not_ready=skipped[:50], skipped_not_ready_count=len(skipped), groups=groups)
                    reason = 'invalid-file: ' + info[:200]
        if row['miner_hex'] in ready and not row.get('grant'):
            return dict(base, status='turn', route='legacy', **who, tasks=ready[row['miner_hex']],
                        skipped_not_ready=skipped[:50], skipped_not_ready_count=len(skipped))
        skipped.append(dict(uid=u, hotkey=hk, reason=reason or 'no-file'))
    return dict(base, status='skip', reason='no-unserved-hotkey-with-valid-batch', skipped_not_ready=skipped[:50],
                skipped_not_ready_count=len(skipped))


def documents(turn, m, current):
    """One canonical document per prompt group -> [(sha256, bytes, row)]."""
    out = []
    for i, g in enumerate(turn['groups']):
        ps = hashlib.sha256(g['prompt'].encode()).hexdigest()
        doc = dict(version=DOCUMENT, epoch=m['epoch'], checkpoint=current['checkpoint']['id'], hotkey=turn['hotkey'], group_index=i,
                   prompt_sha256=ps, prompt_tokens=g['prompt_tokens'], rollouts=g['rollouts'])
        raw = json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        h = hashlib.sha256(raw).hexdigest()
        out.append((h, raw, dict(index=i, prompt_sha256=ps, rollouts=len(g['rollouts']),
                                 document=dict(sha256=h, key=TURN_PREFIX + m['epoch'] + '/docs/' + h + '.json', size=len(raw)))))
    return out


def record_turn(d, e, turn, current, m):
    """K1-TURN.json + K1-TURNS.jsonl exactly as the K1 path writes them (queue and credit stay unchanged)."""
    public = {k: v for k, v in turn.items() if k != 'groups'}
    public.update(version='sn120-k1-turn-v1', epoch=m.get('epoch'), head=current['checkpoint']['id'],
                  optimizer_step=current['optimizer_step'], at=time.time())
    wj(d / 'K1-TURN.json', public)
    if public['status'] == 'turn' and not any(t.get('epoch_index') == e for t in L.k1_turns()):
        L.K1_STATE.mkdir(parents=True, exist_ok=True)
        rec = {k: public.get(k) for k in ('epoch_index', 'epoch', 'uid', 'hotkey', 'miner', 'tasks', 'rollouts', 'queue_position',
                                          'turns_before', 'head', 'at', 'reg_block', 'queue_rule', 'grant')}
        rec.update(format=ff_format.FORMAT, file_sha256=public.get('file_sha256'))
        with open(L.K1_TURNS, 'a') as fh:
            fh.write(json.dumps(rec, sort_keys=True) + '\n')
    return public


def build_job(self, d, e, m, opening, sealed, current, adapter, turn):
    from subnet.storage import canonical
    from subnet.mimo_release_policy import sign, digest
    from subnet.mimo_lifecycle import save_manifest
    from subnet.complete_group_capture import _immutable
    from subnet.mimo_worker import VERSION as WORKER
    docs = documents(turn, m, current)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda x: _immutable(adapter.bucket, x[2]['document']['key'], x[1]), docs))
    rows = [x[2] for x in docs]
    payload = dict(version=DISPATCH, epoch=sealed['epoch'], run_id=sealed['run_id'], checkpoint=sealed['checkpoint'],
                   manifest_sha256=sealed['manifest_sha256'], training_requires_sampling_audit=False, format=ff_format.FORMAT,
                   hotkey=turn['hotkey'], uid=turn['uid'], file_sha256=turn['file_sha256'], snapshot_key=turn['snapshot_key'],
                   selected=rows, selected_task_count=len(rows), selected_rollout_count=sum(r['rollouts'] for r in rows))
    filtered = sign(self.identity, payload)
    key = TURN_PREFIX + m['epoch'] + '/dispatch.json'
    _immutable(adapter.bucket, key, canonical(filtered))
    t0 = time.time()
    value = dict(schema=1, worker_policy=WORKER, job_id='mimo-train-' + digest(filtered)[:48], role='train',
                 created_at=t0, expires_at=t0 + adapter.job_seconds, manifest=opening,
                 source_files=adapter.source_files, runtime_versions=adapter.runtime_versions,
                 trainer_worker=adapter.trainer_worker, checkpoint_read_urls=adapter.checkpoint_urls(m['checkpoint']),
                 training=dict(run_id=m['run_id'], optimizer_step=current['optimizer_step'], config=adapter.trainer_config,
                               dispatch=dict(url=adapter.bucket.presign(key, 'get_object', 86400), sha256=digest(filtered),
                                             size=len(canonical(filtered))),
                               documents={r['document']['sha256']: dict(url=adapter.bucket.presign(r['document']['key'], 'get_object', 86400),
                                                                        size=r['document']['size']) for r in rows},
                               publication_namespace=self.policy.namespace(), publisher_id=adapter.publisher_id,
                               role_config_file=adapter.role_config))
    save_manifest(d / 'TRAIN-DISPATCH-FF.json', filtered)
    save_manifest(d / 'TRAINING-JOB.json', sign(self.identity, value))
    return filtered


def install_training(Lifecycle):
    prior = Lifecycle._training          # L's K1 path (falls through to the original training)
    inner = L.INNER['training']          # the original Lifecycle._training

    def _training(self, directory, opening, dispatch, current, now):
        f, k1f, d = flag(), L.k1_flag(), Path(directory)
        try:
            e = int(d.name)
        except ValueError:
            return prior(self, directory, opening, dispatch, current, now)
        if f is None or k1f is None or e < f['from_epoch'] or k1f.get('queue_rule', L.K1_DEFAULT_RULE) != 'hotkey-registration-once':
            return prior(self, directory, opening, dispatch, current, now)
        fp = d / 'TRAIN-DISPATCH-FF.json'
        if (d / 'TRAINING-JOB.json').exists():
            return inner(self, directory, opening, json.loads(fp.read_bytes()), current, now) if fp.exists() else \
                prior(self, directory, opening, dispatch, current, now)
        ctl = L.train_control()
        if ctl is not None and ctl['mode'] != 'train':
            return prior(self, directory, opening, dispatch, current, now)
        from subnet.backend_jobs import signed
        adapter = getattr(self.services.make_training_job, '__self__', None)
        L.need(adapter is not None and type(getattr(adapter, 'job_seconds', None)) is int, 'ff training adapter')
        adapter.job_seconds = L.TRAIN_JOB_SECONDS
        m, sealed = signed(opening, self.identity.id), signed(dispatch, self.identity.id)
        sp = d / 'FF-SELECT.json'
        prev = rj(sp)
        if prev is not None and prev.get('route') == 'legacy':
            return prior(self, directory, opening, dispatch, current, now)
        if prev is not None and prev.get('status') == 'turn':        # resume after a crash: re-check the stored snapshot
            raw = (d / 'ff' / (prev['hotkey'] + '.json')).read_bytes()
            L.need(hashlib.sha256(raw).hexdigest() == prev['file_sha256'], 'ff snapshot unchanged')
            tok = tokenizer(f, current['checkpoint']['files'])
            v = ff_format.validate(raw, prev['hotkey'], tok, prev['caps']['max_tasks'], prev['caps']['max_rollouts'])
            turn = dict(prev, groups=v['groups'])
        elif prev is not None:
            turn = prev
        else:
            try:
                turn = select(self, d, e, m, sealed, current, adapter, f, k1f)
            except Exception as exc:     # never stall the epoch: old-format path while it is open, else a skip
                turn = dict(status='skip', epoch_index=e, reason='ff-selection-error', error=repr(exc)[:300],
                            **({'route': 'legacy'} if legacy_open(f) else {}))
            wj(sp, {k: v for k, v in turn.items() if k != 'groups'})
        say(epoch=e, **{k: turn.get(k) for k in ('status', 'route', 'uid', 'hotkey', 'tasks', 'rollouts', 'queue_position', 'reason',
                                                 'skipped_not_ready_count', 'file_sha256')})
        if turn.get('route') == 'legacy':
            return prior(self, directory, opening, dispatch, current, now)
        record_turn(d, e, turn, current, m)
        if turn['status'] != 'turn':
            L.join_audit_population(self, directory)
            return dict(status='no-update', checkpoint=current['checkpoint'], optimizer_step=current['optimizer_step'],
                        checkpoint_authorization=current.get('checkpoint_authorization'), reason='k1-skip-' + turn.get('reason', 'none'))
        filtered = build_job(self, d, e, m, opening, sealed, current, adapter, turn)
        say(epoch=e, status='ff-dispatch', uid=turn['uid'], tasks=turn['tasks'], rollouts=turn['rollouts'])
        return inner(self, directory, opening, filtered, current, now)

    _training.__qualname__, _training.__module__ = prior.__qualname__, prior.__module__
    Lifecycle._training = _training
    return dict(ff_training=True)


def install_predict():
    """Early seal (v829t) also fires for a predicted free-file turn: the first queue member that has a file the ff service
    found valid (same etag as the live listing is not re-checked here; the turn itself re-validates the copy)."""
    legacy_predict = L.v829t_predict

    def v829t_predict(st, manifest, pre):
        f = flag()
        if f is None or st['epoch'] < f['from_epoch']:
            return legacy_predict(st, manifest, pre)
        lt = legacy_predict(st, manifest, pre) if legacy_open(f) else None
        r = rj(FF_READY) or {}
        if time.time() - float(r.get('checked_at', 0)) > 180:
            return lt
        valid = {hk for hk, v in (r.get('miners') or {}).items() if v.get('valid') is True}
        chain = json.loads(L.K1_CHAIN.read_bytes())
        if not time.time() - float(chain.get('taken_at', 0)) < st['flag']['chain_max_age_seconds']:
            return lt
        for pos, u, row in queue(chain, L.k1_turns(before_epoch=st['epoch'])):
            if lt is not None and pos >= lt.get('queue_position', 1 << 30):
                return lt
            if row['hotkey'] in valid:
                v = r['miners'][row['hotkey']]
                return dict(status='turn', uid=u, hotkey=row['hotkey'], reg_block=int(row.get('reg_block') or 0),
                            tasks=v.get('groups'), queue_position=pos, queue_rule='hotkey-registration-once')
        return lt

    L.v829t_predict = v829t_predict
    return dict(ff_predict=True)


def install_capture(Lifecycle):
    """FF.json legacy_capture false (set after the cutoff) -> no old-format commitment is captured any more."""
    prior = Lifecycle._capture

    def _capture(self, coordinator, manifest, now):
        f = flag()
        if f is not None and f.get('legacy_capture') is False and not legacy_open(f):
            return {}
        return prior(self, coordinator, manifest, now)

    _capture.__qualname__, _capture.__module__ = prior.__qualname__, prior.__module__
    Lifecycle._capture = _capture
    return dict(ff_capture=True)


def install(launcher, Lifecycle):
    global L
    L = launcher
    return dict(install_training(Lifecycle), **install_predict(), **install_capture(Lifecycle), flag=str(FF_FLAG), on_now=flag() is not None)
