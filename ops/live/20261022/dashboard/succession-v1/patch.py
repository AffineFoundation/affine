#!/usr/bin/env python3
"""Apply the release-succession patch to the copied dashboard source (exact, once-only edits)."""
import sys
from pathlib import Path
N=Path(sys.argv[1]); S=N/'source'
def edit(rel, pairs):
    p=S/rel; t=p.read_text()
    for old,new in pairs:
        if t.count(old)!=1: raise SystemExit(f'{rel}: expected exactly one match for {old[:60]!r}, got {t.count(old)}')
        t=t.replace(old,new)
    p.write_text(t)

# 1. Projection: validate each epoch/finalized/eval against its own signed release member.
edit('dashboard/mimo_live_projection.py',[
("""        if type(first_round) is not int or first_round < 0 or finite(started_at) is None:
            raise ValueError("dashboard explicit run boundary")
""","""        if type(first_round) is not int or first_round < 0 or finite(started_at) is None:
            raise ValueError("dashboard explicit run boundary")

    def policy_for(self, manifest):
        # A signed release chain selects the member the manifest embeds and validates
        # against that member; a single ReleasePolicy keeps the original behaviour.
        if hasattr(self.policy, 'policy_for_manifest'):
            return self.policy.policy_for_manifest(manifest)
        self.policy.validate_manifest(manifest)
        return self.policy

    def check_round(self, policy, round_):
        # The member must govern this epoch's global round (append-only chain ranges).
        members = getattr(self.policy, 'members', None)
        if members is None:
            return
        for position, (start, member) in enumerate(members):
            if member.policy_sha256 == policy.policy_sha256:
                end = members[position+1][0] if position+1 < len(members) else None
                if round_ < start or (end is not None and round_ >= end):
                    raise ValueError('dashboard epoch round outside its signed release member')
                return
        raise ValueError('dashboard unknown release member')

    def baseline(self):
        # Run boundary stays the run's first release (base checkpoint and source).
        members = getattr(self.policy, 'members', None)
        return members[0][1] if members else self.policy
"""),
("""        value = signed(envelope, self.authority)
        self.policy.validate_manifest(value)
""","""        value = signed(envelope, self.authority)
        self.policy_for(value)
"""),
("""            display_epoch = len(epochs)+1
""","""            display_epoch = len(epochs)+1
            self.check_round(self.policy_for(m), first_round+display_epoch-1)
"""),
("""final['policy_sha256']!=self.policy.policy_sha256""","""final['policy_sha256']!=self.policy_for(m).policy_sha256"""),
("""            receipt=signed(read(path),self.authority)
            if receipt['policy_sha256']!=self.policy.policy_sha256:
                raise ValueError('dashboard evaluation release')
            envelope=receipt['job'];opening=signed(envelope,self.authority)['manifest'];m=self.opening(opening)
""","""            receipt=signed(read(path),self.authority)
            envelope=receipt['job'];opening=signed(envelope,self.authority)['manifest'];m=self.opening(opening)
            if receipt['policy_sha256']!=self.policy_for(m).policy_sha256:
                raise ValueError('dashboard evaluation release')
"""),
("""base_checkpoint=self.policy.scope['initial_checkpoint']['id']""","""base_checkpoint=self.baseline().scope['initial_checkpoint']['id']"""),
("""source_sha256=self.policy.scope['immutable_manifest']['source_bundle']['sha256'])""","""source_sha256=self.baseline().scope['immutable_manifest']['source_bundle']['sha256'])"""),
])

# 2. Server poll loop: log failures (journal) instead of silently passing; warn on a stale export.
edit('dashboard/server.py',[
("""    def poll():
        while True:
            time.sleep(15)
            try:
                db.refresh()
                export()
            except Exception:
                # Preserve previous committed snapshot on incomplete source writes.
                pass
""","""    import sys, traceback
    stale_after = 1800  # seconds without a successful export before warning
    state = {'ok': time.time(), 'last_error': None, 'logged': 0.0, 'stale_logged': 0.0}
    def log(message):
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + message, file=sys.stderr, flush=True)
    def poll():
        while True:
            time.sleep(15)
            try:
                db.refresh()
                export()
                if state['last_error'] is not None:
                    log('dashboard refresh recovered; export resumed')
                state.update(ok=time.time(), last_error=None)
            except Exception as error:
                # Preserve previous committed snapshot, but never fail silently.
                summary = type(error).__name__ + ': ' + str(error)[:300]
                now = time.time()
                if summary != state['last_error'] or now - state['logged'] >= 600:
                    log('ERROR dashboard refresh/export failed (previous export kept): ' + summary)
                    if summary != state['last_error']:
                        log(traceback.format_exc(limit=8))
                    state.update(last_error=summary, logged=now)
            age = time.time() - state['ok']
            if args.export:
                try: age = max(age, time.time() - Path(args.export).stat().st_mtime)
                except OSError: pass
            if age > stale_after and time.time() - state['stale_logged'] >= 600:
                log('WARNING dashboard export stale: last successful export %d s ago (threshold %d s)' % (age, stale_after))
                state['stale_logged'] = time.time()
"""),
])

# 3. Publisher: build the signed release chain from the live lifecycle config.
edit('ops/publish_mimo_dashboard.py',[
("""    policy=ReleasePolicy(lifecycle['release_policy'],cfg['authority'],allow_public=True)
    if policy.mode!='public':raise ValueError('public dashboard requires public release')
    projection=Projection(lifecycle['state_root'],policy,cfg['authority'],lifecycle['workers'],validate_report,
        first_round=lifecycle['lifecycle']['first_round'],started_at=cfg['started_at'],model=cfg['model'],
        audit_projection=AuditProjection(lifecycle['state_root'],lifecycle['queue_database'],cfg['authority'],
            lifecycle['workers'],policy.validate_manifest,
            {policy.scope['immutable_manifest']['source_bundle']['sha256']:
                {k:lifecycle['adapters'][k] for k in ('source_files','runtime_versions')}}))
""","""    active=ReleasePolicy(lifecycle['release_policy'],cfg['authority'],allow_public=True)
    from subnet.mimo_release_succession import ReleaseChain
    # Every epoch/finalized result/eval receipt/audit is validated against the signed
    # chain member governing its own round (v7 r189 -> v8.2 r228 -> v8.2.1 r229 -> v8.2.2/3 r231).
    policy=ReleaseChain(lifecycle['release_succession'],cfg['authority'],allow_public=True)
    if policy.active.policy_sha256!=active.policy_sha256:
        raise ValueError('configured release policy must be the active succession member')
    if policy.chain_sha256!=cfg['release_chain_sha256']:raise ValueError('dashboard exact release chain pin')
    if policy.mode!='public':raise ValueError('public dashboard requires public release')
    # Per-archive audit execution exactly as the lifecycle derives it: ROOT-approved
    # inventory per chain archive, with the active archive bound to the live closure.
    archives={p.policy_sha256:p.scope['immutable_manifest']['source_bundle']['sha256'] for _,p in policy.members}
    live={k:lifecycle['adapters'][k] for k in ('source_files','runtime_versions')}
    execution=copy.deepcopy(lifecycle['archive_execution'])
    if set(execution)!=set(archives.values()):raise ValueError('every signed release chain archive needs execution metadata')
    if execution[archives[policy.active.policy_sha256]]!=live:raise ValueError('active archive execution must equal the live closure')
    projection=Projection(lifecycle['state_root'],policy,cfg['authority'],lifecycle['workers'],validate_report,
        first_round=lifecycle['lifecycle']['first_round'],started_at=cfg['started_at'],model=cfg['model'],
        audit_projection=AuditProjection(lifecycle['state_root'],lifecycle['queue_database'],cfg['authority'],
            lifecycle['workers'],policy.validate_manifest,execution))
"""),
])
print('patched')
