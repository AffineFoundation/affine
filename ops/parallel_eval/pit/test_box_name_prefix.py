"""Shadow and fakepass boxes have a name that live never matches.

Nothing of the code under test is replaced. The live side is the real ``live/pe_common.py`` and
``live/pe_rent.py`` in a process of its own, as on the host. The shadow side is the real PIT rent service on
the real Lium helpers of that module, in a second process. That process sees the same live directories, has
the hard deny of shadow mode on, and has its shadow state root in a different place. The two processes share
one Lium account. Only the ``lium`` and ``ssh`` programs are stand-ins: ``lium`` keeps the pods of the account
in a file and logs every call.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ops.jobs.kinds import NAME_PREFIX
from ops.parallel_eval.pit import LIVE_NAME_ROOT, PREFIX_LIVE, PREFIX_SHADOW
from ops.parallel_eval.pit.durable import pit_state_root
from ops.parallel_eval.pit.isolated_rent import (
    ISOLATED_PREFIXES,
    LEGACY_ISOLATED_PREFIXES,
    PREFIX_FAKEPASS,
    IsolatedRent,
    IsolatedRentError,
    live_name_problem,
)
from ops.parallel_eval.pit.pe_rent import (
    NEVER_NAME_PREFIXES,
    RentError,
    RentService,
    bind_isolated_rent_service,
    is_fakepass_name,
    is_live_pit_name,
    is_pit_name,
    is_shadow_pit_name,
)

REPO = Path(__file__).resolve().parents[3]
LIVE_CODE = REPO / 'ops' / 'parallel_eval' / 'live'
ROLES = REPO / 'ops' / 'pods' / 'roles.json'

LANE = 'affine-pe-e1090-a0'                 # a live PE lane
LIVE_PIT = PREFIX_LIVE + '1090-a0'          # a live PIT box
SHADOW = PREFIX_SHADOW + '1090-a0'
FAKEPASS = PREFIX_FAKEPASS + '1090-a0'
ISOLATED = (SHADOW, FAKEPASS)

FAKE_LIUM = r'''#!/usr/bin/env python3
"""Stand-in for the Lium CLI: the pods and offers of one account are JSON files next to this program."""
import json, sys, uuid
from pathlib import Path
root = Path(__file__).resolve().parent
argv = sys.argv[1:]
with open(root / 'calls.jsonl', 'a') as log:
    log.write(json.dumps(argv) + '\n')
pods = json.loads((root / 'pods.json').read_text())
offers = json.loads((root / 'offers.json').read_text())
if argv[:1] == ['ps']:
    print(json.dumps(pods))
elif argv[:1] == ['ls']:
    print(json.dumps([o for o in offers if o['gpu_model'] == argv[argv.index('--gpu') + 1]]))
elif argv[:1] == ['up']:
    offer = next(o for o in offers if o['huid'] == argv[1])
    pod = dict(id=str(uuid.uuid4()), name=argv[argv.index('--name') + 1], huid=argv[1], ip='127.0.0.1',
               ssh_port=22000 + len(pods), status='RUNNING', price_per_hour=offer['price_per_hour'],
               gpu_model=offer['gpu_model'])
    (root / 'pods.json').write_text(json.dumps(pods + [pod]))
    print(json.dumps({'pod': pod}))
elif argv[:1] == ['rm']:
    left = [p for p in pods if p['id'] != argv[1]]
    if len(left) == len(pods):
        sys.exit('no pod %s' % argv[1])
    (root / 'pods.json').write_text(json.dumps(left))
else:
    sys.exit('unknown command %r' % argv)
'''

# Runs in the live process: the real live modules, imported by bare name from the live code dir.
LIVE_SIDE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import pe_common, pe_rent
isolated = json.loads(sys.argv[2])
ids = {p['name']: p['id'] for p in pe_common.lium_ps()}
out = dict(our_pods=sorted(p['name'] for p in pe_common.our_pods()),
           eval_nodes=sorted(p['name'] for p in pe_rent.eval_nodes()), rent={}, release={})
for name in isolated:
    try:
        pe_rent.rent(name + '-b', ttl='4h')
        out['rent'][name] = 'rented'
    except AssertionError:
        out['rent'][name] = 'refused'
    out['release'][name] = [pe_rent.release(name, 'test'), pe_rent.release(ids[name], 'test')]
count = len(out['eval_nodes'])
try:
    pe_rent.rent('affine-pe-cap-a', ttl='4h', max_eval_nodes=count)
    out['rent_at_cap'] = 'rented'
except pe_rent.CapReached:
    out['rent_at_cap'] = 'cap reached'
got = pe_rent.rent('affine-pe-cap-b', ttl='4h', max_eval_nodes=count + 1)
out['rent_below_cap'] = got and got['name']
print(json.dumps(out))
'''

# Runs in the shadow process: the real PIT rent service on the Lium helpers of the real host module. The
# process sees the live directories (it reads live state), and the hard deny of shadow mode is on before a
# live module loads: a write outside the shadow state root stops the run.
SHADOW_SIDE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from ops.parallel_eval.pit import PREFIX_LIVE, shadow_guard
guard = shadow_guard.enter_shadow(scratch=())
import pe_rent
from ops.parallel_eval.pit.durable import pit_state_root
from ops.parallel_eval.pit.isolated_rent import IsolatedRentError
from ops.parallel_eval.pit.pe_rent import RentService, bind_host_rent_service
from ops.parallel_eval.pit.shadow import ShadowWriteError
svc = RentService(max_boxes=2, shadow=True, name_prefix=sys.argv[2], share_pe_cap=False, require_host=True,
                  gpu_order=('RTX5090', 'H100'), price_ceiling_usd_h=56.0)
bind_host_rent_service(svc, module_path='pe_rent')      # the call of the PIT controller
isolated = svc.host_rent.__self__
out = dict(rent_state=str(pit_state_root() / 'isolated-rent'), listed=sorted(p['name'] for p in svc.list_pods()))
out['reconcile'] = svc.reconcile(item_box_ids=())
box = svc.rent(item_id='i' * 32, epoch=7, attempt=1)
out['box'] = {k: box.get(k) for k in ('name', 'gpu', 'price_h', 'auto_backup_disabled')}
out['listed_after_rent'] = sorted(p['name'] for p in svc.list_pods())
live = {p['name']: p['id'] for p in pe_rent.lium_ps() if p['name'].startswith('affine-')}
out['release_of_live'] = {n: [isolated.release(n, 'test'), isolated.release(i, 'test')] for n, i in live.items()}
try:
    isolated.rent(PREFIX_LIVE + '9-a0')
    out['rent_of_live_name'] = 'rented'
except IsolatedRentError:
    out['rent_of_live_name'] = 'refused'
out['released'] = svc.release(box['box_id'], reason='done')['ok']
out['listed_after_release'] = sorted(p['name'] for p in svc.list_pods())
out['denied_by_the_shadow_path'] = list(guard.denied)
# The rent path of live keeps its lock and its pod ledger in live state: the hard deny stops it.
try:
    pe_rent.rent('affine-pe-from-shadow', ttl='4h')
    out['live_rent_path'] = 'rented'
except ShadowWriteError:
    out['live_rent_path'] = 'denied'
out['denied_by_the_live_path'] = guard.denied[len(out['denied_by_the_shadow_path']):]
print(json.dumps(out))
'''


def tree(root: Path) -> dict:
    """Every file below ``root`` with its bytes."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


class TestNameRules(unittest.TestCase):
    def test_the_isolated_prefixes_are_not_live_names(self):
        self.assertEqual((PREFIX_SHADOW, PREFIX_FAKEPASS), ('sn120-shadow-pit-', 'sn120-fakepass-pit-'))
        self.assertEqual(ISOLATED_PREFIXES, (PREFIX_SHADOW, PREFIX_FAKEPASS))
        for prefix in ISOLATED_PREFIXES:
            self.assertFalse(prefix.startswith(LIVE_NAME_ROOT))
            self.assertIsNone(live_name_problem(prefix + '7-a0'))
        self.assertFalse(PREFIX_SHADOW.startswith(PREFIX_FAKEPASS) or PREFIX_FAKEPASS.startswith(PREFIX_SHADOW))
        for name in (LANE, LIVE_PIT, 'Affine-pe-x', 'affine-bench-1', '', None) + LEGACY_ISOLATED_PREFIXES:
            self.assertIsNotNone(live_name_problem(name), name)

    def test_the_name_classes_do_not_overlap(self):
        self.assertTrue(is_shadow_pit_name(SHADOW) and is_pit_name(SHADOW))
        self.assertTrue(is_fakepass_name(FAKEPASS))
        self.assertTrue(is_live_pit_name(LIVE_PIT) and is_pit_name(LIVE_PIT))
        for name in ISOLATED + tuple(p + '7-a0' for p in LEGACY_ISOLATED_PREFIXES):
            self.assertFalse(is_live_pit_name(name), name)
        self.assertFalse(is_fakepass_name(SHADOW) or is_shadow_pit_name(FAKEPASS) or is_pit_name(FAKEPASS))

    def test_no_role_and_no_job_kind_matches_an_isolated_name(self):
        roles = json.loads(ROLES.read_text())
        self.assertEqual(roles['pit-box']['shadow_name'], PREFIX_SHADOW)
        self.assertEqual(roles['pit-box']['shadow_name_glob'], PREFIX_SHADOW + '*')
        self.assertEqual(roles['pit-box']['fakepass_name'], PREFIX_FAKEPASS)
        self.assertEqual(roles['pit-box']['fakepass_name_glob'], PREFIX_FAKEPASS + '*')
        excluded = re.compile(roles['_excluded']['names_re'], re.I)
        for name in ISOLATED:
            self.assertIsNone(excluded.search(name))        # replace_pod leaves "excluded" pods to their owners
            self.assertFalse(name.startswith('affine-'))    # replace_pod status lists unmapped "affine-" pods
            for role, spec in roles.items():
                if role.startswith('_'):
                    continue
                live_names = [spec.get('name')] + list(spec.get('names') or [])
                self.assertFalse(any(n and name.startswith(n) for n in live_names), (role, name))
                self.assertFalse(spec.get('name_glob') and fnmatch.fnmatch(name, spec['name_glob']), (role, name))
            self.assertFalse(any(name.startswith(p) for p in NAME_PREFIX.values()))
            self.assertFalse(any(name.startswith(p) for p in NEVER_NAME_PREFIXES))

    def test_the_isolated_rent_refuses_a_live_prefix(self):
        for prefix in (PREFIX_LIVE, 'affine-pe-', 'affine-', 'Affine-shadow-') + LEGACY_ISOLATED_PREFIXES:
            with self.assertRaises(IsolatedRentError, msg=prefix):
                IsolatedRent(prefix=prefix, state_dir=Path('unused'), lium=None, lium_ps=None, pick_node=None)

    def test_a_shadow_service_refuses_a_live_prefix_before_it_calls_the_host(self):
        calls = []
        for prefix in (PREFIX_LIVE, 'affine-pe-', LEGACY_ISOLATED_PREFIXES[0]):
            service = RentService(shadow=True, name_prefix=prefix, require_host=True,
                                  host_rent=lambda *a, **k: calls.append(a))
            with self.assertRaisesRegex(RentError, 'live tools match'):
                service.rent(item_id='i' * 32, epoch=7, attempt=0)
        self.assertEqual(calls, [])

    def test_a_shadow_service_reads_the_state_root_when_it_rents_and_not_when_it_binds(self):
        """The controller binds the rent path when it starts. A state root that fails closed refuses the rent."""
        calls = []
        host = types.SimpleNamespace(
            lium=lambda *args, **kw: calls.append(args) or (1, '', 'not called'),
            lium_ps=lambda: [dict(id='id-1', name=SHADOW), dict(id='id-2', name=LANE)],
            pick_node=lambda *args, **kw: calls.append(args) or (None, None))
        service = RentService(shadow=True, name_prefix=PREFIX_SHADOW)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ):
            os.environ.pop('PIT_SHADOW_STATE', None)
            os.environ.update(PIT_STATE=os.path.join(tmp, 'pit'), PE_STATE=os.path.join(tmp, 'pe'))
            with self.assertRaisesRegex(RuntimeError, 'fail closed'):
                pit_state_root()                                    # the two names disagree: no valid state root
            bind_isolated_rent_service(service, host)
            self.assertEqual([pod['name'] for pod in service.list_pods()], [SHADOW])
            for use in (lambda: service.host_rent(SHADOW), lambda: service.host_release(SHADOW, 'test')):
                with self.assertRaisesRegex(RuntimeError, 'fail closed'):
                    use()
            self.assertEqual((calls, os.listdir(tmp)), ([], []))    # no Lium call and no file
            # The same service with a valid root: its files are in that root.
            os.environ['PIT_SHADOW_STATE'] = tmp
            self.assertIsNone(service.host_rent(SHADOW))            # Lium has no offer
            self.assertEqual(sorted(os.listdir(os.path.join(tmp, 'pit', 'isolated-rent'))), ['events.jsonl', 'rent.lock'])
            self.assertEqual(os.listdir(tmp), ['pit'])

    def test_each_service_reconciles_the_boxes_of_its_own_prefix_only(self):
        account = [dict(id='id-%d' % n, name=name) for n, name in enumerate((
            LANE, LIVE_PIT, SHADOW, FAKEPASS, 'affine-pe-bench-1', 'affine-pe-warm-1',
            LEGACY_ISOLATED_PREFIXES[0] + '3-a0', LEGACY_ISOLATED_PREFIXES[1] + '3-a0'))]
        by_name = {row['name']: row['id'] for row in account}
        cases = {
            LIVE_PIT: dict(shadow=False, name_prefix=PREFIX_LIVE),
            SHADOW: dict(shadow=True, name_prefix=PREFIX_SHADOW),
            FAKEPASS: dict(shadow=True, name_prefix=PREFIX_FAKEPASS),
        }
        for own, kwargs in cases.items():
            released = []
            service = RentService(host_release=lambda box, reason: released.append((box, reason)),
                                  list_pods=lambda: list(account), **kwargs)
            # No item holds a box: every box of the service is an orphan. A box of a different service is not.
            self.assertEqual(service.reconcile(item_box_ids=()), dict(adopted=[], released=[by_name[own]]), own)
            self.assertEqual(released, [(by_name[own], 'reconcile-orphan')], own)
            # A box that an item holds is adopted by its own service only.
            adopt = RentService(list_pods=lambda: list(account), **kwargs)
            self.assertEqual(adopt.reconcile(item_box_ids=by_name.values())['adopted'], [by_name[own]], own)
            self.assertEqual(sorted(adopt.ledger.active), [by_name[own]], own)


class TwoProcessCase(unittest.TestCase):
    """One Lium account, a live process and a shadow process.

    The two processes see the same live directories, as on the host. The shadow process also has its shadow
    state root, which is in a different place.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.account = self.root / 'home' / '.lium' / 'bin'      # the live code runs $HOME/.lium/bin/lium
        self.account.mkdir(parents=True)
        for name, body in (('lium', FAKE_LIUM), ('ssh', '#!/bin/sh\nexit 0\n')):
            path = self.account / name
            path.write_text(body)
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        (self.account / 'calls.jsonl').write_text('')
        self.set_offers(RTX5090=40.0, H100=30.0, B300=50.0)
        self.set_pods(LANE, LIVE_PIT, SHADOW, FAKEPASS)

    def set_offers(self, **price_of_gpu):
        (self.account / 'offers.json').write_text(json.dumps([
            dict(huid='exec-%s' % gpu.lower(), gpu_model=gpu, price_per_hour=price, gpu_count=8,
                 available_gpu_count=8, country_code='US', tier='t1') for gpu, price in price_of_gpu.items()]))

    def set_pods(self, *names):
        (self.account / 'pods.json').write_text(json.dumps([
            dict(id='id-%s' % name, name=name, huid='exec-old', ip='127.0.0.1', ssh_port=22, status='RUNNING',
                 price_per_hour=40.0, gpu_model='B300') for name in names]))

    def pods(self):
        return sorted(p['name'] for p in json.loads((self.account / 'pods.json').read_text()))

    def calls(self, command):
        rows = [json.loads(line) for line in (self.account / 'calls.jsonl').read_text().splitlines()]
        return [row for row in rows if row[0] == command]

    @property
    def live(self) -> Path:
        """The live directories: PE state, jobs, gate state, lifecycle and the flag."""
        return self.root / 'live'

    @property
    def shadow_state(self) -> Path:
        return self.root / 'shadow-state'

    def write_flag(self, flag) -> Path:
        (self.live / 'state').mkdir(parents=True, exist_ok=True)
        path = self.live / 'flag.json'
        path.write_text(json.dumps(flag))
        return path

    def run_process(self, script: str, *args: str, flag=None, shadow: bool = False):
        """Run ``script`` in a new process with the live directories. ``shadow``: it also gets the shadow state root."""
        here = self.live
        flag_path = self.write_flag(flag or {})
        # On the host, pe_common finds replace_pod in the checkout below $HOME. Here $HOME is the test account.
        env = dict(
            PATH='%s:/usr/bin:/bin' % self.account, HOME=str(self.root / 'home'),
            PYTHONPATH='%s:%s' % (REPO, ROLES.parent),
            PODS_ROLES=str(ROLES), PODS_STATE=str(here / 'pods'), PE_HERE=str(here), PE_ST=str(here / 'state'),
            PE_JOBS=str(here / 'jobs'), PE_FLAG=str(flag_path), HG_GATE_STATE=str(here / 'gate'),
            PE_LIFE=str(here / 'life'),
        )
        if shadow:
            env['PIT_SHADOW_STATE'] = str(self.shadow_state)
        done = subprocess.run([sys.executable, '-c', script, str(LIVE_CODE), *args], env=env, cwd=str(REPO),
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        return json.loads(done.stdout.strip().splitlines()[-1])


class TestLiveNeverMatches(TwoProcessCase):
    def test_live_does_not_list_count_rent_or_release_an_isolated_box(self):
        out = self.run_process(LIVE_SIDE, json.dumps(ISOLATED))
        self.assertEqual(out['our_pods'], sorted([LANE, LIVE_PIT]))
        self.assertEqual(out['eval_nodes'], sorted([LANE, LIVE_PIT]))
        self.assertEqual(out['rent'], {SHADOW: 'refused', FAKEPASS: 'refused'})
        self.assertEqual(out['release'], {SHADOW: [False, False], FAKEPASS: [False, False]})
        self.assertEqual(self.calls('rm'), [])
        # The cap of live counts the 2 live nodes only. The 2 isolated boxes take no slot.
        self.assertEqual(out['rent_at_cap'], 'cap reached')
        self.assertEqual(out['rent_below_cap'], 'affine-pe-cap-b')
        self.assertEqual([row[row.index('--name') + 1] for row in self.calls('up')], ['affine-pe-cap-b'])
        self.assertEqual(self.pods(), sorted([LANE, LIVE_PIT, SHADOW, FAKEPASS, 'affine-pe-cap-b']))


class TestShadowRentPath(TwoProcessCase):
    FLAG = {'parallel_eval': {'pit': {'enabled': True, 'mode': 'shadow', 'allow_sharded_gpus': True,
                                      'gpu_order': ['RTX5090', 'H100']}}}

    def shadow(self, prefix=PREFIX_SHADOW):
        return self.run_process(SHADOW_SIDE, prefix, flag=self.FLAG, shadow=True)

    def test_shadow_rents_lists_and_releases_its_own_boxes_and_no_live_box(self):
        self.write_flag(self.FLAG)
        (self.live / 'state' / 'pods.json').write_text('{"live": true}\n')
        before = tree(self.live)
        out = self.shadow()
        # The orphan of an earlier shadow run is released. The live pods and the fakepass box stay.
        self.assertEqual(out['listed'], [SHADOW])
        self.assertEqual(out['reconcile'], dict(adopted=[], released=['id-' + SHADOW]))
        name = PREFIX_SHADOW + '7-a1'
        self.assertEqual(out['box'], dict(name=name, gpu='RTX5090', price_h=40.0, auto_backup_disabled=True))
        self.assertEqual(out['listed_after_rent'], [name])
        self.assertEqual(out['release_of_live'], {LANE: [False, False], LIVE_PIT: [False, False]})
        self.assertEqual(out['rent_of_live_name'], 'refused')
        self.assertTrue(out['released'])
        self.assertEqual(out['listed_after_release'], [])
        self.assertEqual(self.pods(), sorted([LANE, LIVE_PIT, FAKEPASS]))
        up = self.calls('up')
        self.assertEqual(len(up), 1)
        self.assertEqual(up[0][:4] + up[0][4:6], ['up', 'exec-rtx5090', '-c', '8', '--name', name])
        self.assertIn('--no-backup', up[0])
        self.assertEqual(up[0][up[0].index('--ttl') + 1], '4h')
        self.assertEqual(len(self.calls('rm')), 2)             # the orphan and the box of this run
        # The hard deny refused no write of the shadow rent path: its lock, pod ledger and events are in the
        # shadow state root, and no other file was written there.
        self.assertEqual(out['denied_by_the_shadow_path'], [])
        rent_state = self.shadow_state / 'pit' / 'isolated-rent'
        self.assertEqual(out['rent_state'], str(rent_state))
        self.assertEqual(sorted(tree(self.shadow_state)), [
            'pit/isolated-rent/events.jsonl', 'pit/isolated-rent/pods.json', 'pit/isolated-rent/rent.lock'])
        self.assertTrue(json.loads((rent_state / 'pods.json').read_text())[name]['released'])
        # The rent path of live can not run in the shadow process: its first write (the rent lock in live
        # state) is refused, and it rents nothing.
        self.assertEqual(out['live_rent_path'], 'denied')
        self.assertEqual(len(out['denied_by_the_live_path']), 1)
        self.assertIn(str(self.live / 'state' / 'rent.lock'), out['denied_by_the_live_path'][0])
        self.assertEqual(len(self.calls('up')), 1)
        # Live state did not change.
        self.assertEqual(tree(self.live), before)

    def test_h100_is_the_fallback_when_no_rtx5090_offer_is_under_the_price_ceiling(self):
        self.set_offers(RTX5090=57.0, H100=30.0, B300=20.0)
        out = self.shadow()
        self.assertEqual((out['box']['gpu'], out['box']['price_h']), ('H100', 30.0))
        self.assertEqual(self.calls('up')[0][1], 'exec-h100')

    def test_the_fakepass_prefix_has_the_same_isolation(self):
        out = self.shadow(PREFIX_FAKEPASS)
        self.assertEqual(out['listed'], [FAKEPASS])
        self.assertEqual(out['box']['name'], PREFIX_FAKEPASS + '7-a1')
        self.assertEqual(out['release_of_live'], {LANE: [False, False], LIVE_PIT: [False, False]})
        self.assertEqual(self.pods(), sorted([LANE, LIVE_PIT, SHADOW]))


if __name__ == '__main__':
    unittest.main()
