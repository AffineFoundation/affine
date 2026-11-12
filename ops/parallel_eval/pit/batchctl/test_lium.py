"""The Lium provider against a fake command line.

Nothing here talks to Lium. What the tests fix: the exact command forms, the
limits of an offer, the cap, and which failure means "no box" and which
means "a box may exist".
"""
from __future__ import annotations

import threading

import pytest

from . import BOX_PREFIX_LIVE, BOX_PREFIX_SHADOW, MODE_LIVE, MODE_SHADOW
from . import guard as G
from . import lium as L
from ..shadow import ShadowWriteError
from .interfaces import BoxInfo, ProviderError, RentUnavailable
from .test_fakes import FakeLium, offer

RUN = 'run0001'
NAME = G.box_name(RUN, 1, 0)
HOURS8 = 8 * 3600.0


def provider(world: FakeLium, **changes) -> L.LiumProvider:
    never = changes.pop('never', None)
    settings = dict(run_id=RUN, gpus=('RTX5090',))
    settings.update(changes)
    return L.LiumProvider(
        world, L.LiumConfig(**settings), never=never,
        monotonic=world.clock.monotonic, sleep=world.clock.sleep,
    )


def rent(p: L.LiumProvider, name: str = NAME, **changes) -> BoxInfo:
    args = dict(
        name=name, gpu='RTX5090', tier='secure', gpu_count=8, max_price_usd_h=56.0, ttl_s=HOURS8,
        timeout_s=900.0,
    )
    args.update(changes)
    return p.rent(**args)


# ----------------------------------------------------------------------------
# The rental
# ----------------------------------------------------------------------------

def test_a_rental_takes_the_cheapest_good_node_by_its_exact_id():
    world = FakeLium().add(offer('node-a', price=5.0), offer('node-b', price=3.5), offer('node-c', price=4.0))
    box = rent(provider(world))
    assert box == BoxInfo('pod-0001', NAME, 'lium', 'RTX5090', 8, 'secure', 28.0)
    assert world.verbs('up') == [(
        'up', 'node-b', '-c', '8', '--name', NAME, '--verify-gpus', '--strict-gpus',
        '--no-volume-encryption', '--no-backup', '-y', '--json', '--timeout', '860', '--ttl', '8h',
    )]
    assert world.pods['pod-0001']['huid'] == 'node-b'


def test_the_command_forms_are_the_ones_of_the_ops_code():
    world = FakeLium().add(offer('node-a'))
    p = provider(world)
    box = rent(p)
    p.list()
    p.destroy(box.box_id)
    assert [call for call in world.calls if call[0] != 'up'] == [
        ('ps', '--format', 'json'),
        ('ls', '--gpu', 'RTX5090', '--format', 'json'),
        ('ps', '--format', 'json'),
        ('ps', '--format', 'json'),
        ('rm', 'pod-0001', '-y'),
    ]
    # The command gets less time than the call, and the caller of the command a little more.
    assert dict(world.timeouts) == {'ps': 90.0, 'ls': 120.0, 'up': 880, 'rm': 180.0}


def test_the_ttl_is_rounded_up_to_whole_hours():
    world = FakeLium().add(offer('node-a'), offer('node-b'))
    p = provider(world)
    rent(p, ttl_s=3601.0)
    rent(p, G.box_name(RUN, 2, 0), ttl_s=60.0)
    assert [call[-1] for call in world.verbs('up')] == ['2h', '1h']


def test_a_faster_download_breaks_a_price_tie():
    world = FakeLium().add(offer('node-a', download_mbps=100), offer('node-b', download_mbps=900))
    rent(provider(world))
    assert world.verbs('up')[0][1] == 'node-b'


@pytest.mark.parametrize('changes, reason', [
    (dict(tier='spot'), 'tier'),
    (dict(count=10), 'not_a_free_whole_node'),
    (dict(count=4), 'not_a_free_whole_node'),
    (dict(free=7), 'not_a_free_whole_node'),
    (dict(available_gpu_count=None), 'not_a_free_whole_node'),
    (dict(gpu_power_limited=True), 'power_limited'),
    (dict(price=7.01), 'price'),
    (dict(price_per_hour=56.5), 'price'),
    (dict(price_per_gpu_hour=None, price_per_hour=None), 'price'),
    (dict(price_per_gpu_hour=-1.0, price_per_hour=None), 'price'),
    (dict(price_per_gpu_hour='free', price_per_hour=None), 'price'),
    (dict(ram=299.9), 'ram'),
    (dict(ram=64), 'ram'),
    (dict(ram=None), 'ram_unknown'),
    (dict(ram='lots'), 'ram_unknown'),
    (dict(ram=True), 'ram_unknown'),
    (dict(ram=10 ** 7), 'ram_unknown'),
    (dict(ram=0), 'ram_unknown'),
    (dict(disk_gb=100), 'disk'),
    (dict(download_mbps=50), 'download'),
], ids=lambda value: value if isinstance(value, str) else ','.join('%s=%s' % item for item in value.items()))
def test_an_offer_outside_the_limits_is_never_rented(changes, reason):
    world = FakeLium().add(offer('node-a', **changes))
    p = provider(world, min_disk_gb=500, min_download_mbps=100)
    assert p.survey(gpu='RTX5090', tier='secure', max_price_usd_h=56.0)['reasons'] == {reason: 1}
    with pytest.raises(RentUnavailable):
        rent(p)
    assert not world.verbs('up') and not world.pods


@pytest.mark.parametrize('row, reason', [
    ({key: value for key, value in offer('node-a').items() if key != 'huid'}, 'no_id'),
    (dict(offer('node-a'), huid=7), 'no_id'),
    (dict(offer('node-a'), huid=''), 'no_id'),
    (offer('node-a', gpu='H100'), 'gpu_class'),
    (offer('node-a', gpu='B300'), 'gpu_class'),
    (offer('baseline-node'), 'never_list'),
    (offer('node-a', name='affine-baseline-1'), 'never_list'),
    (offer('node-x'), 'never_list'),
], ids=lambda value: value if isinstance(value, str) else None)
def test_an_offer_that_must_not_be_touched_has_a_reason(row, reason):
    p = provider(FakeLium(), never_names=('baseline',), never=lambda row: row.get('huid') == 'node-x')
    assert p.why_not(row, gpu='RTX5090', tier='secure', max_price_usd_h=56.0) == reason
    assert p.choose([row], gpu='RTX5090', tier='secure', max_price_usd_h=56.0) == []


@pytest.mark.parametrize('word', [
    'brave-shark-ec', 'noble-wolf-26', 'calm-matrix-56', 'noble-fox-5c', 'gentle-wolf-f6', 'noble-shark-d6',
])
def test_the_machines_of_other_work_are_never_rented_and_never_removed_by_default(word):
    assert word in L.NEVER_NAMES_DEFAULT and L.LiumConfig(run_id=RUN, gpus=('RTX5090',)).never_names == L.NEVER_NAMES_DEFAULT
    world = FakeLium().add(offer(word, price=1.0), offer('node-b', price=5.0))
    p = provider(world)
    assert p.why_not(offer(word), gpu='RTX5090', tier='secure', max_price_usd_h=56.0) == 'never_list'
    assert p.why_not(offer('node-a', name=word), gpu='RTX5090', tier='secure', max_price_usd_h=56.0) == 'never_list'
    # The cheaper node is protected: the rental takes the other one.
    assert world.pods[rent(p).box_id]['huid'] == 'node-b'


def test_ram_of_exactly_the_minimum_is_enough_and_the_highest_price_at_the_ceiling_too():
    world = FakeLium().add(offer('node-a', ram=300, price=7.0))
    assert rent(provider(world)).price_usd_h == 56.0


def test_the_ram_can_stand_in_another_field_and_unit():
    row = offer('node-a', ram=None, specs={'ram': {'total_kb': 512 * 1024 * 1024}})
    world = FakeLium().add(row)
    with pytest.raises(RentUnavailable):
        rent(provider(world))
    p = provider(world, ram_fields=(('specs.ram.total_kb', 1024.0 * 1024.0),))
    assert L.ram_gb(row, p.cfg.ram_fields) == 512.0
    assert rent(p).box_id == 'pod-0001'


def test_without_a_ram_minimum_the_field_is_not_needed():
    world = FakeLium().add(offer('node-a', ram=None))
    assert rent(provider(world, min_ram_gb=0.0)).box_id == 'pod-0001'


def test_secure_and_spot_are_never_mixed():
    world = FakeLium().add(offer('cheap-spot', tier='spot', price=1.0), offer('dear-secure', price=6.0))
    p = provider(world)
    assert rent(p).tier == 'secure' and world.verbs('up')[0][1] == 'dear-secure'
    second = rent(p, G.box_name(RUN, 2, 0), tier='spot')
    assert second.tier == 'spot' and world.verbs('up')[1][1] == 'cheap-spot'
    with pytest.raises(RentUnavailable):
        rent(p, G.box_name(RUN, 3, 0), tier='secure')


@pytest.mark.parametrize('changes', [
    dict(gpu='B300'), dict(gpu='H100'), dict(gpu=''), dict(tier='community'), dict(gpu_count=4),
    dict(ttl_s=0.0), dict(timeout_s=0.0),
], ids=str)
def test_a_request_outside_the_run_rents_nothing_and_asks_nothing(changes):
    world = FakeLium().add(offer('node-a'), offer('node-b', gpu='B300'), offer('node-c', gpu='H100'))
    with pytest.raises(RentUnavailable):
        rent(provider(world), **changes)
    assert world.calls == []


def test_the_gpu_list_of_the_run_decides_what_can_be_rented():
    world = FakeLium().add(offer('node-a', gpu='H100'))
    with pytest.raises(RentUnavailable):
        rent(provider(world), gpu='H100')
    assert rent(provider(world, gpus=('RTX5090', 'H100')), gpu='H100').gpu == 'H100'


@pytest.mark.parametrize('name', [
    'affine-pe-pit-e1', 'affine-pe-lane-1', 'affine-trainer-1', G.box_name('otherrun', 1, 0),
    NAME + '-x', BOX_PREFIX_SHADOW + RUN, '', 'x',
])
def test_a_name_that_this_run_did_not_make_is_refused(name):
    world = FakeLium().add(offer('node-a'))
    with pytest.raises(ShadowWriteError):
        rent(provider(world), name)
    assert world.calls == []


def test_a_protected_word_in_the_name_is_refused():
    world = FakeLium().add(offer('node-a'))
    with pytest.raises(ShadowWriteError):
        rent(provider(world, never_names=(RUN,)))
    assert world.calls == []


# ----------------------------------------------------------------------------
# The cap
# ----------------------------------------------------------------------------

def test_the_cap_counts_every_shadow_box_of_the_account():
    world = FakeLium().add(offer('node-a'), offer('node-b'))
    for n in range(7):
        world.pod(G.box_name('otherrun', n + 1, 0))
    world.pod('affine-pe-lane-1')
    world.pod('affine-trainer-prod')
    p = provider(world)
    assert rent(p).name == NAME
    with pytest.raises(RentUnavailable, match='cap of 8'):
        rent(p, G.box_name(RUN, 2, 0))
    # At the cap nothing is asked beyond the pod list.
    assert world.calls[-1] == ('ps', '--format', 'json')
    assert len(world.verbs('up')) == 1 and world.most == 10


def test_the_cap_can_only_be_set_per_run():
    world = FakeLium().add(*[offer('node-%d' % n) for n in range(4)])
    p = provider(world, max_boxes=2)
    rent(p)
    rent(p, G.box_name(RUN, 2, 0))
    with pytest.raises(RentUnavailable):
        rent(p, G.box_name(RUN, 3, 0))
    p.destroy('pod-0001')
    assert rent(p, G.box_name(RUN, 3, 0)).box_id == 'pod-0003'


class Gate:
    """Holds every ``up`` until the test opens the gate."""

    def __init__(self, world: FakeLium):
        self.world = world
        self.open = threading.Event()
        self.waiting = 0
        self.lock = threading.Lock()

    def __call__(self, *args, timeout=180):
        if args[0] == 'up':
            with self.lock:
                self.waiting += 1
            assert self.open.wait(30)
        return self.world(*args, timeout=timeout)


def test_the_cap_holds_when_many_rentals_run_at_once():
    world = FakeLium().add(*[offer('node-%02d' % n) for n in range(12)])
    gate = Gate(world)
    p = L.LiumProvider(gate, L.LiumConfig(run_id=RUN, gpus=('RTX5090',)))
    results = {}

    def one(n: int) -> None:
        try:
            results[n] = rent(p, G.box_name(RUN, n + 1, 0))
        except Exception as exc:
            results[n] = exc

    threads = [threading.Thread(target=one, args=(n,)) for n in range(12)]
    for thread in threads:
        thread.start()
    # Eight rentals are in flight and no pod exists yet. The other four must see them.
    for thread in threads:
        thread.join(0.01)
    while sum(1 for thread in threads if thread.is_alive()) > 8:
        for thread in threads:
            thread.join(0.01)
    assert gate.waiting == 8 and not world.pods
    gate.open.set()
    for thread in threads:
        thread.join(30)
    boxes = [value for value in results.values() if isinstance(value, BoxInfo)]
    refused = [value for value in results.values() if isinstance(value, RentUnavailable)]
    assert len(boxes) == 8 and len(refused) == 4 and world.most == 8
    # Eight different nodes: a node of a rental in flight is not offered twice.
    assert len({call[1] for call in world.verbs('up')}) == 8


# ----------------------------------------------------------------------------
# A box that exists, and an answer that nobody got
# ----------------------------------------------------------------------------

def test_a_box_that_already_has_the_name_is_taken_and_nothing_is_rented():
    world = FakeLium().add(offer('node-a'), offer('node-b'))
    first = rent(provider(world))
    again = rent(provider(world))
    assert again == first and len(world.verbs('up')) == 1 and len(world.pods) == 1


@pytest.mark.parametrize('word', ['no_json', 'fail_created', 'raise_created', 'slow:3', 'late:2'])
def test_a_box_that_was_made_is_found_by_its_name(word):
    world = FakeLium().add(offer('node-a'))
    world.script['up'].append(word)
    box = rent(provider(world))
    assert box == BoxInfo('pod-0001', NAME, 'lium', 'RTX5090', 8, 'secure', 32.0)
    assert len(world.verbs('up')) == 1


def test_a_request_that_is_still_running_gets_more_time():
    world = FakeLium().add(offer('node-a'))
    world.script['up'].append('slow:20')
    assert rent(provider(world)).box_id == 'pod-0001'
    assert sum(world.clock.slept) == 190.0
    # Without those words the wait is short, and the box shows up later.
    world = FakeLium().add(offer('node-a'))
    world.script['up'].append('late:20')
    with pytest.raises(ProviderError):
        rent(provider(world))
    assert sum(world.clock.slept) == 50.0


@pytest.mark.parametrize('word', ['fail', 'raise', 'late:50', 'other_name', (0, '{"pod": 7}', ''), (0, '[]', '')],
                         ids=str)
def test_after_the_rental_was_sent_a_failure_never_means_no_box(word):
    world = FakeLium().add(offer('node-a'), offer('node-b'))
    world.script['up'].append(word)
    p = provider(world)
    with pytest.raises(ProviderError) as caught:
        rent(p)
    assert not isinstance(caught.value, RentUnavailable)
    assert len(world.verbs('up')) == 1
    if word == 'late:50':
        # The pod shows up after the call gave up. The listing finds it by name, so the sweep can remove it.
        for _ in range(60):
            found = [box for box in p.list() if box.name == NAME]
        assert [box.box_id for box in found] == ['pod-0001']
        assert G.parse_box_name(RUN, found[0].name) == (1, 0)


@pytest.mark.parametrize('verb, step', [
    ('ps', (1, '', 'Error: not logged in')),
    ('ps', TimeoutError('no answer')),
    ('ps', (0, 'not json', '')),
    ('ps', (0, '{"pods": []}', '')),
    ('ps', (0, '[7]', '')),
    ('ps', (0, '', '')),
    ('ls', (1, '', 'Error: bad gateway')),
    ('ls', OSError('no command line')),
    ('ls', (0, '<html>', '')),
], ids=str)
def test_a_failure_before_the_rental_was_sent_means_no_box(verb, step):
    world = FakeLium().add(offer('node-a'))
    world.script[verb].append(step)
    with pytest.raises(RentUnavailable):
        rent(provider(world))
    assert not world.verbs('up') and not world.pods


def test_too_little_time_rents_nothing():
    world = FakeLium().add(offer('node-a'))
    with pytest.raises(RentUnavailable, match='no time'):
        rent(provider(world), timeout_s=89.0)
    assert not world.verbs('up')
    assert rent(provider(world), timeout_s=90.0).box_id == 'pod-0001'
    assert world.verbs('up')[0][world.verbs('up')[0].index('--timeout') + 1] == '50'


def test_one_rental_sends_one_command_unless_the_run_allows_a_second_node():
    world = FakeLium().add(offer('node-a', price=3.0), offer('node-b', price=4.0))
    world.script['up'].append('fail')
    with pytest.raises(ProviderError):
        rent(provider(world))
    assert [call[1] for call in world.verbs('up')] == ['node-a']

    world = FakeLium().add(offer('node-a', price=3.0), offer('node-b', price=4.0))
    world.script['up'].append('fail')
    box = rent(provider(world, nodes_per_rent=2))
    assert box.price_usd_h == 32.0 and [call[1] for call in world.verbs('up')] == ['node-a', 'node-b']
    # The second command gets the time that is left after the look for the first pod.
    assert [call[call.index('--timeout') + 1] for call in world.verbs('up')] == ['860', '810']


def test_a_node_that_gave_no_pod_is_not_tried_again_by_this_process():
    world = FakeLium().add(offer('node-a', price=3.0), offer('node-b', price=4.0))
    world.script['up'].append('fail')
    p = provider(world)
    with pytest.raises(ProviderError):
        rent(p)
    assert rent(p, G.box_name(RUN, 1, 1)).price_usd_h == 32.0
    assert [call[1] for call in world.verbs('up')] == ['node-a', 'node-b']


def test_the_same_name_cannot_be_rented_twice_at_the_same_time():
    world = FakeLium().add(offer('node-a'), offer('node-b'))
    gate = Gate(world)
    p = L.LiumProvider(gate, L.LiumConfig(run_id=RUN, gpus=('RTX5090',)))
    first = []
    thread = threading.Thread(target=lambda: first.append(rent(p)))
    thread.start()
    while gate.waiting < 1:
        thread.join(0.01)
    with pytest.raises(ProviderError, match='in flight') as caught:
        rent(p)
    assert not isinstance(caught.value, RentUnavailable)
    gate.open.set()
    thread.join(30)
    assert first[0].box_id == 'pod-0001' and len(world.pods) == 1


# ----------------------------------------------------------------------------
# What the controller sees of a pod
# ----------------------------------------------------------------------------

def test_a_pod_that_hides_its_class_tier_or_price_is_shown_with_empty_fields():
    world = FakeLium(pod_shows=()).add(offer('node-a'))
    p = provider(world)
    # The process that rented knows the offer that it took.
    assert rent(p) == BoxInfo('pod-0001', NAME, 'lium', 'RTX5090', 8, 'secure', 32.0)
    assert p.list() == [BoxInfo('pod-0001', NAME, 'lium', 'RTX5090', 8, 'secure', 32.0)]
    # A new process does not. It shows nothing that it cannot see; the controller then refuses the box.
    assert provider(world).list() == [BoxInfo('pod-0001', NAME, 'lium', '', 0, '', -1.0)]


@pytest.mark.parametrize('row, expected', [
    (dict(gpu_type='RTX 5090'), 'RTX5090'),
    (dict(gpu_model='NVIDIA GeForce RTX 5090'), 'RTX5090'),
    (dict(config='8\u00d7RTX5090'), 'RTX5090'),
    (dict(gpu_name='5090'), 'RTX5090'),
    (dict(gpu='H100'), 'H100'),
    (dict(gpu_type='NVIDIA H100 PCIe'), 'H100'),
    (dict(gpu_type='B300'), ''),
    # The first field that names a class decides. A later field cannot overrule it.
    (dict(gpu_type='B300', config='8\u00d7RTX5090'), ''),
    (dict(gpu='', config='8\u00d7RTX5090'), 'RTX5090'),
    (dict(gpu_type=''), ''),
    (dict(name='affine-RTX5090'), ''),
    (dict(), ''),
], ids=str)
def test_the_gpu_class_is_read_from_what_the_row_says(row, expected):
    assert L.gpu_class(row, ('RTX5090', 'H100')) == expected


def test_the_gpu_count_can_stand_in_the_config_text():
    for config in ('8\u00d7RTX5090', '8x RTX 5090', ' 8 X NVIDIA GeForce RTX 5090'):
        world = FakeLium(pod_shows=())
        world.pod(NAME, config=config, tier='spot', price_per_hour=30.0)
        assert provider(world).list() == [BoxInfo('pod-0001', NAME, 'lium', 'RTX5090', 8, 'spot', 30.0)]
    world = FakeLium(pod_shows=())
    world.pod(NAME, config='RTX5090', tier='spot', price_per_hour=30.0)
    assert provider(world).list() == [BoxInfo('pod-0001', NAME, 'lium', 'RTX5090', 0, 'spot', 30.0)]


def test_a_tier_that_is_not_a_tier_is_not_shown():
    world = FakeLium()
    world.pod(NAME, tier='premium')
    assert provider(world).list()[0].tier == ''


# ----------------------------------------------------------------------------
# Release
# ----------------------------------------------------------------------------

def test_a_pod_is_removed_by_its_exact_id():
    world = FakeLium().add(offer('node-a'))
    p = provider(world)
    box = rent(p)
    p.destroy(box.box_id)
    assert world.verbs('rm') == [('rm', 'pod-0001', '-y')] and not world.pods
    # A pod that is gone is not an error, and no command is sent for it.
    p.destroy(box.box_id)
    p.destroy('')
    assert len(world.verbs('rm')) == 1


@pytest.mark.parametrize('name', [
    'affine-pe-lane-1', 'affine-pe-pit-e7', 'affine-trainer-prod', G.box_name('otherrun', 1, 0), NAME + '-old',
])
def test_a_pod_that_this_run_did_not_make_is_never_removed(name):
    world = FakeLium()
    pod = world.pod(name)
    with pytest.raises(ShadowWriteError):
        provider(world).destroy(pod['id'])
    assert not world.verbs('rm') and len(world.pods) == 1


def test_a_live_provider_and_a_shadow_provider_of_one_run_id_never_touch_the_boxes_of_each_other():
    live_name = G.box_name(RUN, 1, 0, MODE_LIVE)
    assert live_name == BOX_PREFIX_LIVE + RUN + '-f1-t0' and live_name != NAME
    world = FakeLium().add(offer('node-a'))
    shadow_pod = world.pod(NAME)
    live = provider(world, mode=MODE_LIVE, max_boxes=2)

    # One cap for the account: the shadow pod counts against the cap of the live run. Its name is not a live name.
    assert live.shadow_boxes(live.pods()) == 1
    with pytest.raises(ShadowWriteError):
        rent(live, NAME)
    assert not world.verbs('up')
    box = rent(live, live_name)
    up = world.verbs('up')[0]
    assert box.name == live_name and up[up.index('--name') + 1] == live_name
    assert live.shadow_boxes(live.pods()) == 2
    with pytest.raises(RentUnavailable, match='cap of 2'):
        rent(live, G.box_name(RUN, 2, 0, MODE_LIVE))

    # The live provider does not remove the shadow pod and does not give its address.
    with pytest.raises(ShadowWriteError):
        live.destroy(shadow_pod['id'])
    assert live.ssh_target(shadow_pod['id']) is None and live.ssh_target(box.box_id) is not None

    # The shadow provider of the same run id does the same with the live pod.
    shadow = provider(world, max_boxes=1)
    assert shadow.cfg.mode == MODE_SHADOW and shadow.shadow_boxes(shadow.pods()) == 2
    with pytest.raises(ShadowWriteError):
        rent(shadow, live_name)
    with pytest.raises(ShadowWriteError):
        shadow.destroy(box.box_id)
    assert shadow.ssh_target(box.box_id) is None and shadow.ssh_target(shadow_pod['id']) is not None
    assert not world.verbs('rm') and len(world.pods) == 2

    # Each provider removes its own pod.
    live.destroy(box.box_id)
    shadow.destroy(shadow_pod['id'])
    assert world.verbs('rm') == [('rm', box.box_id, '-y'), ('rm', shadow_pod['id'], '-y')] and not world.pods


def test_a_protected_pod_is_never_removed_even_with_a_name_of_this_run():
    world = FakeLium()
    pod = world.pod(NAME, huid='baseline-node')
    with pytest.raises(ShadowWriteError):
        provider(world, never_names=('baseline',)).destroy(pod['id'])
    with pytest.raises(ShadowWriteError):
        provider(world, never=lambda row: row['id'] == pod['id']).destroy(pod['id'])
    assert not world.verbs('rm')


@pytest.mark.parametrize('verb, step', [
    ('rm', (1, '', 'Error: bad gateway')), ('rm', TimeoutError('no answer')),
    ('ps', (1, '', 'Error: bad gateway')), ('ps', TimeoutError('no answer')),
], ids=str)
def test_a_removal_that_failed_is_an_error_that_the_controller_can_retry(verb, step):
    world = FakeLium().add(offer('node-a'))
    p = provider(world)
    box = rent(p)
    world.script[verb].append(step)
    with pytest.raises(ProviderError):
        p.destroy(box.box_id)
    assert len(world.pods) == 1
    p.destroy(box.box_id)
    assert not world.pods


def test_a_listing_that_failed_is_an_error():
    world = FakeLium()
    world.script['ps'].append((1, '', 'Error: not logged in'))
    with pytest.raises(ProviderError):
        provider(world).list()


# ----------------------------------------------------------------------------
# The address of a box
# ----------------------------------------------------------------------------

def test_the_address_of_a_box_is_read_for_the_ssh_helper():
    world = FakeLium().add(offer('node-a'))
    p = provider(world)
    box = rent(p)
    assert p.ssh_target(box.box_id) == {
        'id': 'pod-0001', 'ip': 'box-1.test', 'port': 22, 'huid': 'node-a', 'name': NAME,
    }
    other = world.pod('affine-pe-lane-1')
    assert p.ssh_target(other['id']) is None and p.ssh_target('pod-9999') is None


@pytest.mark.parametrize('row, expected', [
    (dict(ip='box.test', ssh_port='2200'), ('box.test', 2200)),
    (dict(ssh_ip='box.test', port=2201), ('box.test', 2201)),
    (dict(ip='box.test', ports={'22': 2202}), ('box.test', 2202)),
    (dict(ssh_command='ssh -p 2203 -o X=y root@box.test'), ('box.test', 2203)),
    (dict(ssh_command='ssh root@box.test -p2204'), ('box.test', 2204)),
    (dict(ip='box.test'), None),
    (dict(ssh_port=22), None),
    (dict(ip='box.test', ssh_port=0), None),
    (dict(ip='box.test', ssh_port=70000), None),
    (dict(ip='box.test', ssh_port='soon'), None),
    (dict(), None),
], ids=str)
def test_the_address_fields_of_a_pod(row, expected):
    target = L.ssh_target(dict(row, id='pod-1'))
    assert (None if target is None else (target['ip'], target['port'])) == expected


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

@pytest.mark.parametrize('changes', [
    dict(gpus=()), dict(gpus=('',)), dict(gpus=(7,)), dict(max_boxes=0), dict(gpu_count=0),
    dict(nodes_per_rent=0), dict(min_ram_gb=-1.0), dict(min_disk_gb=-1.0), dict(min_download_mbps=-1.0),
    dict(ram_fields=()), dict(ram_fields=(('ram_gb', 0.0),)), dict(ram_fields=(('', 1.0),)),
    dict(ram_fields=(('ram_gb',),)),
], ids=str)
def test_settings_out_of_range_are_refused(changes):
    settings = dict(run_id=RUN, gpus=('RTX5090',))
    settings.update(changes)
    with pytest.raises(ValueError):
        L.LiumConfig(**settings)


def test_the_mode_of_the_settings_is_shadow_or_live():
    assert L.LiumConfig(run_id=RUN, gpus=('RTX5090',)).mode == MODE_SHADOW
    assert L.LiumConfig(run_id=RUN, gpus=('RTX5090',), mode=MODE_LIVE).mode == MODE_LIVE
    for mode in ('LIVE', 'prod', '', None, 7):
        with pytest.raises(ValueError, match='mode'):
            L.LiumConfig(run_id=RUN, gpus=('RTX5090',), mode=mode)


def test_the_defaults_are_the_limits_of_the_shadow_run():
    cfg = L.LiumConfig(run_id=RUN, gpus=('RTX5090',))
    assert (cfg.max_boxes, cfg.gpu_count, cfg.min_ram_gb, cfg.nodes_per_rent) == (8, 8, 300.0, 1)
    assert '--no-backup' in cfg.up_flags and '--strict-gpus' in cfg.up_flags
