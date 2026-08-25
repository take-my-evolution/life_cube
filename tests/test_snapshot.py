import numpy as np

from life_cube.snapshot import (Tracker, describe_components, label_components,
                                pack_cells, unpack_cells, make_snapshot)
from life_cube import Config
from life_cube.sim import init_state
from life_cube.backend import get_backend


def test_pack_roundtrip():
    rng = np.random.default_rng(0)
    vol = (rng.random((20, 20, 20)) < 0.05).astype(np.int8) * rng.integers(1, 5, (20, 20, 20)).astype(np.int8)
    coords, vals = pack_cells(vol)
    assert coords.dtype == np.uint16 and len(coords) == np.count_nonzero(vol)
    back = unpack_cells(coords, vals, 20)
    assert np.array_equal(back, vol)


def _world(shapes):
    """shapes: список (вид, x0:x1, y0:y1, z0:z1)."""
    v = np.zeros((16, 16, 16), np.int8)
    for s, xs, ys, zs in shapes:
        v[xs, ys, zs] = s
    return v


def test_label_components_shapes():
    v = _world([(1, slice(2, 4), slice(2, 4), slice(0, 6)),     # столб
                (1, slice(8, 10), slice(8, 10), slice(0, 3)),   # второй столб
                (1, slice(8, 10), slice(8, 10), slice(4, 7)),   # разорван на z=3 -> отдельный
                (2, slice(4, 6), slice(2, 4), slice(0, 6))])    # другой вид вплотную к первому
    lab, k = label_components(v)
    assert k == 4
    lab_all, k_all = label_components(v, per_species=False)
    assert k_all == 3          # без учёта вида столб 1 и 2 сливаются
    # угловое касание — тоже связь (26-связность)
    w = np.zeros((5, 5, 5), np.int8); w[0, 0, 0] = 1; w[1, 1, 1] = 1
    assert label_components(w)[1] == 1


def test_tracker_identity_growth_split():
    t = Tracker()
    v1 = _world([(1, slice(2, 4), slice(2, 4), slice(0, 4))])
    ids1 = t.assign(label_components(v1)[0], gen=0)
    a = int(ids1[2, 2, 0]); assert a == 1
    # рост: тот же id
    v2 = _world([(1, slice(2, 4), slice(2, 4), slice(0, 7))])
    ids2 = t.assign(label_components(v2)[0], gen=1)
    assert int(ids2[2, 2, 6]) == a
    # деление: большая половина сохраняет id, меньшая — новый
    v3 = _world([(1, slice(2, 4), slice(2, 4), slice(0, 4)),
                 (1, slice(2, 4), slice(2, 4), slice(5, 7))])
    ids3 = t.assign(label_components(v3)[0], gen=2)
    assert int(ids3[2, 2, 0]) == a and int(ids3[2, 2, 6]) != a
    assert t.born[int(ids3[2, 2, 6])] == 2
    # совсем новый организм в другом месте — новый id, старые нетронуты
    v4 = _world([(1, slice(2, 4), slice(2, 4), slice(0, 4)),
                 (3, slice(10, 12), slice(10, 12), slice(0, 2))])
    ids4 = t.assign(label_components(v4)[0], gen=3)
    assert int(ids4[2, 2, 0]) == a and int(ids4[10, 10, 0]) not in (a, int(ids3[2, 2, 6]))


def test_describe_components_sorted():
    v = _world([(1, slice(0, 2), slice(0, 2), slice(0, 5)),   # 20 клеток
                (2, slice(8, 9), slice(8, 9), slice(0, 2))])  # 2 клетки
    lab, _ = label_components(v)
    coords, sp = pack_cells(v)
    ids = lab[coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)].astype(np.uint32)
    comps = describe_components(coords, sp, ids, {1: 0, 2: 5})
    assert [c.size for c in comps] == [20, 2]
    assert comps[0].species == 1 and comps[0].zmax == 4 and comps[1].born == 5


def test_make_snapshot_from_state():
    cfg = Config(n=24, seed_density=0.02)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    snap = make_snapshot(state, 0, cfg, Tracker())
    assert snap.n == 24 and len(snap.coords) == sum(snap.pops) > 0
    assert len(snap.labels) == len(snap.coords) and snap.labels.min() >= 1
    assert sum(c.size for c in snap.components) == len(snap.coords)
