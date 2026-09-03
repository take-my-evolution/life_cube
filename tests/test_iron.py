"""Движок «железо»: то, что обязано быть верно, чтобы этот же мир считала ПЛИС.

  * состояние и правила — только целые типы;
  * тот же сид → тот же мир бит-в-бит (и на GPU, если он есть);
  * хеш координат совпадает с эталоном на чистом Python (спецификация для HDL);
  * все три варианта живут: жизнь 3D не вымирает и не кипит, волны бегут,
    в экологии через сотни поколений живы все пять ярусов;
  * звери не проходят сквозь стволы и не висят в воздухе.
"""

import numpy as np
import pytest

from life_cube.backend import get_backend
from life_cube.engines import get_rules
from life_cube.engines.iron import (GRASS, HERB, MOSS, NAMES, PRED, TREE, hash32,
                                    hash32_py)


def world(variant, n=32, gens=200, seed=20260903, xp_gpu=False, **kw):
    R = get_rules("iron")
    cfg = R.make_config(n=n, variant=variant, seed_world=seed, seed_mut=seed, **kw)
    xp, corr, _ = get_backend(xp_gpu)
    st, _ = R.init_state(cfg, xp)
    pops = []
    for g in range(1, gens + 1):
        pops.append(R.step(st, cfg, xp, corr, g))
    return R, cfg, st, pops


def test_registered():
    from life_cube.engines import list_engines
    assert "iron" in {e["name"] for e in list_engines()}


def test_hash_matches_pure_python_reference():
    xp = np
    xs = np.array([0, 1, 2, 31, 63, 200], dtype=np.uint32)
    for gen, salt in ((0, 0), (1, 7), (12345, 0xBEEF), (2 ** 31 + 5, 0xFFFFFFFF)):
        got = hash32(xp, xs, xs * 3, xs * 5, gen, salt)
        want = [hash32_py(int(x), int(x) * 3, int(x) * 5, gen, salt) for x in xs]
        assert [int(v) for v in got] == want, (gen, salt)
    # разные соли — разные потоки, а не сдвиг одного и того же
    a = hash32(xp, xs, 0, 0, 5, 1)
    b = hash32(xp, xs, 0, 0, 5, 2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_state_is_integer_only(variant):
    _, _, st, _ = world(variant, gens=5)
    for k, v in st.items():
        if hasattr(v, "dtype"):
            assert v.dtype.kind in "uib", f"{k} имеет тип {v.dtype} — в железе такого нет"


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_same_seed_same_world_bit_for_bit(variant):
    _, _, a, pa = world(variant, gens=60)
    _, _, b, pb = world(variant, gens=60)
    assert pa == pb
    for k in ("species", "energy", "age", "stone_h", "soil_h", "wet", "anim", "aen"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_different_seed_different_world():
    _, _, a, _ = world(1, gens=30, seed=1)
    _, _, b, _ = world(1, gens=30, seed=2)
    assert not np.array_equal(np.asarray(a["species"]), np.asarray(b["species"]))


def test_gpu_matches_cpu_if_available():
    try:
        import cupy  # noqa: F401
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("нет GPU")
    _, _, a, pa = world(1, gens=40)
    _, _, b, pb = world(1, gens=40, xp_gpu=True)
    assert pa == pb
    assert np.array_equal(np.asarray(a["species"]), b["species"].get())


def test_life3d_neither_dies_nor_boils():
    _, cfg, st, pops = world(0, gens=200)
    alive = sum(pops[-1])
    assert alive > 100, pops[-1]
    # активность: меняется заметная часть, но не все клетки разом
    R = get_rules("iron")
    xp, corr, _ = get_backend(False)
    prev = np.asarray(st["species"]).copy()
    R.step(st, cfg, xp, corr, 201)
    changed = int((np.asarray(st["species"]) != prev).sum())
    assert 20 < changed < alive * 3, (changed, alive)


def test_waves_settle_into_running_fronts():
    _, cfg, st, pops = world(2, gens=200)
    R = get_rules("iron")
    xp, corr, _ = get_backend(False)
    prev = np.asarray(st["species"]).copy()
    R.step(st, cfg, xp, corr, 201)
    changed = int((np.asarray(st["species"]) != prev).sum())
    total = sum(pops[-1])
    # фронты волн: перекрашивается доля клеток, а не всё и не ничего
    assert 0.02 * total < changed < 0.5 * total, (changed, total)
    assert all(p > 0 for p in pops[-1]), pops[-1]


def test_ecology_all_five_tiers_alive():
    _, _, _, pops = world(1, gens=400)
    assert len(pops[-1]) == len(NAMES) == 5
    assert all(p > 0 for p in pops[-1]), dict(zip(NAMES, pops[-1]))
    # хищник — вершина пирамиды: его должно быть меньше травоядных
    assert pops[-1][PRED - 1] < pops[-1][HERB - 1] * 3


def test_ecology_substrate_rules():
    _, cfg, st, _ = world(1, gens=150)
    sp = np.asarray(st["species"]); stone = np.asarray(st["stone_h"]); soil = np.asarray(st["soil_h"])
    surf = stone + soil
    n = cfg.n
    zz = np.arange(n)[None, None, :]
    hgt = zz - surf[:, :, None]
    moss = sp == MOSS
    # мох — только на поверхности и только на голом камне
    assert not (moss & (hgt != 0)).any()
    assert not (moss & (soil > 0)[:, :, None]).any()
    # трава — только на поверхности
    assert not ((sp == GRASS) & (hgt != 0)).any()
    # ниже поверхности жизни нет
    assert not ((sp > 0) & (hgt < 0)).any()


def test_animals_walk_on_surface_not_through_trunks():
    _, cfg, st, _ = world(1, gens=150)
    sp = np.asarray(st["species"]); surf = np.asarray(st["stone_h"] + st["soil_h"])
    anim = np.asarray(st["anim"])
    n = cfg.n
    xs, ys = np.nonzero(anim > 0)
    assert len(xs) > 0
    for x, y in zip(xs, ys):
        col = sp[x, y]
        assert col[surf[x, y]] != TREE, "зверь стоит в столбце ствола"
        here, above = col[surf[x, y]], col[min(surf[x, y] + 1, n - 1)]
        assert here in (HERB, PRED) or above in (HERB, PRED), "маркер зверя не на своём месте"


def test_events_flow_every_generation():
    """Для звука важно не население, а поток изменений: рождения и гибели
    должны идти каждое поколение, а не раз в сто."""
    R, cfg, st, _ = world(1, gens=100)
    xp, corr, _ = get_backend(False)
    births = deaths = 0
    for g in range(101, 121):
        prev = np.asarray(st["species"]).copy()
        R.step(st, cfg, xp, corr, g)
        cur = np.asarray(st["species"])
        births += int(((prev == 0) & (cur > 0)).sum())
        deaths += int(((prev > 0) & (cur == 0)).sum())
    assert births > 20 and deaths > 20, (births, deaths)
