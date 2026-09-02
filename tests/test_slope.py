"""Движок «склон»: подложка решает, кто где растёт.

Проверяем ровно то, ради чего движок написан:
  * мох рождается только на голом камне и только на поверхности;
  * трава и деревья рождаются только на почве;
  * дерево строит ствол вверх, а ветвится выше ствола;
  * крона гасит свет, и под деревом травы меньше;
  * звери ходят по камню, почве и траве, но не по стволу и кроне;
  * почва съезжает вниз: низины зарастают, вершины остаются под мхом.
"""

import numpy as np
import pytest

from life_cube.backend import get_backend
from life_cube.engines import get_rules
from life_cube.engines.slope import IDX, NAMES

MOSS, GRASS, TREE, HERB, PRED = 1, 2, 3, 4, 5


def world(n=40, gens=200, seed=11, **kw):
    R = get_rules("slope")
    cfg = R.make_config(n=n, seed_world=seed, **kw)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    pop = None
    for g in range(1, gens + 1):
        pop = R.step(st, cfg, xp, corr, g)
    return R, cfg, st, pop


@pytest.fixture(scope="module")
def run():
    return world()


def arrays(st):
    return (np.asarray(st["species"]), np.asarray(st["stone_h"]),
            np.asarray(st["soil_h"]))


def test_registered():
    from life_cube.engines import list_engines
    assert "slope" in {e["name"] for e in list_engines()}


def test_five_species_alive(run):
    _, _, _, pop = run
    assert len(pop) == len(NAMES) == 5
    assert all(p > 0 for p in pop), dict(zip(NAMES, pop))


def test_moss_only_on_bare_stone_surface(run):
    _, _, st, _ = run
    sp, stone, soil = arrays(st)
    cells = np.argwhere(sp == MOSS)
    assert len(cells) > 20
    surf = stone + soil
    off = sum(1 for x, y, z in cells if soil[x, y] != 0 or z != surf[x, y])
    # исключений быть не должно: съехавшая подложка убивает стелющееся растение
    assert off == 0, f"{off} из {len(cells)}"


def test_plants_only_on_soil(run):
    _, _, st, _ = run
    sp, stone, soil = arrays(st)
    surf = stone + soil
    for s in (GRASS, TREE):
        cells = np.argwhere(sp == s)
        assert len(cells) > 20
        # корень — клетка, стоящая на подложке; крона может нависать над камнем
        root = [(x, y, z) for x, y, z in cells if z == surf[x, y]]
        assert root
        off = sum(1 for x, y, _ in root if soil[x, y] == 0)
        # растение на голом камне не рождается; уцелеть может только то, из-под
        # чего почва уехала только что — оно уже не кормится и доживает
        assert off <= 0.05 * len(root), f"вид {s}: {off} из {len(root)}"


def test_trees_build_trunks(run):
    _, cfg, st, _ = run
    sp, stone, soil = arrays(st)
    surf = stone + soil
    trunk = int(cfg.genomes[TREE - 1][IDX["trunk"]])
    heights = []
    for x, y in np.argwhere((sp == TREE).any(axis=2)):
        col = np.nonzero(sp[x, y] == TREE)[0]
        heights.append(int(col.max() - surf[x, y] + 1))
    assert heights and max(heights) > trunk, heights[:10]


def test_crown_shades_grass(run):
    _, _, st, _ = run
    sp, stone, soil = arrays(st)
    n = sp.shape[0]
    tree2d = (sp == TREE).any(axis=2)
    shade = np.zeros_like(tree2d)
    for x, y in np.argwhere(tree2d):
        shade[max(0, x - 1):x + 2, max(0, y - 1):y + 2] = True
    grass2d = (sp == GRASS).any(axis=2)
    ground = soil > 0
    lit, dark = grass2d[ground & ~shade], grass2d[ground & shade]
    assert lit.size and dark.size
    assert dark.mean() < lit.mean(), (dark.mean(), lit.mean())


def test_animals_stay_on_the_ground(run):
    _, _, st, _ = run
    sp, stone, soil = arrays(st)
    surf = stone + soil
    for s in (HERB, PRED):
        cells = np.argwhere(sp == s)
        assert len(cells) > 5
        # шаг делается по поверхности; допустима одна клетка на плоском растении
        high = sum(1 for x, y, z in cells if z > surf[x, y] + 1)
        assert high == 0, f"вид {s}: {high} зверей выше подложки"


def test_soil_slides_into_hollows():
    R, cfg, st, _ = world(gens=250)
    stone0 = np.asarray(R.init_state(cfg, get_backend(False)[0])[0]["stone_h"])
    soil = np.asarray(st["soil_h"])
    order = stone0.ravel().argsort()
    q = len(order) // 4
    low = np.zeros(stone0.size, bool); low[order[:q]] = True
    high = np.zeros(stone0.size, bool); high[order[-q:]] = True
    low = low.reshape(stone0.shape); high = high.reshape(stone0.shape)
    assert soil[low].mean() > soil[high].mean() + 1.0, (soil[low].mean(), soil[high].mean())
    moss2d = (np.asarray(st["species"]) == MOSS).any(axis=2)
    assert moss2d[high].mean() > moss2d[low].mean()


def test_moss_makes_soil_from_stone():
    R = get_rules("slope")
    cfg = R.make_config(n=32, seed_world=3, soil_start=0.0, seed_density=0.5,
                        erode_rate=0.2, seed_tree=0.0, seed_animals=0.0)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    assert int(np.asarray(st["soil_h"]).sum()) == 0
    for g in range(1, 61):
        R.step(st, cfg, xp, corr, g)
    assert int(np.asarray(st["soil_h"]).sum()) > 0


def test_reseed_returns_a_single_extinct_species():
    """Выпавшее звено цепи возвращается: без этого мир после гибели хищника
    (или травоядного) навсегда остаётся лесом."""
    from life_cube.engine import Engine
    R = get_rules("slope")
    e = Engine(cfg=R.make_config(n=32, seed_world=5, reseed_every=1), rules="slope")
    sp = np.asarray(e.state["species"])
    sp[sp == PRED] = 0                       # хищник вымер, остальные живы
    e.state["species"] = e.xp.asarray(sp)
    pops = e.advance()
    assert pops[PRED - 1] == 0 and sum(pops) > 0
    e.maybe_reseed(pops)
    back = int((np.asarray(e.state["species"]) == PRED).sum())
    assert back > 0, "хищник не вернулся"


def test_predator_eats_adjacent_prey():
    """Охота ищет добычу по заранее посчитанной карте столбцов, а не сканом
    столбца на каждое направление каждого зверя (это стоило 90 % шага). Карта
    обязана видеть ту же добычу, что видел скан."""
    R = get_rules("slope")
    cfg = R.make_config(n=16, seed_world=1, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    sp = np.asarray(st["species"]).copy()
    en = np.asarray(st["energy"]).copy()
    surf = np.asarray(st["stone_h"]) + np.asarray(st["soil_h"])
    sp[:] = 0
    x, y = 8, 8
    z, zp = int(surf[x, y]), int(surf[x + 1, y])
    sp[x, y, z] = PRED                     # хищник и травоядное бок о бок
    en[x, y, z] = 5.0
    sp[x + 1, y, zp] = HERB
    en[x + 1, y, zp] = 5.0
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)
    for g in range(1, 60):
        R.step(st, cfg, xp, corr, g)
        if not (np.asarray(st["species"]) == HERB).any():
            break
    else:
        raise AssertionError("хищник за 60 поколений не съел соседа")
