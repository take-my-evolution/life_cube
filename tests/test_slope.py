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


def test_crown_shades_the_ground(run):
    """Крона гасит свет, и под деревом у земли его заметно меньше.

    Меряем именно СВЕТ, а не густоту травы: с тех пор как стволы стоят с
    просветом, лес стал редким, света под кроной хватает, и трава под деревьями
    растёт не хуже. Механизм тени от этого никуда не делся — он в поле света, и
    ресурс растения считается прямо из него.
    """
    from life_cube.engines.slope import light_field

    _, cfg, st, _ = run
    sp, stone, soil = arrays(st)
    n = sp.shape[0]
    surf = stone + soil
    G = np.asarray(cfg.genomes)
    idx = np.clip(sp.astype(int) - 1, 0, len(G) - 1)
    plants = (sp > 0) & (G[idx, IDX["speed"]] == 0)
    absorb = np.where(plants, G[idx, IDX["absorb"]], 0.0).astype(np.float32)
    xp, _, _ = get_backend(False)
    L = np.asarray(light_field(xp.asarray(plants), xp.asarray(absorb), xp))
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ground_light = L[xs, ys, np.clip(surf, 0, n - 1)]

    crown = (sp == TREE).any(axis=2)
    assert crown.sum() > 3, "деревьев в мире нет — тень мерить не на чем"
    shade = np.zeros_like(crown)
    for x, y in np.argwhere(crown):
        shade[max(0, x - 1):x + 2, max(0, y - 1):y + 2] = True
    ground = soil > 0
    under, open_ = ground_light[shade & ground], ground_light[(~shade) & ground]
    assert under.size and open_.size
    assert under.mean() < 0.8 * open_.mean(), (under.mean(), open_.mean())


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
    e = Engine(cfg=R.make_config(n=32, seed_world=5, reseed_every=1,
                                 reseed=True), rules="slope")
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


def test_a_hungry_animal_does_not_always_step_the_same_way():
    """Жалоба: «существа текут в одном направлении, как река».

    Шум в оценке направления был МНОЖИТЕЛЕМ поля добычи и потому исчезал ровно
    там, где поле пустое, — а это единственное место, где случайный ход и
    нужен. Все ничьи разрешались первым направлением списка, и зверь, не
    видящий добычи, каждый шаг шагал строго в +x.
    """
    R = get_rules("slope")
    cfg = R.make_config(n=24, seed_world=4, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0)
    xp, corr, _ = get_backend(False)
    seen = set()
    for trial in range(12):
        st, _ = R.init_state(cfg, xp)
        sp = np.zeros_like(np.asarray(st["species"]))
        en = np.zeros_like(np.asarray(st["energy"]))
        surf = np.asarray(st["stone_h"]) + np.asarray(st["soil_h"])
        x, y = 12, 12                       # один зверь, добычи в мире нет вовсе
        sp[x, y, int(surf[x, y])] = PRED
        en[x, y, int(surf[x, y])] = 50.0
        st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)
        st["rng"] = xp.random.default_rng(trial)
        cfg.seed_mut = 1000 + trial
        R.step(st, cfg, xp, corr, trial)
        where = np.argwhere(np.asarray(st["species"]) == PRED)
        if len(where):
            seen.add((int(where[0][0]) - x, int(where[0][1]) - y))
    assert len(seen) > 1, f"голодный зверь всегда идёт в одну сторону: {seen}"


def test_soil_does_not_jitter_on_flat_ground():
    """Жалоба: «земля бурлит». Единица почвы, уехавшая при перепаде 1, делает
    перепад −1 — и на следующем шаге едет обратно. Почва должна двигаться
    только там, где перенос выравнивает столбцы."""
    R, cfg, st, _ = world(n=48, gens=150, seed=20260902)
    xp, corr, _ = get_backend(False)
    stone = np.asarray(st["stone_h"])
    flat = stone <= stone.min() + 1              # ровное дно, склона нет вовсе
    assert flat.sum() > 100
    moved = 0
    net = np.zeros_like(np.asarray(st["soil_h"]))
    for g in range(1, 21):
        prev = np.asarray(st["soil_h"]).copy()
        R.step(st, cfg, xp, corr, 200 + g)
        d = np.asarray(st["soil_h"]) - prev
        moved += int((d != 0).sum())
        net += d
        assert not (d != 0)[flat].any(), "почва шевелится на ровном дне"
    if moved:
        # дрожь — это когда перемещений много, а чистого переноса нет
        assert np.abs(net).sum() / moved > 0.15, "почва ходит туда-сюда вместо сползания"


def test_crown_falls_when_the_trunk_is_gone():
    """Крона держится на стволе. Съели ствол снизу — крона обваливается тем же
    шагом, а не висит зелёными кубами в воздухе."""
    R = get_rules("slope")
    cfg = R.make_config(n=16, seed_world=2, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    sp = np.asarray(st["species"]).copy()
    en = np.asarray(st["energy"]).copy()
    surf = np.asarray(st["stone_h"]) + np.asarray(st["soil_h"])
    sp[:] = 0
    x, y = 8, 8
    z0 = int(surf[x, y])
    trunk = int(cfg.genomes[TREE - 1][IDX["trunk"]])
    for k in range(trunk + 2):                    # ствол и пара клеток кроны
        sp[x, y, z0 + k] = TREE
        en[x, y, z0 + k] = 20.0
    sp[x + 1, y, z0 + trunk + 1] = TREE           # ветка вбок
    en[x + 1, y, z0 + trunk + 1] = 20.0
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)
    R.step(st, cfg, xp, corr, 1)
    assert int((np.asarray(st["species"]) == TREE).sum()) > 3, "дерево не устояло целым"

    sp = np.asarray(st["species"]).copy()
    sp[x, y, z0] = 0                              # ствол подъели у земли
    st["species"] = xp.asarray(sp)
    R.step(st, cfg, xp, corr, 2)
    left = np.argwhere(np.asarray(st["species"]) == TREE)
    hanging = [c for c in left if c[2] > surf[c[0], c[1]]]
    assert not hanging, f"в воздухе осталось {len(hanging)} клеток кроны"


# --- дерево как организм ----------------------------------------------------

def test_tree_is_one_organism_with_a_shared_purse(run):
    """Ствол и крона — один организм: клетки помечены общим номером, а энергия
    лежит в корне, а не размазана по клеткам."""
    _, _, st, _ = run
    sp, stone, soil = arrays(st)
    tid = np.asarray(st["tree_id"])
    cells = np.argwhere(sp == TREE)
    assert len(cells) > 10
    assert all(tid[x, y, z] > 0 for x, y, z in cells), "у клетки дерева нет организма"
    ids = np.unique(tid[tid > 0])
    assert 0 < len(ids) < len(cells), "каждая клетка оказалась отдельным деревом"
    # у каждого организма ровно один корень на подложке, и энергия — там
    surf = stone + soil
    en = np.asarray(st["energy"])
    for i in ids[:8]:
        mine = np.argwhere(tid == i)
        roots = [c for c in mine if c[2] == surf[c[0], c[1]]]
        assert len(roots) == 1, f"у организма {i} корней {len(roots)}"
        x, y, z = roots[0]
        assert en[x, y, z] > 0, "кошелёк дерева пуст, а дерево живо"


def test_herbivore_eats_the_crown_from_the_top():
    """Травоядное объедает крону СВЕРХУ. Раньше оно ело нижнюю клетку столбца —
    основание ствола — и дерево валилось с одного укуса."""
    R = get_rules("slope")
    cfg = R.make_config(n=16, seed_world=6, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0, mutate_rate=0.0)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    sp = np.zeros_like(np.asarray(st["species"]))
    en = np.zeros_like(np.asarray(st["energy"]))
    surf = np.asarray(st["stone_h"]) + np.asarray(st["soil_h"])
    x, y = 8, 8
    z0 = int(surf[x, y])
    trunk = int(cfg.genomes[TREE - 1][IDX["trunk"]])
    top = z0 + trunk + 1
    for z in range(z0, top + 1):
        sp[x, y, z] = TREE
        en[x, y, z] = 30.0
    sp[x + 1, y, int(surf[x + 1, y])] = HERB       # травоядное рядом, на земле
    en[x + 1, y, int(surf[x + 1, y])] = 30.0
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)

    for g in range(1, 400):
        R.step(st, cfg, xp, corr, g)
        cur = np.asarray(st["species"])
        if not (cur[x, y] == TREE).any():
            raise AssertionError("дерево исчезло целиком, а не объедалось сверху")
        if int(np.nonzero(cur[x, y] == TREE)[0].max()) < top:
            break                                   # верхушку откусили
    else:
        raise AssertionError("травоядное так и не тронуло крону")
    cur = np.asarray(st["species"])
    assert cur[x, y, z0] == TREE, "ствол у земли съеден раньше кроны"


def test_trunks_keep_their_distance():
    """Стволы не встают вплотную: минимум `trunk_spacing` клеток между ними."""
    import itertools

    R, cfg, st, _ = world(n=48, gens=300, seed=11)
    sp, stone, soil = arrays(st)
    tid = np.asarray(st["tree_id"])
    n = sp.shape[0]
    surf = stone + soil
    roots = [(x, y) for x, y, z in np.argwhere(sp == TREE)
             if z == surf[x, y] and tid[x, y, z] == x * n + y + 1]
    assert len(roots) > 3, f"стволов всего {len(roots)}"
    d = [max(abs(a[0] - b[0]), abs(a[1] - b[1]))
         for a, b in itertools.combinations(roots, 2)]
    assert min(d) >= cfg.trunk_spacing, f"минимальный просвет {min(d)}"


def test_trees_spread_by_seeds_beyond_their_crown():
    """Со ссылкой на просвет между стволами: расти вбок дерево не может, значит
    новые деревья берутся только из семян, улетевших дальше кроны."""
    R, cfg, st, _ = world(n=48, gens=60, seed=11)
    tid = np.asarray(st["tree_id"])
    was = int(np.unique(tid[tid > 0]).size)
    xp, corr, _ = get_backend(False)
    for g in range(61, 400):
        R.step(st, cfg, xp, corr, g)
    tid = np.asarray(st["tree_id"])
    now = int(np.unique(tid[tid > 0]).size)
    assert now > was, f"деревьев было {was}, стало {now} — семена не всходят"


# --- мутация вместо подсева -------------------------------------------------

def test_mutation_returns_an_extinct_species():
    """Выпавшее звено возвращается изнутри живого, а не кубиками на карте:
    у соседнего яруса подскакивает шанс ошибиться при делении."""
    R = get_rules("slope")
    xp, corr, _ = get_backend(False)
    for gone in (TREE, HERB, PRED):
        cfg = R.make_config(n=40, seed_world=11, reseed=False)
        st, _ = R.init_state(cfg, xp)
        for g in range(1, 150):
            R.step(st, cfg, xp, corr, g)
        sp = np.asarray(st["species"]).copy()
        sp[sp == gone] = 0
        st["species"] = xp.asarray(sp)
        for g in range(150, 400):
            pop = R.step(st, cfg, xp, corr, g)
            if pop[gone - 1] > 0:
                break
        else:
            raise AssertionError(f"вид {gone} не вернулся за 250 поколений")


def test_mutation_is_rarer_when_the_niche_is_full():
    """Шанс мутации смотрит на тесноту: в пустую нишу мутируют куда охотнее,
    чем в занятую."""
    R = get_rules("slope")
    cfg = R.make_config(n=40)
    st = {"pops": [100, 800, 50, 200, 100], "niche": {1: 800, 2: 1000, 3: 100,
                                                      4: 1600, 5: 1600}}
    busy = R.mutation_chance(st, cfg, 4, 5)
    st_empty = dict(st, pops=[100, 800, 50, 200, 0])
    empty = R.mutation_chance(st_empty, cfg, 4, 5)
    assert empty > busy * 10, (empty, busy)
    # и на тесноту самого родителя
    thin = R.mutation_chance(dict(st, pops=[100, 20, 50, 200, 100]), cfg, 2, 4)
    thick = R.mutation_chance(st, cfg, 2, 4)
    assert thick > thin, (thick, thin)


def test_species_table_counts_tree_organisms_not_cells():
    """В таблице видов у дерева стоит число организмов, а не клеток."""
    from life_cube.engine import Engine

    R = get_rules("slope")
    e = Engine(cfg=R.make_config(n=40, seed_world=11), rules="slope", components=False)
    for _ in range(200):
        e.advance()
    snap = e.publish(force=True, components=False)
    assert snap.organisms is not None
    assert snap.organisms[TREE - 1] < snap.pops[TREE - 1], (snap.organisms, snap.pops)
    for s in (MOSS, GRASS, HERB, PRED):
        assert snap.organisms[s - 1] == snap.pops[s - 1]


def test_mutation_knobs_apply_without_recreating_the_world():
    """Обе ручки мутации живые: шанс читается на каждом делении, поэтому
    менять их можно на ходу, не стирая популяцию пересозданием мира."""
    from life_cube.engine import Engine

    R = get_rules("slope")
    e = Engine(cfg=R.make_config(n=32, seed_world=11), rules="slope", components=False)
    for _ in range(60):
        e.advance()
    st = {"pops": [100, 800, 50, 200, 0], "niche": {1: 800, 2: 1000, 3: 100,
                                                    4: 1000, 5: 1000}}
    was = R.mutation_chance(st, e.cfg, 4, 5)
    gen_before, k_before = e.gen, int((np.asarray(e.state["species"]) > 0).sum())
    e.set_world(mutate_rescue=100.0)
    now = R.mutation_chance(st, e.cfg, 4, 5)
    assert now > was * 5, (now, was)
    assert e.gen == gen_before and int((np.asarray(e.state["species"]) > 0).sum()) == k_before, \
        "мир пересоздался от поворота ручки"


def test_default_rescue_is_modest():
    """По умолчанию пустая ниша ускоряет возврат в 10 раз, не в 60: иначе
    вымираний не видно вовсе."""
    assert get_rules("slope").Config().mutate_rescue == 10.0
