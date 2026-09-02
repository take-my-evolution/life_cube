"""Движок «склон»: подложка решает, кто где растёт.

Проверяем ровно то, ради чего движок написан:
  * мох рождается только на голом камне и только на поверхности;
  * трава и деревья рождаются только на почве;
  * дерево строит ствол вверх, а ветвится выше ствола;
  * крона гасит свет, и под деревом травы меньше;
  * звери ходят по камню, почве и траве, но не по стволу и кроне;
  * почва съезжает вниз: низины зарастают, вершины остаются под мхом.
"""

from pathlib import Path

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


def surface_of(st):
    return np.asarray(st["stone_h"]) + np.asarray(st["soil_h"])


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
    surf = surface_of(st)
    off = sum(1 for x, y, z in cells if soil[x, y] != 0 or z != surf[x, y])
    # исключений быть не должно: съехавшая подложка убивает стелющееся растение
    assert off == 0, f"{off} из {len(cells)}"


def test_plants_only_on_soil(run):
    _, _, st, _ = run
    sp, stone, soil = arrays(st)
    surf = surface_of(st)
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
    surf = surface_of(st)
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
    surf = surface_of(st)
    G = np.asarray(cfg.genomes)
    idx = np.clip(sp.astype(int) - 1, 0, len(G) - 1)
    plants = (sp > 0) & (G[idx, IDX["speed"]] == 0)
    absorb = np.where(plants, G[idx, IDX["absorb"]], 0.0).astype(np.float32)
    xp, _, _ = get_backend(False)
    L = np.asarray(light_field(xp.asarray(plants), xp.asarray(absorb), xp))
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ground_light = L[xs, ys, np.clip(surf, 0, n - 1)]

    # меряем ПОД кроной, а не в кольце вокруг: крона компактная, и соседние
    # столбцы она не затеняет — от этого утверждение про кольцо разъезжается
    under_crown = (sp == TREE).any(axis=2)
    assert under_crown.sum() > 3, "деревьев в мире нет — тень мерить не на чем"
    ground = soil > 0
    under = ground_light[under_crown & ground]
    open_ = ground_light[(~under_crown) & ground]
    assert under.size and open_.size
    assert under.mean() < 0.5 * open_.mean(), (under.mean(), open_.mean())



def test_shade_thins_the_grass_under_the_trees():
    """Под кроной должны быть ПРОГАЛИНЫ, а не просто тёмный свет.

    Одного затенения мало: трава, которой хватает крох, под деревом просто
    растёт медленнее — и всё равно смыкается, да ещё и оказывается защищена от
    травоядных (в столбец с деревом им хода нет). Прогалина появляется, только
    когда трава ТРАТИТ больше, чем добирает в тени. Мир берём зрелый: на
    двухсотом поколении лес ещё молод, и разница видна сама собой.
    """
    R, cfg, st, _ = world(n=40, gens=400, seed=11)
    sp, stone, soil = arrays(st)
    crown = (sp == TREE).any(axis=2)
    assert crown.sum() > 20, "лес не вырос — тень мерить не на чем"
    grass = (sp == GRASS).any(axis=2)
    ground = soil > 0
    under, open_ = grass[crown & ground], grass[(~crown) & ground]
    assert under.size > 10 and open_.size > 10
    # замер: до починки трава под кроной была ГУЩЕ открытой (отношение 1.1–1.5
    # — крона защищала её от травоядных), после стало 0.68–0.79
    assert under.mean() < 0.9 * open_.mean(), (under.mean(), open_.mean())


def test_animals_stay_on_the_ground(run):
    _, _, st, _ = run
    sp, stone, soil = arrays(st)
    surf = surface_of(st)
    seen = 0
    for s in (HERB, PRED):
        cells = np.argwhere(sp == s)
        if not len(cells):
            continue          # баланс на краю: вид мог как раз проваливаться
        seen += 1
        # шаг делается по поверхности; допустима одна клетка на плоском растении
        high = sum(1 for x, y, z in cells if z > surf[x, y] + 1)
        assert high == 0, f"вид {s}: {high} зверей выше подложки"
    assert seen, "в мире не осталось ни одного зверя — проверять нечего"


def test_soil_slides_into_hollows():
    R, cfg, st, _ = world(gens=250)
    stone0 = np.asarray(R.init_state(cfg, get_backend(False)[0])[0]["stone_h"])
    soil = np.asarray(st["soil_h"])
    order = stone0.ravel().argsort()
    q = len(order) // 4
    low = np.zeros(stone0.size, bool); low[order[:q]] = True
    high = np.zeros(stone0.size, bool); high[order[-q:]] = True
    low = low.reshape(stone0.shape); high = high.reshape(stone0.shape)
    # порог относительный: слой почвы на старте тонкий (гора не должна тонуть в
    # равнине), и абсолютная разница «на клетку» тут больше ничего не значит
    assert soil[low].mean() > 3 * soil[high].mean(), (soil[low].mean(), soil[high].mean())
    assert soil[low].mean() > 0.5
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
    moved = 0
    net = np.zeros_like(np.asarray(st["soil_h"]))
    for g in range(1, 21):
        prev = np.asarray(st["soil_h"]).copy()
        stone_prev = np.asarray(st["stone_h"]).copy()
        surf = stone_prev + prev
        # ровное место считаем по ТЕКУЩЕЙ поверхности: колода от погибшего
        # дерева — настоящий бугор, и сползание с него законно
        level = np.ones_like(surf, dtype=bool)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            level &= np.roll(surf, (dx, dy), (0, 1)) == surf
        level[0], level[-1], level[:, 0], level[:, -1] = False, False, False, False
        assert level.sum() > 100
        R.step(st, cfg, xp, corr, 200 + g)
        d = np.asarray(st["soil_h"]) - prev
        # колода от погибшего дерева накрывает почву камнем — это не дрожь,
        # а событие; такие столбцы из проверки исключаем
        logged = np.asarray(st["stone_h"]) != stone_prev
        moved += int((d != 0).sum())
        net += d
        assert not (d != 0)[level & ~logged].any(), "почва шевелится на ровном месте"
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
    surf = surface_of(st)
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
    surf = surface_of(st)
    roots = [(x, y) for x, y, z in np.argwhere(sp == TREE)
             if z == surf[x, y] and tid[x, y, z] == x * n + y + 1]
    assert len(roots) > 3, f"стволов всего {len(roots)}"
    d = [max(abs(a[0] - b[0]), abs(a[1] - b[1]))
         for a, b in itertools.combinations(roots, 2)]
    assert min(d) >= cfg.trunk_spacing, f"минимальный просвет {min(d)}"


def test_trees_spread_by_seeds_beyond_their_crown():
    """Со ссылкой на просвет между стволами: расти вбок дерево не может, значит
    новые деревья берутся только из семян, улетевших дальше кроны.

    Мир строим нарочно: одно взрослое дерево посреди чистой почвы. В зрелом
    лесу свободных мест почти нет, и «стало больше деревьев» там уже ничего не
    доказывает — численность держится смертями, а не всходами.
    """
    R = get_rules("slope")
    cfg = R.make_config(n=32, seed_world=5, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0, mutate_rate=0.0, soil_slide=0.0,
                        soil_start=1.0, seed_fall=0.05)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    soil = np.asarray(st["soil_h"])
    stone = np.asarray(st["stone_h"])
    surf = stone + soil
    sp = np.zeros_like(np.asarray(st["species"]))
    en = np.zeros_like(np.asarray(st["energy"]))
    ground = np.argwhere(soil > 0)
    assert len(ground) > 50
    x, y = map(int, ground[len(ground) // 2])
    trunk = int(cfg.genomes[TREE - 1][IDX["trunk"]])
    for k in range(trunk + 2):                   # одно взрослое дерево
        sp[x, y, int(surf[x, y]) + k] = TREE
    en[x, y, int(surf[x, y])] = 500.0            # кошелёк полон: пора сеять
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)

    for g in range(1, 300):
        R.step(st, cfg, xp, corr, g)
        cur = np.asarray(st["species"])
        roots = [(int(a), int(b)) for a, b, c in np.argwhere(cur == TREE)
                 if c == surf[a, b]]
        far = [(a, b) for a, b in roots if max(abs(a - x), abs(b - y)) >= cfg.trunk_spacing]
        if far:
            break
    else:
        raise AssertionError("ни одно семя не взошло за 300 поколений")
    assert all(max(abs(a - x), abs(b - y)) >= cfg.trunk_spacing for a, b in far)


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


# --- рацион, колода, рельеф, случайные гены ---------------------------------

def test_herbivore_does_not_eat_moss():
    """Мох — не корм, а порода: он точит камень и готовит почву. Раньше добычей
    считалось всё, что ярусом ниже, и травоядные паслись на камнях."""
    R = get_rules("slope")
    cfg = R.make_config(n=16, seed_world=7, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0, mutate_rate=0.0, soil_slide=0.0,
                        erode_rate=0.0, p_shock=0.0)
    # снимаем со мха броню: защищать его должен РАЦИОН, а не удачный бросок
    g = np.asarray(cfg.genomes).copy()
    g[MOSS - 1, IDX["armor"]] = 0.0
    g[HERB - 1, IDX["hunt"]] = 1.0
    cfg.genomes = g
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    sp = np.zeros_like(np.asarray(st["species"]))
    en = np.zeros_like(np.asarray(st["energy"]))
    stone, soil = np.asarray(st["stone_h"]), np.asarray(st["soil_h"])
    surf = stone + soil
    bare = np.argwhere(soil == 0)
    assert len(bare) > 4
    for x, y in bare[:6]:                      # полянка мха на голом камне
        sp[x, y, int(surf[x, y])] = MOSS
        en[x, y, int(surf[x, y])] = 8.0
    x, y = bare[0]
    zw = int(surf[x, y]) + 1
    sp[x, y, zw] = HERB                        # травоядное стоит прямо на мху
    en[x, y, zw] = 40.0
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)
    before = int((np.asarray(st["species"]) == MOSS).sum())
    for g in range(1, 120):
        R.step(st, cfg, xp, corr, g)
    after = int((np.asarray(st["species"]) == MOSS).sum())
    assert after >= before, f"мха было {before}, стало {after} — его съели"


def test_a_dying_tree_falls_and_kills_the_soil_under_it():
    """Ствол ПАДАЕТ: валится в случайную сторону и по всей длине подменяет
    почву камнем. Высота поверхности при этом не меняется вовсе — значит,
    ничего не повисает в воздухе и ничего не хоронится."""
    R = get_rules("slope")
    cfg = R.make_config(n=20, seed_world=3, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0, mutate_rate=0.0, soil_slide=0.0)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    stone0, soil0 = np.asarray(st["stone_h"]).copy(), np.asarray(st["soil_h"]).copy()
    surf0 = stone0 + soil0
    x, y = map(int, np.argwhere(soil0 > 0)[0])
    sp = np.zeros_like(np.asarray(st["species"]))
    en = np.zeros_like(np.asarray(st["energy"]))
    z0 = int(surf0[x, y])
    trunk = int(cfg.genomes[TREE - 1][IDX["trunk"]])
    for k in range(trunk + 3):
        sp[x, y, z0 + k] = TREE
        en[x, y, z0 + k] = 30.0
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)
    R.step(st, cfg, xp, corr, 1)
    height = int((np.asarray(st["species"])[x, y] == TREE).sum())
    en = np.asarray(st["energy"]).copy()
    en[x, y, z0] = -1000.0                       # дерево гибнет от голода
    st["energy"] = xp.asarray(en)
    R.step(st, cfg, xp, corr, 2)

    assert int((np.asarray(st["species"]) == TREE).sum()) == 0
    stone, soil = np.asarray(st["stone_h"]), np.asarray(st["soil_h"])
    killed = np.argwhere((soil == 0) & (soil0 > 0))
    assert len(killed) > 1, "ствол не упал: мертва только одна клетка"
    assert len(killed) <= height, (len(killed), height)
    assert (len(set(map(int, killed[:, 0]))) == 1) or (len(set(map(int, killed[:, 1]))) == 1), \
        "полоса не прямая — ствол не лёг, а рассыпался"
    assert any(int(a) == x and int(b) == y for a, b in killed)
    # высота поверхности не изменилась НИГДЕ: висеть и хорониться нечему
    assert np.array_equal(stone + soil, surf0)


def test_nothing_grows_where_a_trunk_fell():
    """Под упавшим стволом земля мертва: расти там нельзя никому, кроме мха."""
    R = get_rules("slope")
    cfg = R.make_config(n=16, seed_world=3, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0, mutate_rate=0.0, soil_slide=0.0,
                        erode_rate=0.0)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    stone = np.asarray(st["stone_h"]).copy()
    soil = np.asarray(st["soil_h"]).copy()
    strip = [tuple(map(int, c)) for c in np.argwhere(soil > 0)[:6]]
    for x, y in strip:                           # полоса упавшего ствола
        stone[x, y] += soil[x, y]
        soil[x, y] = 0
    st["stone_h"], st["soil_h"] = xp.asarray(stone), xp.asarray(soil)
    sp = np.zeros_like(np.asarray(st["species"]))
    en = np.zeros_like(np.asarray(st["energy"]))
    surf = stone + soil
    for x, y in [tuple(map(int, c)) for c in np.argwhere(soil > 0)[:20]]:
        sp[x, y, int(surf[x, y])] = GRASS        # трава кругом полосы
        en[x, y, int(surf[x, y])] = 20.0
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)
    for g in range(1, 200):
        R.step(st, cfg, xp, corr, g)
    cur = np.asarray(st["species"])
    grown = [(x, y) for x, y in strip if (cur[x, y] == GRASS).any()]
    assert not grown, f"на мёртвой полосе выросла трава: {grown[:4]}"


def test_moss_takes_over_the_dead_strip_and_makes_soil_again():
    """Круг замыкается: дерево упало → камень → мох → снова почва."""
    R = get_rules("slope")
    cfg = R.make_config(n=16, seed_world=3, seed_animals=0.0, seed_tree=0.0,
                        seed_density=0.0, mutate_rate=0.0, soil_slide=0.0,
                        erode_rate=0.2, rain_rate=0.5, rain_amount=2.0)
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    stone = np.asarray(st["stone_h"]).copy()
    soil = np.asarray(st["soil_h"]).copy()
    x, y = map(int, np.argwhere(soil > 0)[0])
    stone[x, y] += soil[x, y]
    soil[x, y] = 0
    st["stone_h"], st["soil_h"] = xp.asarray(stone), xp.asarray(soil)
    sp = np.zeros_like(np.asarray(st["species"]))
    en = np.zeros_like(np.asarray(st["energy"]))
    sp[x, y, int(stone[x, y])] = MOSS            # мох сел на мёртвую полосу
    en[x, y, int(stone[x, y])] = 10.0
    st["species"], st["energy"] = xp.asarray(sp), xp.asarray(en)
    for g in range(1, 400):
        R.step(st, cfg, xp, corr, g)
        if int(np.asarray(st["soil_h"])[x, y]) > 0:
            break
    else:
        raise AssertionError("мох не вернул полосу в почву за 400 поколений")


def test_the_hill_is_not_buried_in_the_plain():
    """Почва на старте лежит ТОНКИМ слоем: при трёх клетках подножие горы
    тонуло в равнине и гора выглядела наполовину закопанной."""
    R = get_rules("slope")
    cfg = R.make_config(n=64, seed_world=20260825)
    xp, _, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    stone, soil = np.asarray(st["stone_h"]), np.asarray(st["soil_h"])
    plain = stone <= stone.min() + 1
    sunk = int(np.median((stone + soil)[plain])) - int(stone.min())
    assert sunk <= 1, f"подножие утоплено на {sunk} клеток"


def test_role_genes_are_off_limits_to_the_random_button():
    """Роль, подложка и форма роста случайными не бывают: мох со стволом 9 —
    это не «другой мох», это сломанный мир. Список движок отдаёт и клиенту."""
    R = get_rules("slope")
    cfg = R.make_config(n=32)
    fixed = R.fixed_genes(cfg)
    for name in ("trophic", "speed", "substrate", "trunk"):
        assert name in fixed, name
    assert R.to_json(cfg)["fixed_genes"] == list(fixed)
    before = np.asarray(cfg.genomes).copy()
    after = R.randomize(cfg, np.random.default_rng(1))
    for name in fixed:
        assert np.array_equal(before[:, IDX[name]], after[:, IDX[name]]), name
    assert not np.array_equal(before[:, IDX["light"]], after[:, IDX["light"]])


def test_client_random_button_respects_the_fixed_list():
    """Кнопка в панели должна брать список у движка, а не знать свой."""
    html = (Path(__file__).resolve().parents[1] / "life_cube" / "viewers" / "web"
            / "static" / "index.html").read_text()
    i = html.index("btnRandom')")
    chunk = html[i:i + 1200]
    assert "fixed_genes" in chunk, "кнопка не спрашивает движок"


def test_the_chain_lives_on_the_edge():
    """Баланс намеренно поставлен на край: хищник достаточно удачлив, чтобы
    выесть травоядных и издохнуть следом, а мутация возвращает выпавшее звено —
    к этому времени корма снова хватает, чтобы вид разошёлся.

    Проверяем ровно это: цепь ДОЛЖНА срываться (иначе мир скучный и вечный) и
    ДОЛЖНА возвращаться (иначе он мёртвый).
    """
    R = get_rules("slope")
    xp, corr, _ = get_backend(False)
    cfg = R.make_config(n=48, seed_world=7)
    st, _ = R.init_state(cfg, xp)
    crashes = np.zeros(5, int)
    was_alive = np.ones(5, bool)
    for g in range(1, 1500):
        pop = np.array(R.step(st, cfg, xp, corr, g))
        if g < 200:
            continue
        gone = pop == 0
        crashes += (gone & was_alive).astype(int)
        was_alive = ~gone
    assert crashes[HERB - 1] + crashes[PRED - 1] >= 2, \
        f"цепь ни разу не сорвалась: {crashes}"
    assert crashes[MOSS - 1] == 0 and crashes[GRASS - 1] == 0, \
        f"основание пирамиды не должно падать: {crashes}"
    # и в конце мир жив: провалы затягиваются мутацией, а не остаются навсегда
    assert sum(int(v) for v in pop) > 500


def test_the_engine_has_no_genome_drift():
    """В этом движке гены не плывут: мутация переводит клетку в ДРУГОЙ ВИД, а
    геном вида остаётся тем, что задан в конструкторе. Иначе за тысячи
    поколений таблица генов уезжает сама собой, и мир уже не тот, что настроил
    пользователь."""
    R = get_rules("slope")
    cfg = R.make_config(n=32, seed_world=5)
    before = np.asarray(cfg.genomes).copy()
    xp, corr, _ = get_backend(False)
    st, _ = R.init_state(cfg, xp)
    for g in range(1, 300):
        R.step(st, cfg, xp, corr, g)
    assert np.array_equal(np.asarray(cfg.genomes), before), "геном уехал сам"
    assert np.array_equal(np.asarray(st["genomes"]), before)
