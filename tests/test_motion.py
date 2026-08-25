"""Подвижные существа: ходят к еде, едят только свой трофический уровень − 1,
тратят энергию, делятся, стареют, падают под тяжестью."""

import numpy as np
import pytest

from life_cube import Config, run
from life_cube.backend import get_backend
from life_cube.config import GENOME_FIELDS, IDX
from life_cube.motion import shift
from life_cube.sim import init_state
from life_cube.step import step


def genome(**kw):
    g = np.zeros(len(GENOME_FIELDS), np.float32)
    for k, v in kw.items():
        g[IDX[k]] = v
    return g


GRASS = genome(absorb=0.3, up=1.0, birth=1.6, need=0.3, water=0.4, branch=0.2)
HERB = genome(hunt=0.9, trophic=1, speed=1, sense=4, metabolism=0.05, repro=4.0,
              lifespan=200)
PRED = genome(hunt=0.9, trophic=2, speed=2, sense=6, metabolism=0.05, repro=5.0,
              lifespan=300)


def world(genomes, n=16, **kw):
    cfg = Config(n=n, genomes=np.array(genomes, np.float32), p_shock=0.0,
                 p_dissolve=0.0, p_mutate=0.0, seed_density=0.05,
                 animal_share=0.0, **kw)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    stone = np.zeros((n, n, n), bool); stone[:, :, :3] = True
    state["stone"] = stone
    state["soil"] = np.zeros_like(stone)
    state["wet"] = np.ones((n, n), np.float32)
    state["species"] = np.zeros((n, n, n), np.int8)
    state["energy"] = np.zeros((n, n, n), np.float32)
    state["age"] = np.zeros((n, n, n), np.int32)
    return state, cfg, xp, corr


def put(state, cfg, x, y, z, s, energy=5.0):
    state["species"][x, y, z] = s
    state["energy"][x, y, z] = energy


def test_shift_helper():
    a = np.zeros((5, 5, 5)); a[2, 2, 2] = 1
    assert shift(a, (1, 0, 0), np)[3, 2, 2] == 1
    assert shift(a, (0, 0, -1), np)[2, 2, 1] == 1
    assert shift(a, (1, 0, 0), np).sum() == 1


def test_animal_walks_toward_food():
    """Существо идёт к еде, а не стоит на месте и не бродит наугад."""
    state, cfg, xp, corr = world([GRASS, HERB], n=20)
    for x in range(12, 16):                       # заросли травы справа
        for y in range(8, 12):
            put(state, cfg, x, y, 3, 1, 0.9)
    put(state, cfg, 8, 10, 3, 2, 3.0)             # энергии меньше порога деления
    xs = []
    for _ in range(4):
        step(state, cfg, xp, corr)
        pos = np.argwhere(state["species"] == 2)
        assert len(pos) >= 1
        xs.append(int(pos[:, 0].max()))           # передовое существо
    assert xs[-1] > xs[0] + 1, xs                 # заметно сдвинулось к еде


def test_animal_eats_prey_and_gains_energy():
    # трава, которая не размножается (порог рождения недостижим): так видно
    # именно поедание, а не гонку роста
    still = GRASS.copy(); still[IDX["birth"]] = 99.0
    state, cfg, xp, corr = world([still, HERB], n=12)
    put(state, cfg, 5, 5, 3, 1, 0.9)              # трава
    put(state, cfg, 6, 5, 3, 2, 2.0)              # травоядное рядом
    e0 = float(state["energy"][6, 5, 3])
    for _ in range(20):
        step(state, cfg, xp, corr)
        if not (state["species"] == 1).any():
            break
    assert not (state["species"] == 1).any()      # трава съедена
    herb = state["energy"][state["species"] == 2]
    # съеденная трава добавила энергии (0.9 × eat_efficiency), а обмен веществ
    # успел съесть немного
    assert len(herb) and herb.max() > e0


def test_trophic_levels_carnivore_ignores_plants():
    """Хищник не ест траву: рядом с одной травой он голодает и умирает."""
    state, cfg, xp, corr = world([GRASS, PRED], n=12)
    for x in range(4, 8):
        put(state, cfg, x, 5, 3, 1, 0.9)
    put(state, cfg, 6, 6, 3, 2, 1.0)              # хищник в траве
    grass0 = int((state["species"] == 1).sum())
    for _ in range(30):
        step(state, cfg, xp, corr)
    assert int((state["species"] == 2).sum()) == 0        # умер с голоду
    assert int((state["species"] == 1).sum()) >= grass0   # трава цела


def test_full_chain_predator_eats_herbivore():
    state, cfg, xp, corr = world([GRASS, HERB, PRED], n=14)
    put(state, cfg, 7, 7, 3, 2, 5.0)              # травоядное
    put(state, cfg, 8, 7, 3, 3, 5.0)              # хищник рядом
    eaten = False
    for _ in range(15):
        step(state, cfg, xp, corr)
        if not (state["species"] == 2).any():
            eaten = True
            break
    assert eaten
    assert (state["species"] == 3).any()          # хищник жив


def test_metabolism_and_lifespan():
    hungry = genome(hunt=0.5, trophic=1, speed=1, sense=2, metabolism=0.5, repro=99)
    state, cfg, xp, corr = world([GRASS, hungry], n=10)
    put(state, cfg, 5, 5, 3, 2, 1.0)              # энергии на 2 поколения
    for _ in range(3):
        step(state, cfg, xp, corr)
    assert int((state["species"] == 2).sum()) == 0

    old = genome(hunt=0.0, trophic=1, speed=1, sense=1, metabolism=0.0,
                 repro=0.0, lifespan=5)          # repro=0 -> не делится
    state, cfg, xp, corr = world([GRASS, old], n=10)
    put(state, cfg, 5, 5, 3, 2, 100.0)            # энергии полно, но век короткий
    for _ in range(4):
        step(state, cfg, xp, corr)
    assert int((state["species"] == 2).sum()) == 1
    for _ in range(4):
        step(state, cfg, xp, corr)
    assert int((state["species"] == 2).sum()) == 0


def test_reproduction_splits_energy():
    breeder = genome(hunt=0.0, trophic=1, speed=1, sense=1, metabolism=0.0, repro=4.0)
    state, cfg, xp, corr = world([GRASS, breeder], n=10)
    put(state, cfg, 5, 5, 3, 2, 10.0)
    step(state, cfg, xp, corr)
    cells = state["species"] == 2
    assert int(cells.sum()) == 2
    assert float(state["energy"][cells].sum()) == pytest.approx(10.0, rel=1e-5)


def test_gravity_pulls_animals_down():
    state, cfg, xp, corr = world([GRASS, HERB], n=12)
    put(state, cfg, 5, 5, 9, 2, 8.0)              # висит в воздухе
    step(state, cfg, xp, corr)
    z = int(np.argwhere(state["species"] == 2)[0][2])
    assert z < 9


def test_armor_protects():
    tough = genome(hunt=0.0, trophic=1, speed=1, sense=1, metabolism=0.0,
                   repro=99, armor=0.95)
    soft = genome(hunt=0.0, trophic=1, speed=1, sense=1, metabolism=0.0, repro=99)
    survived = {}
    for name, prey in (("броня", tough), ("без", soft)):
        alive = 0
        for trial in range(30):
            state, cfg, xp, corr = world([GRASS, prey, PRED], n=10)
            put(state, cfg, 5, 5, 3, 2, 5.0)
            put(state, cfg, 6, 5, 3, 3, 5.0)
            state["rng"] = np.random.default_rng(trial)
            step(state, cfg, xp, corr)
            alive += int((state["species"] == 2).any())
        survived[name] = alive / 30
    assert survived["броня"] > survived["без"] + 0.3, survived


def test_default_world_has_living_chain():
    res = run(Config(n=48, gens=200, seed_density=0.02), verbose=False)
    h = res["hist"]
    plants, herb, pred = h[:, :6].sum(axis=1), h[:, 6], h[:, 7]
    assert plants[-1] > 0 and herb[-1] > 0
    assert pred.max() > 0
    # травоядные не выедают мир под ноль
    assert plants[-1] > plants.max() * 0.2
