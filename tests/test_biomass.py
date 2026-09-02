"""Биомасса и экономика роста растений.

До этого клетка была клеткой: все шесть растений удваивались за ОДНО
поколение, гены metabolism/repro/lifespan у растений не читались вовсе, а
травоядное получало одинаковую константу за клетку мха и за клетку дерева.
"""

import numpy as np
import pytest

from life_cube import Config
from life_cube.backend import get_backend
from life_cube.config import DEFAULT_GENOMES, IDX, SPECIES_NAMES
from life_cube.sim import init_state
from life_cube.step import step


def _mono(species_i, gens, **kw):
    """Мир из одного вида растений: чистая скорость роста, без конкуренции."""
    cfg = Config(n=32, genomes=DEFAULT_GENOMES[species_i].reshape(1, -1).copy(),
                 seed_density=0.02, animal_share=0.0, **kw)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    k0 = int((state["species"] > 0).sum())
    hist = [k0]
    for g in range(gens):
        step(state, cfg, xp, corr, g)
        hist.append(int((state["species"] > 0).sum()))
    return cfg, state, hist


def _doubling(hist):
    k0 = hist[0]
    return next((t for t, v in enumerate(hist) if v >= 2 * k0), None)


def test_species_grow_at_different_speeds():
    """Главное: скорость роста больше не одинакова. Раньше ВСЕ шесть видов
    удваивали биомассу за одно поколение — мох рос ровно так же быстро, как
    дерево, потому что рождение было чистой проверкой места, без цены."""
    dbl = {}
    for i in range(6):
        _, _, hist = _mono(i, 200)
        d = _doubling(hist)
        dbl[SPECIES_NAMES[i]] = d if d is not None else 999
    assert all(d > 1 for d in dbl.values()), dbl        # никто не удваивается за такт
    fast, slow = min(dbl.values()), max(dbl.values())
    assert slow > fast * 4, dbl                          # разброс скоростей реальный
    # корка — пионер, дерево — самый медленный
    assert dbl["корка"] < dbl["мох"] < dbl["дерево"], dbl


def test_growth_costs_energy_and_drains_the_neighbourhood():
    """Клетка копит энергию и тратит её на постройку новой. Если отобрать
    накопления, рост обязан встать."""
    cfg, state, _ = _mono(0, 30)                 # мох
    xp, corr, _ = get_backend(False)
    assert float(state["energy"][state["species"] > 0].mean()) > 0

    before = int((state["species"] > 0).sum())
    state["energy"][:] = 0.0                     # обнулили накопления
    for g in range(10):
        step(state, cfg, xp, corr, g)
    after_poor = int((state["species"] > 0).sum())

    cfg2, state2, _ = _mono(0, 30)
    b2 = int((state2["species"] > 0).sum())
    for g in range(10):
        step(state2, cfg2, xp, corr, g)
    after_rich = int((state2["species"] > 0).sum())
    assert (after_rich - b2) > (after_poor - before), (after_poor - before, after_rich - b2)


def test_cell_mass_differs_and_drives_biomass():
    """Биомасса — не число клеток: клетка дерева тяжелее клетки мха."""
    from life_cube.engines.ecology import RULES
    cfg = Config()
    mass = RULES.species_mass(cfg)
    assert len(mass) == cfg.n_species
    i_moss, i_tree = SPECIES_NAMES.index("мох"), SPECIES_NAMES.index("дерево")
    assert mass[i_tree] > mass[i_moss] * 4, mass
    assert len(set(mass[:6])) >= 5, mass          # массы у видов разные

    # снимок несёт биомассу отдельно от населения
    from life_cube.engine import Engine
    e = Engine(Config(n=24, seed_density=0.05), rate=0)
    e.advance()
    snap = e.publish(force=True)
    assert snap.biomass is not None and len(snap.biomass) == cfg.n_species
    for i, (p, b) in enumerate(zip(snap.pops, snap.biomass)):
        assert b == pytest.approx(p * mass[i], rel=1e-3), (i, p, b, mass[i])
    # и это НЕ то же самое, что просто число клеток
    assert sum(snap.biomass) != pytest.approx(sum(snap.pops))


def test_food_value_follows_mass_not_a_constant():
    """Травоядное получает за клетку её МАССУ, а не общую константу: раньше
    клетка мха кормила ровно так же, как клетка дерева."""
    from life_cube.motion import animals_step
    xp, corr, _ = get_backend(False)
    i_moss, i_tree = SPECIES_NAMES.index("мох"), SPECIES_NAMES.index("дерево")
    gains = {}
    for name, prey_i in (("мох", i_moss), ("дерево", i_tree)):
        g = np.zeros((2, DEFAULT_GENOMES.shape[1]), np.float32)
        g[0] = DEFAULT_GENOMES[prey_i]          # жертва — растение
        g[1] = DEFAULT_GENOMES[6]               # травоядное
        g[1, IDX["hunt"]] = 1.0                 # ест наверняка
        g[1, IDX["armor"]] = 0.0
        n = 8
        cfg = Config(n=n, genomes=g, seed_density=0.05, p_shock=0.0)
        state, _ = init_state(cfg, xp)
        sp = np.zeros((n, n, n), np.int8)
        stone = np.zeros((n, n, n), bool); stone[:, :, :2] = True
        sp[4, 4, 2] = 1                          # растение
        sp[4, 5, 2] = 2                          # травоядное вплотную
        state.update(species=sp, stone=stone, soil=np.zeros_like(stone),
                     energy=np.zeros((n, n, n), np.float32),
                     age=np.zeros((n, n, n), np.int32))
        state["energy"][4, 4, 2] = 1.0           # запас жертвы одинаков у обоих прогонов
        animals_step(state, cfg, xp, corr, state["rng"])
        gains[name] = float(state["energy"][state["species"] == 2].sum())
    assert gains["дерево"] > gains["мох"] * 3, gains


def test_zero_genes_keep_the_old_behaviour():
    """Совместимость: repro=0 / metabolism=0 / lifespan=0 выключают экономику,
    и вид растёт как раньше — на этом стоят legacy-сравнение и старые пресеты."""
    g = DEFAULT_GENOMES[0].copy()
    g[IDX["repro"]] = 0.0
    g[IDX["metabolism"]] = 0.0
    g[IDX["lifespan"]] = 0.0
    cfg = Config(n=32, genomes=g.reshape(1, -1), seed_density=0.02, animal_share=0.0)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    k0 = int((state["species"] > 0).sum())
    step(state, cfg, xp, corr, 0)
    assert int((state["species"] > 0).sum()) >= 2 * k0   # снова удвоение за такт


def test_plants_die_of_old_age():
    """Ген lifespan у растений наконец работает: раньше он не читался вовсе
    и растительная клетка не старела никогда."""
    xp, corr, _ = get_backend(False)
    g = DEFAULT_GENOMES[1].copy()          # корка, lifespan 400
    g[IDX["lifespan"]] = 5.0               # укорачиваем, чтобы тест был быстрым
    cfg = Config(n=16, genomes=g.reshape(1, -1), seed_density=0.05,
                 animal_share=0.0, p_shock=0.0)
    state, _ = init_state(cfg, xp)
    state["age"][state["species"] > 0] = 50      # все клетки заведомо старые
    step(state, cfg, xp, corr, 0)
    old_survivors = int(((state["species"] > 0) & (state["age"] > 5)).sum())
    assert old_survivors == 0, old_survivors
