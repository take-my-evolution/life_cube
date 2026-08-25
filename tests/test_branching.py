"""Ветвление: вода течёт вбок по телу, боковое рождение только у видов с
branch > 0 и только на свету, крона затеняет то, что под ней."""

import numpy as np
import pytest

from life_cube import Config
from life_cube.backend import get_backend
from life_cube.config import DEFAULT_GENOMES
from life_cube.fields import water_field, water_supply, light_field
from life_cube.sim import init_state
from life_cube.step import step


def test_lateral_water_flows_along_body_and_stops_at_gap():
    n = 12
    cfg = Config(n=n, lateral_steps=4, lateral_decay=0.8)
    stone = np.zeros((n, n, n), bool); stone[:, :, 0] = True
    soil = np.zeros_like(stone); wet = np.ones((n, n), np.float32)
    alive = np.zeros_like(stone)
    alive[2, 2, 1:4] = True                      # ствол
    alive[3:7, 2, 3] = True                      # ветка вбок на z=3 (4 клетки)
    alive[9, 2, 3] = True                        # оторванная клетка за разрывом
    W = water_field(alive, stone, soil, wet, cfg, np)
    assert W[2, 2, 3] == pytest.approx(0.9 ** 3)
    assert W[3, 2, 3] == pytest.approx(0.9 ** 3 * 0.8)
    assert W[6, 2, 3] == pytest.approx(0.9 ** 3 * 0.8 ** 4)
    assert W[9, 2, 3] == 0.0                     # разрыв обрывает поток
    # без бокового течения ветка сухая — старое поведение
    W0 = water_field(alive, stone, soil, wet, Config(n=n, lateral_decay=0.0), np)
    assert W0[3, 2, 3] == 0.0 and W0[2, 2, 3] == pytest.approx(0.9 ** 3)
    # вода для рождения доступна сбоку от ветки
    Wsup = water_supply(W, cfg, np)
    assert Wsup[7, 2, 3] == pytest.approx(W[6, 2, 3] * 0.8)
    assert Wsup[6, 3, 3] == pytest.approx(W[6, 2, 3] * 0.8)


def _flat_world(n=16, branch=1.0, absorb=0.7, water=0.35):
    """Плоский камень, одна колонна вида 1 высотой 5. Возвращает state."""
    g = np.array([[absorb, 1.5, 1.75, 0.40, water, branch]], np.float32)
    cfg = Config(n=n, genomes=g, p_shock=0, p_dissolve=0, p_mutate=0, seed_density=0.01)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    stone = np.zeros((n, n, n), bool); stone[:, :, :3] = True
    state["stone"] = stone; state["soil"] = np.zeros_like(stone)
    state["wet"] = np.ones((n, n), np.float32)
    sp = np.zeros((n, n, n), np.int8); sp[8, 8, 3:8] = 1
    state["species"] = sp
    return state, cfg, xp, corr


def test_branching_only_with_gene():
    for branch, expect_side in ((0.0, False), (1.0, True)):
        state, cfg, xp, corr = _flat_world(branch=branch)
        for _ in range(3):
            step(state, cfg, xp, corr)
        sp = state["species"]
        side = (sp[:, :, 4:] > 0).copy(); side[8, 8, :] = False   # всё, что не в колонне, выше слоя 4
        assert bool(side.any()) == expect_side, (branch, int(side.sum()))


def test_branches_prefer_light_and_shade_below():
    state, cfg, xp, corr = _flat_world(branch=1.2)
    for _ in range(12):
        step(state, cfg, xp, corr)
    sp = state["species"]
    alive = sp > 0
    support = np.zeros_like(alive); support[:, :, 1:] = (alive | state["stone"] | state["soil"])[:, :, :-1]
    hanging = alive & ~support                      # ветви: висят без опоры снизу
    assert hanging.sum() >= 20
    assert hanging[:, :, 5:].sum() > hanging[:, :, 3:5].sum()   # ярус кроны выше земли
    # под кроной свет ослаблен, на открытом месте — полный
    absorb = np.where(alive, cfg.genomes[0][0], 0).astype(np.float32)
    L = light_field(alive, absorb, np)
    shaded = L[:, :, 3][hanging[:, :, 5:].any(axis=2)]      # земля под висящими клетками
    assert L[0, 0, 3] == 1.0 and shaded.max() <= 0.3 + 1e-6, shaded.max()


def test_default_world_grows_trees_and_keeps_others():
    cfg = Config(n=32, gens=40, seed_density=0.02)
    from life_cube import run
    res = run(cfg, verbose=False)
    pops = res["hist"][-1]
    assert cfg.n_species == 5 and pops[4] > 0          # дерево выжило
    assert (pops > 0).sum() >= 3
    # у дерева есть клетки без опоры снизу — ветви
    sp = res["species"]; alive = sp > 0
    support = np.zeros_like(alive); support[:, :, 1:] = (alive | res["stone"] | res["soil"])[:, :, :-1]
    hanging = (sp == 5) & ~support
    assert hanging.sum() > 0
    assert hanging.sum() / max((sp == 5).sum(), 1) > 0.05
