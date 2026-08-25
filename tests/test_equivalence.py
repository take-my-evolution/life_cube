"""Физика растений не изменилась с исходного монолита cube_ecology.py.

Сравнивать целые прогоны больше нельзя: генерация мира другая (камень на треть
куба, засев с животными). Поэтому сравнивается именно ШАГ: обоим движкам даётся
одно и то же начальное состояние из legacy-мира, и дальше N поколений должны
совпасть побитово при выключенных новых механиках (branch = 0, боковая вода
выключена, животных нет).
"""

import importlib.util
import pathlib

import numpy as np
import pytest
from scipy.ndimage import correlate as sp_correlate

from life_cube import Config
from life_cube.backend import get_backend
from life_cube.config import GENOME_FIELDS
from life_cube.sim import init_state
from life_cube.step import step

LEGACY = pathlib.Path(__file__).resolve().parents[1] / "legacy" / "cube_ecology.py"


@pytest.mark.skipif(not LEGACY.exists(), reason="нет legacy/cube_ecology.py")
def test_plant_step_matches_legacy():
    spec = importlib.util.spec_from_file_location("legacy_cube", LEGACY)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    n, gens = 32, 20
    lcfg = legacy.Config(n=n, gens=gens, seed_density=0.02)
    stone, wet, species, relief = legacy.build_world(lcfg, np)

    lstate = {"species": species.copy(), "stone": stone.copy(),
              "soil": np.zeros((n,) * 3, bool), "wet": wet.copy(),
              "kernel": legacy.build_kernel(np),
              "genomes": lcfg.genomes,
              "rng": np.random.default_rng(lcfg.seed_mut)}

    # тот же геном в новом формате: первые пять чисел совпадают, branch = 0
    g = np.zeros((4, len(GENOME_FIELDS)), np.float32)
    g[:, :5] = lcfg.genomes
    cfg = Config(n=n, gens=gens, seed_density=0.02, genomes=g,
                 lateral_decay=0.0, animal_share=0.0)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    state.update({"species": species.copy(), "stone": stone.copy(),
                  "soil": np.zeros((n,) * 3, bool), "wet": wet.copy(),
                  "energy": np.zeros((n,) * 3, np.float32),
                  "rng": np.random.default_rng(lcfg.seed_mut)})

    for gen in range(gens):
        lpops = legacy.step(lstate, lcfg, np, sp_correlate, gen)
        pops = step(state, cfg, xp, corr, gen)
        assert pops == lpops, gen
        assert np.array_equal(state["species"], lstate["species"]), gen
        assert np.array_equal(state["soil"], lstate["soil"]), gen
