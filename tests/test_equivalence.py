"""Модульная версия должна побитово совпадать с исходным монолитом
cube_ecology.py (лежит в legacy/), если он доступен."""

import importlib.util
import pathlib

import numpy as np
import pytest

from life_cube import Config, run

LEGACY = pathlib.Path(__file__).resolve().parents[1] / "legacy" / "cube_ecology.py"


@pytest.mark.skipif(not LEGACY.exists(), reason="нет legacy/cube_ecology.py")
def test_bitwise_equivalent_to_legacy():
    spec = importlib.util.spec_from_file_location("legacy_cube", LEGACY)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    kw = dict(n=32, gens=25, seed_density=0.02)
    new = run(Config(**kw), verbose=False)
    old = legacy.run(legacy.Config(**kw), verbose=False)
    assert np.array_equal(new["hist"], old["hist"])
    assert np.array_equal(new["species"], old["species"])
    assert np.array_equal(new["soil"], old["soil"])
