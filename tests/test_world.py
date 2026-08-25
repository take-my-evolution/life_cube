import numpy as np
import pytest

from life_cube.config import Config
from life_cube.world import build_kernel, build_world


def test_kernel_anisotropy():
    K = build_kernel(np)
    assert K.shape == (3, 3, 3)
    assert K[1, 1, 1] == 0                      # центр нулевой
    # снизу — опора, сверху — помеха
    assert K[1, 1, 0] == pytest.approx(1.6 * 1.25)
    assert K[1, 1, 2] == pytest.approx(0.45 * 1.25)
    assert K[1, 1, 0] > K[0, 1, 1] > K[1, 1, 2]
    # граневой > рёберный > угловой при равном dz
    assert K[0, 1, 1] > K[0, 0, 1] > K[0, 0, 0] * (1.0 / 1.6)  # угловой снизу
    assert np.count_nonzero(K) == 26


def test_world_deterministic_and_seeded():
    cfg = Config(n=32)
    a = build_world(cfg, np)
    b = build_world(cfg, np)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    c = build_world(Config(n=32, seed_world=1), np)
    assert not np.array_equal(a[3], c[3])       # другой сид — другой рельеф


def test_stone_fraction_controls_relief():
    for frac in (0.15, 0.33, 0.5):
        cfg = Config(n=48, stone_fraction=frac, seed_density=0.02)
        stone = build_world(cfg, np)[0]
        assert abs(stone.mean() - frac) < 0.08, (frac, stone.mean())


def test_world_layout():
    cfg = Config(n=32)
    stone, wet, species, relief, energy = build_world(cfg, np)
    n = cfg.n
    assert stone.shape == species.shape == (n, n, n)
    assert wet.shape == relief.shape == (n, n)
    assert relief.min() >= 2 and abs(relief.mean() - cfg.stone_fraction * n) < cfg.relief_amp * cfg.stone_fraction * n
    assert 0.15 <= wet.min() and wet.max() <= 1.0
    # споры лежат ровно на первом слое над камнем и не в камне
    assert not (stone & (species > 0)).any()
    xs, ys, zs = np.nonzero(species)
    assert np.array_equal(zs, relief[xs, ys])
    # представлены и растения, и животные; энергия расставлена
    counts = np.bincount(species[species > 0], minlength=cfg.n_species + 1)[1:]
    mobile = cfg.mobile_mask()
    assert counts[~mobile].sum() > 0 and counts[mobile].sum() > 0
    assert energy[species > 0].min() > 0


def test_empty_seed_raises():
    with pytest.raises(RuntimeError):
        build_world(Config(n=16, seed_density=0.0), np)
