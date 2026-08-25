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


def test_world_layout():
    cfg = Config(n=32)
    stone, wet, species, relief = build_world(cfg, np)
    n = cfg.n
    assert stone.shape == species.shape == (n, n, n)
    assert wet.shape == relief.shape == (n, n)
    assert relief.min() >= 3 and relief.max() <= max(4, n // 7)
    assert 0.15 <= wet.min() and wet.max() <= 1.0
    # споры лежат ровно на первом слое над камнем и не в камне
    assert not (stone & (species > 0)).any()
    xs, ys, zs = np.nonzero(species)
    assert np.array_equal(zs, relief[xs, ys])
    # все четыре вида представлены примерно поровну
    counts = np.bincount(species[species > 0], minlength=cfg.n_species + 1)[1:]
    assert (counts > 0).all() and counts.max() - counts.min() <= 1


def test_empty_seed_raises():
    with pytest.raises(RuntimeError):
        build_world(Config(n=16, seed_density=0.0), np)
