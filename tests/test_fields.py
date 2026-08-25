import numpy as np
import pytest

from life_cube.config import Config, DEFAULT_GENOMES
from life_cube.fields import light_field, water_field, resource


def test_light_shading():
    n = 6
    alive = np.zeros((n, n, n), bool)
    absorb = np.zeros((n, n, n), np.float32)
    alive[0, 0, 4] = True
    absorb[0, 0, 4] = 0.5
    L = light_field(alive, absorb, np)
    assert L[0, 0, 5] == 1.0
    assert L[0, 0, 4] == 1.0            # сама клетка видит полный свет
    assert L[0, 0, 3] == pytest.approx(0.5)  # под ней — тень
    assert (L[1, 1, :] == 1.0).all()    # соседний столбец не затенён


def test_water_column_and_break():
    n = 8
    cfg = Config(n=n)
    stone = np.zeros((n, n, n), bool); stone[:, :, 0] = True
    soil = np.zeros_like(stone)
    wet = np.ones((n, n), np.float32)
    alive = np.zeros_like(stone)
    alive[0, 0, 1:4] = True              # сплошной столб z=1..3
    alive[1, 1, 1] = True; alive[1, 1, 3] = True   # разрыв на z=2
    W = water_field(alive, stone, soil, wet, cfg, np)
    assert W[0, 0, 0] == 1.0
    assert W[0, 0, 1] == pytest.approx(0.9)
    assert W[0, 0, 3] == pytest.approx(0.9 ** 3)
    assert W[0, 0, 4] == 0.0             # пустая клетка воды не держит
    assert W[1, 1, 3] == 0.0             # разрыв обрывает столб
    # почва держит воду лучше камня
    soil[2, 2, 1] = True
    W2 = water_field(alive, stone, soil, wet * 0.5, cfg, np)
    assert W2[2, 2, 1] == pytest.approx(0.5 * 1.15)


def test_resource_mix():
    g = DEFAULT_GENOMES[3]     # теневой: water=0.85
    L = np.array([1.0], np.float32); W = np.array([0.0], np.float32)
    r_light = resource(g, L, W, np)[0]
    r_water = resource(g, W, L, np)[0]
    assert r_water > r_light
    g2 = DEFAULT_GENOMES[2]    # башня: water=0.25
    assert resource(g2, L, W, np)[0] > resource(g2, W, L, np)[0]
