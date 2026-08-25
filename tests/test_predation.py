"""Хищники: питание за счёт соседей-растений, гибель жертв, голод без добычи."""

import numpy as np
import pytest

from life_cube import Config, run
from life_cube.backend import get_backend
from life_cube.config import DEFAULT_GENOMES
from life_cube.sim import init_state
from life_cube.step import step

PLANT = [0.30, 1.0, 1.60, 0.30, 0.40, 0.20, 0.00]
HUNTER = [0.00, 1.0, 1.60, 0.85, 0.00, 0.60, 0.90]


def _world(n=14, genomes=(PLANT, HUNTER), **kw):
    g = np.array(list(genomes), np.float32)
    # crowd_max поднят: в этих тестах проверяется хищничество, а не смерть от
    # тесноты (сплошной ковёр даёт ~9 взвешенных соседей и умирал бы сам)
    kw.setdefault("crowd_max", 20.0)
    cfg = Config(n=n, genomes=g, p_shock=0.0, p_dissolve=0.0, p_mutate=0.0,
                 seed_density=0.01, **kw)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    stone = np.zeros((n, n, n), bool); stone[:, :, :3] = True
    state["stone"] = stone; state["soil"] = np.zeros_like(stone)
    state["wet"] = np.ones((n, n), np.float32)
    state["species"] = np.zeros((n, n, n), np.int8)
    return state, cfg, xp, corr


def test_hunter_starves_without_prey():
    """Хищник не кормится ни светом, ни водой: в одиночестве вымирает."""
    state, cfg, xp, corr = _world()
    state["species"][6:9, 6:9, 3] = 2
    for _ in range(6):
        step(state, cfg, xp, corr)
    assert int((state["species"] == 2).sum()) == 0


def test_hunter_lives_next_to_prey_and_kills_it():
    """Рядом с зарослями растений хищник живёт, размножается и выедает их."""
    state, cfg, xp, corr = _world(n=20)
    # разреженный ковёр: сплошной не даёт рождений (окно соседей birth_window)
    xs, ys = np.meshgrid(np.arange(2, 18), np.arange(2, 18), indexing="ij")
    mask = (xs + ys) % 2 == 0
    state["species"][xs[mask], ys[mask], 3] = 1
    state["species"][9, 9, 3] = 2; state["species"][10, 10, 3] = 2
    killed_total = 0
    hunters0 = int((state["species"] == 2).sum())
    for _ in range(10):
        before = state["species"].copy()
        step(state, cfg, xp, corr)
        killed_total += int(((before == 1) & (state["species"] == 0)).sum())
    sp = state["species"]
    # рядом с добычей хищник не голодает (сравни с test_hunter_starves…),
    # хотя за пустые клетки он проигрывает растениям — они плодятся быстрее
    assert int((sp == 2).sum()) >= hunters0
    assert killed_total > 0                      # жертвы гибли
    # гибнут именно соседи хищников: без них тот же мир не теряет ни клетки
    st, cfg2, xp2, c2 = _world(n=20)
    xs2, ys2 = np.meshgrid(np.arange(2, 18), np.arange(2, 18), indexing="ij")
    m2 = (xs2 + ys2) % 2 == 0
    st["species"][xs2[m2], ys2[m2], 3] = 1
    lost = 0
    for _ in range(10):
        before = st["species"].copy()
        step(st, cfg2, xp2, c2)
        lost += int(((before == 1) & (st["species"] == 0)).sum())
    assert lost == 0, lost


def test_predator_does_not_eat_predator():
    """Хищники друг друга не едят: сплошное пятно хищника без растений мрёт
    от голода, а не выедает само себя (иначе был бы вечный двигатель)."""
    state, cfg, xp, corr = _world()
    state["species"][4:10, 4:10, 3] = 2
    n0 = int((state["species"] == 2).sum())
    step(state, cfg, xp, corr)
    n1 = int((state["species"] == 2).sum())
    assert n1 < n0                               # голод, а не рост


def test_kill_probability_scales_with_pressure():
    """Чем больше хищных соседей у жертвы, тем выше шанс погибнуть."""
    def survival(hunters):
        alive = 0
        for _ in range(200):                     # много попыток: это вероятность
            st, cfg2, xp2, c2 = _world(n=16, kill_rate=0.5)
            st["species"][2:14, 2:14, 3] = 1
            if hunters:
                for (x, y) in ((7, 6), (7, 8), (6, 7), (8, 7)):
                    st["species"][x, y, 3] = 2
            st["rng"] = np.random.default_rng(_)
            step(st, cfg2, xp2, c2)
            alive += int(st["species"][7, 7, 3] == 1)
        return alive / 200
    assert survival(False) == 1.0                # без хищников жертва цела
    # 4 граневых хищника из ~9 взвешенных соседей: pressure ≈ 0.5,
    # p_eat ≈ kill_rate/2 = 0.25 -> выживает около 3/4
    assert 0.6 < survival(True) < 0.9


def test_default_world_supports_hunters_and_plants():
    res = run(Config(n=32, gens=60, seed_density=0.02), verbose=False)
    pops = res["hist"][-1]
    assert len(pops) == 8
    assert pops[5] > 0 or pops[6] > 0            # хищник или лиана живы
    plants = pops[[0, 1, 2, 3, 4, 7]].sum()
    assert plants > 0                            # растения не выедены под ноль
    assert (pops > 0).sum() >= 5


def test_genome_backward_compatible():
    """Геном без поля hunt (6 чисел) работает как раньше — без хищничества."""
    g6 = DEFAULT_GENOMES[:5, :6].copy()
    res = run(Config(n=24, gens=15, seed_density=0.03, genomes=g6), verbose=False)
    assert res["hist"][-1].sum() > 0
