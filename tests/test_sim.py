import numpy as np

from life_cube import Config, run
from life_cube.backend import get_backend
from life_cube.sim import init_state
from life_cube.step import step


def small(**kw):
    return Config(n=32, gens=kw.pop("gens", 30), seed_density=0.02, **kw)


def test_run_shapes_and_history():
    cfg = small()
    res = run(cfg, verbose=False)
    n = cfg.n
    assert res["species"].shape == (n, n, n)
    assert res["hist"].shape == (cfg.gens, cfg.n_species)
    assert (res["hist"] >= 0).all()
    assert not (res["stone"] & (res["species"] > 0)).any()
    assert not (res["stone"] & res["soil"]).any()


def test_life_survives_and_grows():
    cfg = small(gens=40)
    res = run(cfg, verbose=False)
    total0 = res["hist"][0].sum()
    total_end = res["hist"][-1].sum()
    assert total_end > 0, "жизнь вымерла"
    assert total_end > total0, "жизнь не растёт"
    # больше одного вида дожило
    assert (res["hist"][-1] > 0).sum() >= 2


def test_reproducible_and_seeds_independent():
    a = run(small(), verbose=False)
    b = run(small(), verbose=False)
    assert np.array_equal(a["species"], b["species"])
    assert np.array_equal(a["hist"], b["hist"])
    # другой сид мутаций — тот же ландшафт, другая история
    c = run(small(seed_mut=7), verbose=False)
    assert np.array_equal(a["stone"] | a["soil"], c["stone"] | c["soil"])
    assert np.array_equal(a["relief"], c["relief"])
    assert not np.array_equal(a["species"], c["species"])


def test_step_no_lonely_death_and_crowd_death():
    """Одиночка не умирает от одиночества; давка убивает."""
    from life_cube.config import DEFAULT_GENOMES
    plants = DEFAULT_GENOMES[DEFAULT_GENOMES[:, 6] == 0]      # без хищников:
    # хищник без добычи голодает и умирает — это отдельная механика (test_predation)
    cfg = Config(n=12, p_shock=0.0, p_dissolve=0.0, p_mutate=0.0, seed_density=0.05,
                 genomes=plants)
    xp, correlate, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    sp0 = state["species"].copy()
    step(state, cfg, xp, correlate)
    # каждая исходная спора (на камне, с водой и светом) должна выжить
    assert ((state["species"] > 0) & (sp0 > 0)).sum() == (sp0 > 0).sum()

    # искусственно плотный шар -> центр умирает от тесноты
    cfg2 = Config(n=12, p_shock=0.0, p_dissolve=0.0, seed_density=0.05, genomes=plants)
    state, _ = init_state(cfg2, xp)
    s = np.zeros((12, 12, 12), np.int8)
    s[3:9, 3:9, 4:10] = 1                 # над камнем (relief >= 3)
    state["species"] = s
    step(state, cfg2, xp, correlate)
    assert state["species"][6, 6, 7] == 0


def test_save_state(tmp_path):
    from life_cube.sim import save_state
    res = run(small(gens=5), verbose=False)
    p = save_state(res, str(tmp_path / "s.npz"))
    d = np.load(p)
    assert set(d.files) >= {"species", "stone", "soil", "relief", "hist"}


def test_cli_and_render(tmp_path):
    from life_cube.cli import main
    out = tmp_path / "pic.png"
    main(["--n", "24", "--gens", "6", "--quiet", "--out", str(out)])
    assert out.exists() and out.stat().st_size > 10_000
