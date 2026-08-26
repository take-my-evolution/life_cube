"""Движок «лишайник»: рельеф, дождь и сток, эрозия камня в почву, смыв почвы
в низины, стресс на чужом субстрате, мутации и появление новых видов."""

import numpy as np
import pytest

from life_cube.engine import Engine
from life_cube.engines import get_rules, list_engines
from life_cube.engines.lichen import GENES, IDX, MAX_SPECIES, LichenConfig

R = get_rules("lichen")


def make(n=32, **kw):
    cfg = LichenConfig(n=n, **kw)
    state, relief = R.init_state(cfg, np)
    return cfg, state


def test_registry_lists_both_engines():
    names = [e["name"] for e in list_engines()]
    assert names == ["ecology", "lichen"]
    assert get_rules("ecology").Config is not get_rules("lichen").Config


def test_world_flat_floor_hill_and_full_lichen_cover():
    cfg, st = make(n=48)
    sh = st["stone_h"]
    assert sh.min() == cfg.floor                              # плоское дно
    assert sh.max() >= 0.8 * cfg.stone_fraction * 48          # гора
    assert (sh == cfg.floor).mean() > 0.3                     # дно занимает заметную площадь
    n = 48
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    assert (st["species"][xx, yy, sh] == 1).all()             # весь камень укрыт лишайником
    assert int((st["species"] > 0).sum()) == n * n            # и только поверхность
    assert st["soil_h"].sum() == 0


def test_rain_falls_and_flows_downhill():
    cfg, st = make(n=32, rain_rate=0.2)
    R.step(st, cfg, np, None, 0)
    assert st["wet"].sum() > 0
    # вода стекает: через десяток шагов дно мокрее склонов
    for g in range(1, 15):
        R.step(st, cfg, np, None, g)
    floor = st["stone_h"] == cfg.floor
    slope = st["stone_h"] > cfg.floor + 3
    if slope.any():
        assert st["wet"][floor].mean() > st["wet"][slope].mean()


def test_erosion_converts_stone_to_soil_only_when_wet():
    dry, st_dry = make(n=32, rain_rate=0.0)
    wet, st_wet = make(n=32, rain_rate=0.3)
    for g in range(40):
        R.step(st_dry, dry, np, None, g)
        R.step(st_wet, wet, np, None, g)
    assert st_dry["soil_h"].sum() == 0
    assert st_wet["soil_h"].sum() > 0
    # камень не исчез ниже дна
    assert st_wet["stone_h"].min() >= wet.floor
    # объём вещества сохраняется: камень + почва = исходный камень
    total0 = int(st_dry["stone_h"].sum())
    assert int(st_wet["stone_h"].sum() + st_wet["soil_h"].sum()) == total0


def test_soil_washes_into_lowlands():
    cfg, st = make(n=48, rain_rate=0.25, wash=0.8)
    for g in range(150):
        R.step(st, cfg, np, None, g)
    soil = st["soil_h"]
    low = st["stone_h"] <= cfg.floor + 1
    high = st["stone_h"] >= cfg.floor + 4
    assert soil[low].mean() > soil[high].mean()               # почва копится внизу


def test_lichen_starves_on_soil_and_thrives_on_stone():
    """Один и тот же лишайник: на камне энергия растёт, на почве падает."""
    cfg, st = make(n=16, rain_rate=0.0, p_shock=0.0)
    st["species"][:] = 0
    st["soil_h"][:] = 0
    st["soil_h"][2:6, 2:6] = 1                                # островок почвы
    R._sync_volumes(st, cfg, np)
    surf = st["stone_h"] + st["soil_h"]
    st["species"][3, 3, surf[3, 3]] = 1                       # на почве
    st["species"][10, 10, surf[10, 10]] = 1                   # на камне
    st["energy"][3, 3, surf[3, 3]] = 1.0; st["energy"][10, 10, surf[10, 10]] = 1.0
    cfg.genomes[0, IDX["repro"]] = 99                         # без деления
    st["genomes"] = np.asarray(cfg.genomes)
    for g in range(10):
        R.step(st, cfg, np, None, g)
    e_soil = float(st["energy"][3, 3, surf[3, 3]])
    e_stone = float(st["energy"][10, 10, surf[10, 10]])
    assert e_stone > 1.0 and e_soil < 1.0, (e_stone, e_soil)


def test_mutation_creates_species_and_lineage():
    cfg, st = make(n=32, rain_rate=0.2, stress_mut=0.6, max_new_species=4)
    for g in range(120):
        R.step(st, cfg, np, None, g)
    reg = st["registry"]
    assert len(reg) > 1
    total = len(reg) + len(st["lineage"])
    assert total > 5
    # у каждого нового вида есть родитель и изменённый ген
    for sid, r in reg.items():
        if sid != 1:
            assert r["parent"] >= 1 and r["changed"] in GENES
            assert not np.allclose(r["genome"], cfg.genomes[r["parent"] - 1]) or r["parent"] not in reg
    # id не больше предела, вымершие освобождают слоты
    assert max(reg) <= MAX_SPECIES
    assert len(reg) + len(st["free_ids"]) == MAX_SPECIES


def test_no_mutation_without_stress_and_base_rate():
    cfg, st = make(n=24, rain_rate=0.0, stress_mut=0.0)
    cfg.genomes[0, IDX["mut"]] = 0.0
    st["genomes"] = np.asarray(cfg.genomes)
    for g in range(60):
        R.step(st, cfg, np, None, g)
    assert len(st["registry"]) == 1 and not st["lineage"]


def test_soil_adapted_species_is_favoured_on_soil():
    """Искусственный мутант с substrate=0.7 на почве набирает энергию и
    вытесняет голодающих соседей-лишайников."""
    cfg, st = make(n=24, rain_rate=0.0, p_shock=0.0, stress_mut=0.0)
    cfg.genomes[0, IDX["mut"]] = 0.0
    st["soil_h"][:] = 1                                       # весь мир — почва
    R._sync_volumes(st, cfg, np)
    surf = st["stone_h"] + st["soil_h"]
    st["species"][:] = 0
    xx, yy = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
    st["species"][xx, yy, surf] = 1                           # лишайник по всей почве
    st["energy"][st["species"] > 0] = 0.6
    g = cfg.genomes[0].copy(); g[IDX["substrate"]] = 0.7
    cfg.genomes[1] = g; st["free_ids"].remove(2)
    st["registry"][2] = {"parent": 1, "born": 0, "died": None, "genome": g.tolist(), "peak": 1, "changed": "substrate"}
    st["species"][12, 12, surf[12, 12]] = 2
    st["genomes"] = np.asarray(cfg.genomes)
    for gen in range(80):
        R.step(st, cfg, np, None, gen)
    n2 = int((st["species"] == 2).sum())
    assert n2 > 20, n2


def test_engine_switch_and_randomize_keep_running():
    e = Engine(rate=0, components=False, rules="ecology")
    e.cfg.n = 24
    e.reset(e.rules.Config(n=24, seed_density=0.03))
    e.advance()
    e.switch_rules("lichen", LichenConfig(n=24))
    assert e.rules.name == "lichen" and e.gen == 0
    for _ in range(3):
        e.advance()
    assert sum(e.hist[-1]) > 0
    e.randomize(seed=1)
    assert e.gen == 0 and e.cfg.genomes[0].any()
    e.advance()
    j = e.rules.to_json(e.cfg, e.state)
    assert j["engine"] == "lichen" and j["fields"] == list(GENES) and j["ids"][0] == 1
