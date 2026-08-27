"""Движок «терра»: рельеф из камня на треть куба, вода и реки, миграция почвы,
неравноценная эрозия, плодородие от бактерий, сукцессия видов."""

import numpy as np
import pytest

from life_cube.engine import Engine
from life_cube.engines import get_rules
from life_cube.engines.terra import GENES, IDX, MAX_SPECIES, TerraConfig, genome_color

R = get_rules("terra")


def make(n=48, **kw):
    cfg = TerraConfig(n=n, **kw)
    st, relief = R.init_state(cfg, np)
    return cfg, st


def test_world_has_stone_third_soil_and_water():
    cfg, st = make(n=64)
    sh, so, wa = st["stone_h"], st["soil_h"], st["water_h"]
    # камень — примерно треть высоты куба, а не тонкая корка
    assert 0.2 * 64 < sh.mean() < 0.5 * 64, sh.mean()
    assert sh.max() - sh.min() > 8                    # горы и долины, а не плоскость
    assert so.sum() > 0 and wa.sum() > 0              # почва и вода есть сразу
    # вода стоит в низинах, а не на вершинах
    low = sh <= np.percentile(sh, 25)
    high = sh >= np.percentile(sh, 75)
    assert wa[low].mean() > wa[high].mean()
    # лишайник — на голом камне, не в воде
    live = np.argwhere(st["species"] > 0)
    assert len(live) > 100
    for x, y, z in live[:200]:
        assert so[x, y] == 0 and wa[x, y] == 0


def test_relief_is_not_a_single_hill():
    """Многооктавный шум: у мира несколько вершин, а не один холм в центре."""
    cfg, st = make(n=64)
    sh = st["stone_h"].astype(float)
    peaks = 0
    for x in range(1, 63):
        for y in range(1, 63):
            v = sh[x, y]
            if v > np.percentile(sh, 80) and v >= sh[x-1:x+2, y-1:y+2].max():
                peaks += 1
    assert peaks >= 3, peaks


def test_erosion_is_unequal_one_stone_gives_several_soil():
    cfg, st = make(n=32, rain_rate=0.25, erode_rate=0.6, soil_per_stone=3)
    stone0, soil0 = int(st["stone_h"].sum()), int(st["soil_h"].sum())
    for g in range(30):
        R.step(st, cfg, np, None, g)
    d_stone = stone0 - int(st["stone_h"].sum())
    d_soil = int(st["soil_h"].sum()) - soil0
    assert d_stone > 0
    # почвы прибавилось примерно втрое больше, чем убыло камня (часть смыло в воду)
    assert d_soil > d_stone, (d_stone, d_soil)


def test_water_flows_down_and_carries_soil():
    cfg, st = make(n=48, rain_rate=0.2, erode_by_water=0.9)
    ground0 = (st["stone_h"] + st["soil_h"]).copy()
    for g in range(60):
        R.step(st, cfg, np, None, g)
    soil = st["soil_h"]
    steep = ground0 >= np.percentile(ground0, 80)
    flat = ground0 <= np.percentile(ground0, 20)
    assert soil[flat].mean() > soil[steep].mean()     # почва уехала вниз
    assert st["water_h"][flat].mean() > st["water_h"][steep].mean()


def test_no_rain_no_erosion():
    cfg, st = make(n=32, rain_rate=0.0)
    stone0 = int(st["stone_h"].sum())
    for g in range(40):
        R.step(st, cfg, np, None, g)
    # сухой мир: камень грызут только слегка (базовая влажность 0.15)
    assert int(st["stone_h"].sum()) >= stone0 * 0.97


def test_lichen_lives_on_stone_and_starves_on_soil():
    cfg, st = make(n=16, rain_rate=0.0, p_shock=0.0)
    st["species"][:] = 0
    st["soil_h"][:] = 0
    st["soil_h"][2:6, 2:6] = 2
    st["water_h"][:] = 0
    ground = st["stone_h"] + st["soil_h"]
    cfg.genomes[0, IDX["repro"]] = 99                 # без деления
    st["genomes"] = np.asarray(cfg.genomes)
    st["species"][3, 3, ground[3, 3]] = 1             # на почве
    st["species"][12, 12, ground[12, 12]] = 1         # на камне
    st["energy"][3, 3, ground[3, 3]] = 1.0
    st["energy"][12, 12, ground[12, 12]] = 1.0
    for g in range(12):
        R.step(st, cfg, np, None, g)
    gr = st["stone_h"] + st["soil_h"]
    e_soil = float(st["energy"][3, 3, gr[3, 3]])
    e_stone = float(st["energy"][12, 12, gr[12, 12]])
    assert e_stone > 1.0 > e_soil, (e_stone, e_soil)


def test_bacteria_fertilise_soil_and_symbiont_needs_it():
    """Бактерия поднимает плодородие столбца; симбионт на пустой почве голодает,
    на удобрённой — кормится."""
    cfg, st = make(n=16, rain_rate=0.0, p_shock=0.0)
    st["species"][:] = 0
    st["soil_h"][:] = 2
    st["water_h"][:] = 0
    ground = st["stone_h"] + st["soil_h"]
    bact = cfg.genomes[0].copy()
    bact[IDX["soil"]], bact[IDX["stone"]], bact[IDX["enrich"]] = 1.0, 0.0, 1.0
    bact[IDX["repro"]] = 99
    plant = bact.copy(); plant[IDX["enrich"]] = 0.0; plant[IDX["symbiont"]] = 1.0
    cfg.genomes[1], cfg.genomes[2] = bact, plant
    for sid in (2, 3):
        st["free_ids"].remove(sid)
        st["registry"][sid] = {"parent": 1, "born": 0, "died": None,
                               "genome": cfg.genomes[sid - 1].tolist(), "peak": 1,
                               "changed": "soil"}
    st["genomes"] = np.asarray(cfg.genomes)
    st["species"][4, 4, ground[4, 4]] = 2             # бактерия
    st["species"][4, 5, ground[4, 5]] = 3             # симбионт рядом с бактерией
    st["species"][12, 12, ground[12, 12]] = 3         # симбионт в чистом поле
    for c in ((4, 4), (4, 5), (12, 12)):
        st["energy"][c[0], c[1], ground[c[0], c[1]]] = 1.0
    for g in range(40):
        R.step(st, cfg, np, None, g)
    assert st["fert"][4, 4] > 0.3                     # бактерия удобрила почву
    assert st["fert"][12, 12] == 0
    gr = st["stone_h"] + st["soil_h"]
    near = float(st["energy"][4, 5, gr[4, 5]])
    far = float(st["energy"][12, 12, gr[12, 12]])
    assert near > far, (near, far)


def test_succession_from_lichen_to_plants():
    """Долгий прогон: из одного лишайника появляются почвенные виды, а затем
    и растения (ген роста вверх)."""
    cfg, st = make(n=48, max_new_species=4)
    kinds = set()
    for g in range(1, 901):
        R.step(st, cfg, np, None, g)
        if g % 100 == 0:
            for sid in st["registry"]:
                if int((st["species"] == sid).sum()) > 3:
                    kinds.add(R._kind(cfg.genomes[sid - 1]))
    assert len(st["registry"]) + len(st["lineage"]) > 30      # эволюция идёт
    assert kinds & {"почвенный", "универсал"}, kinds          # почву заселили
    soil_genes = [cfg.genomes[s - 1][IDX["soil"]] for s in st["registry"]]
    assert max(soil_genes) > 0.4                              # ген почвы вырос


def test_species_colour_comes_from_genome():
    g = np.zeros(len(GENES), np.float32)
    g[IDX["stone"]] = 1.0
    stone_col = genome_color(g)
    g2 = g.copy(); g2[IDX["stone"]] = 0.0; g2[IDX["water"]] = 1.0
    water_col = genome_color(g2)
    g3 = g.copy(); g3[IDX["stone"]] = 0.0; g3[IDX["soil"]] = 1.0
    soil_col = genome_color(g3)
    assert stone_col != water_col != soil_col
    to_rgb = lambda h: tuple(int(h[i:i+2], 16) for i in (1, 3, 5))
    assert to_rgb(water_col)[2] > to_rgb(water_col)[0]        # вода синеватая
    assert to_rgb(soil_col)[1] >= max(to_rgb(soil_col))       # почва зеленоватая
    assert min(to_rgb(stone_col)) > 100                       # камень светлый


def test_engine_runs_and_reports_heightmaps():
    e = Engine(TerraConfig(n=32), rate=0, components=False, rules="terra")
    for _ in range(5):
        e.advance()
    snap = e.publish(force=True)
    assert snap.stone_h is not None and snap.soil_h is not None and snap.water_h is not None
    assert snap.soil_coords is None                  # подложка не поклеточно
    assert snap.stone_h.shape == (32, 32)
    from life_cube.viewers.web.server import encode_snapshot, decode_snapshot
    sent = {}
    hdr, coords, species, labels, soil = decode_snapshot(encode_snapshot(snap, first=True, sent=sent))
    assert hdr["heightmaps"] and hdr["maps"]["stone_h"]
    assert np.array_equal(hdr["stone_h"], snap.stone_h)
    # второй кадр без изменений рельефа его не повторяет
    hdr2, *_ = decode_snapshot(encode_snapshot(snap, sent=sent))
    assert not hdr2.get("maps", {}).get("stone_h")
