"""Кем заселять мир и повторный засев.

Две отдельные вещи, которые легко перепутать:
  * `start_species` — кем мир засевается ПРИ СОЗДАНИИ (можно «только мох»);
  * `reseed` — спасательный круг: вымерший мир получает новую жизнь,
    но не чаще, чем раз в `reseed_every` поколений.
"""

import numpy as np
import pytest

from life_cube.backend import get_backend, to_cpu
from life_cube.engine import Engine
from life_cube.engines import get_rules


@pytest.fixture
def xp():
    return get_backend(False)


def test_start_species_only_moss():
    r = get_rules("ecology")
    cfg = r.Config(n=32, start_species=(1,))
    x, _c, _g = get_backend(False)
    state, _ = r.init_state(cfg, x)
    present = set(np.unique(to_cpu(state["species"])).tolist()) - {0}
    assert present == {1}


def test_start_species_moss_and_herbivore():
    r = get_rules("ecology")
    cfg = r.Config(n=32, start_species=(1, 7), animal_share=0.4)
    x, _c, _g = get_backend(False)
    state, _ = r.init_state(cfg, x)
    present = set(np.unique(to_cpu(state["species"])).tolist()) - {0}
    assert present <= {1, 7} and 1 in present


def test_empty_start_species_means_everyone():
    r = get_rules("ecology")
    cfg = r.Config(n=32)
    x, _c, _g = get_backend(False)
    state, _ = r.init_state(cfg, x)
    present = set(np.unique(to_cpu(state["species"])).tolist()) - {0}
    assert len(present) > 2


def test_terra_starters_seed_their_habitat():
    r = get_rules("terra")
    cfg = r.Config(n=48, start_species=(0, 1, 2))
    x, _c, _g = get_backend(False)
    state, _ = r.init_state(cfg, x)
    starters = {v.get("starter") for v in state["registry"].values()}
    assert starters == {"лишайник", "бактерия", "водоросль"}
    assert int((to_cpu(state["species"]) > 0).sum()) > 0


def test_reseed_revives_extinct_world():
    e = Engine(rules="ecology", cfg=get_rules("ecology").Config(n=32, reseed=True,
                                                               reseed_every=5,
                                                               reseed_count=100),
               rate=0, snapshot_every=0)
    e.state["species"][:] = 0                    # мир вымер
    for _ in range(3):
        pops = e.advance()
    assert sum(pops) > 0 and e.reseeds == 1


def test_reseed_off_leaves_world_dead():
    e = Engine(rules="ecology", cfg=get_rules("ecology").Config(n=32, reseed=False),
               rate=0, snapshot_every=0)
    e.state["species"][:] = 0
    for _ in range(3):
        pops = e.advance()
    assert sum(pops) == 0 and e.reseeds == 0


def test_reseed_does_not_fire_while_world_is_alive():
    """Живой мир не должен получать подкрепление: это стёрло бы эволюцию."""
    e = Engine(rules="ecology", cfg=get_rules("ecology").Config(n=32, reseed=True,
                                                               reseed_every=1),
               rate=0, snapshot_every=0)
    for _ in range(5):
        e.advance()
    assert e.reseeds == 0


def test_reseed_respects_interval():
    e = Engine(rules="ecology", cfg=get_rules("ecology").Config(n=32, reseed=True,
                                                               reseed_every=1000,
                                                               reseed_count=50),
               rate=0, snapshot_every=0)
    e.state["species"][:] = 0
    e.advance()
    assert e.reseeds == 1
    e.state["species"][:] = 0
    for _ in range(5):
        e.advance()
    assert e.reseeds == 1          # интервал ещё не вышел


def test_seeding_json_in_config():
    for name in ("ecology", "terra"):
        r = get_rules(name)
        cfg = r.Config(n=32)
        x, _c, _g = get_backend(False)
        state, _ = r.init_state(cfg, x)
        j = r.to_json(cfg, state)
        assert j["starters"] and "on" in j["reseed"]


def test_server_seeding_command():
    """Панель шлёт одну команду и на стартовый набор, и на повторный засев."""
    pytest.importorskip("aiohttp")
    from life_cube.viewers.web.server import WebViewer
    e = Engine(rules="ecology", cfg=get_rules("ecology").Config(n=32), rate=0,
               snapshot_every=0)
    v = WebViewer(e)
    v.handle({"cmd": "seeding",
              "value": {"start_species": [1], "reseed": True, "reseed_every": 7,
                        "reseed_count": 33, "reseed_on_extinction": False},
              "restart": True})
    assert e.cfg.start_species == (1,)
    assert e.cfg.reseed and e.cfg.reseed_every == 7 and e.cfg.reseed_count == 33
    present = set(np.unique(to_cpu(e.state["species"])).tolist()) - {0}
    assert present == {1}
    with pytest.raises(ValueError):
        v.handle({"cmd": "seeding", "value": {"start_species": []}})


def test_terra_plant_can_climb_out_of_water():
    """Стебель под водой сидит в тени, а вынырнув — получает полный свет:
    именно это делает рост вверх из озера осмысленным."""
    r = get_rules("terra")
    cfg = r.Config(n=48, start_species=(2,), sea_level=0.3, max_new_species=0)
    x, corr, _g = get_backend(False)
    state, _ = r.init_state(cfg, x)
    # сажаем «растение»: кормится в воде и тянется вверх
    from life_cube.engines.terra import GENES, IDX
    g = np.zeros(len(GENES), np.float32)
    g[IDX["light"]] = 1.0; g[IDX["water"]] = 1.0; g[IDX["up"]] = 0.9
    g[IDX["soil"]] = 0.8; g[IDX["metabolism"]] = 0.03; g[IDX["repro"]] = 1.0
    sid = state["free_ids"].pop(0)
    cfg.genomes[sid - 1] = g
    state["registry"][sid] = {"parent": 0, "born": 0, "died": None,
                              "genome": g.tolist(), "peak": 1, "changed": None}
    ground = to_cpu(state["stone_h"]) + to_cpu(state["soil_h"])
    water = to_cpu(state["water_h"])
    deep = np.argwhere(water > 2)
    assert len(deep), "нужен залитый мир"
    sp = to_cpu(state["species"])
    en = to_cpu(state["energy"])
    for (px, py) in deep[:200]:
        sp[px, py, ground[px, py]] = sid
        en[px, py, ground[px, py]] = 1.0
    state["species"] = x.asarray(sp)
    state["energy"] = x.asarray(en)
    state["genomes"] = x.asarray(cfg.genomes)
    for gen in range(160):
        r.step(state, cfg, x, corr, gen)
    sp = to_cpu(state["species"])
    ground2 = to_cpu(state["stone_h"]) + to_cpu(state["soil_h"])
    surface = ground2 + to_cpu(state["water_h"])
    cells = np.argwhere(sp == sid)
    assert len(cells) > 100, "растение не выжило в воде"
    above = cells[:, 2] > surface[cells[:, 0], cells[:, 1]]
    assert above.sum() > 0, "ни одна клетка не вынырнула из воды"
