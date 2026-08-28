"""Перкуссия: дискретные события поколения (охота, гибель, рождение...) на
пути от симуляции до клиента. Три слоя, каждый со своим риском регрессии:

  1. backend.sample_event — дёшево описывает булев массив событием;
  2. step.py/motion.py — где эти события на самом деле возникают;
  3. Engine.publish — раздаёт события РОВНО один раз на поколение;
  4. WebViewer._merge_events/broadcaster — копит события МЕЖДУ фактическими
     отправками кадра (симуляция может тикать быстрее fps) и раздаёт
     накопленное строго в момент реальной отправки.
"""

import numpy as np
import pytest

from life_cube import Config
from life_cube.backend import get_backend, sample_event
from life_cube.config import GENOME_FIELDS, IDX
from life_cube.engine import Engine
from life_cube.sim import init_state
from life_cube.step import step
from life_cube.viewers.web.server import WebViewer


# --------------------------------------------------------------------------
# 1. sample_event
# --------------------------------------------------------------------------

def test_sample_event_none_when_nothing_happened():
    xp, _, _ = get_backend(False)
    mask = xp.zeros((8, 8, 8), dtype=bool)
    assert sample_event(mask, xp) is None


def test_sample_event_counts_and_caps_positions():
    xp, _, _ = get_backend(False)
    mask = xp.zeros((8, 8, 8), dtype=bool)
    mask[0, 0, 0] = mask[1, 0, 0] = mask[2, 0, 0] = mask[3, 0, 0] = True
    ev = sample_event(mask, xp, cap=2)
    assert ev["n"] == 4                 # полное число клеток, не обрезанное
    assert len(ev["x"]) == 2            # но позиций — не больше cap
    assert set(ev["x"]) <= {0.0, 1.0, 2.0, 3.0}


# --------------------------------------------------------------------------
# 2. где события возникают: step.py / motion.py
# --------------------------------------------------------------------------

def genome(**kw):
    g = np.zeros(len(GENOME_FIELDS), np.float32)
    for k, v in kw.items():
        g[IDX[k]] = v
    return g


GRASS = genome(absorb=0.3, up=1.0, birth=1.6, need=0.3, water=0.4, branch=0.2)
HERB = genome(hunt=0.9, trophic=1, speed=1, sense=4, metabolism=0.05, repro=4.0,
              lifespan=200)
PRED = genome(hunt=0.9, trophic=2, speed=2, sense=6, metabolism=0.05, repro=5.0,
              lifespan=300)


def plant_world(n=16, p_shock=0.0, p_dissolve=0.0):
    cfg = Config(n=n, genomes=np.array([GRASS], np.float32), p_shock=p_shock,
                 p_dissolve=p_dissolve, p_mutate=0.0, seed_density=0.05,
                 animal_share=0.0)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    stone = np.zeros((n, n, n), bool); stone[:, :, :3] = True
    state["stone"] = stone
    state["soil"] = np.zeros_like(stone)
    state["wet"] = np.ones((n, n), np.float32)
    state["species"] = np.zeros((n, n, n), np.int8)
    return state, cfg, xp, corr


def animal_world(genomes, n=16, **kw):
    cfg = Config(n=n, genomes=np.array(genomes, np.float32), p_shock=0.0,
                 p_dissolve=0.0, p_mutate=0.0, seed_density=0.05,
                 animal_share=0.0, **kw)
    xp, corr, _ = get_backend(False)
    state, _ = init_state(cfg, xp)
    stone = np.zeros((n, n, n), bool); stone[:, :, :3] = True
    state["stone"] = stone
    state["soil"] = np.zeros_like(stone)
    state["wet"] = np.ones((n, n), np.float32)
    state["species"] = np.zeros((n, n, n), np.int8)
    state["energy"] = np.zeros((n, n, n), np.float32)
    state["age"] = np.zeros((n, n, n), np.int32)
    return state, cfg, xp, corr


def put(state, x, y, z, s, energy=5.0):
    state["species"][x, y, z] = s
    state["energy"][x, y, z] = energy


def test_step_always_sets_last_events_even_when_empty():
    """Контракт с Engine.advance(): state['last_events'] должен появляться
    после КАЖДОГО step(), иначе last_events_gen никогда не проставится
    (см. Engine.advance: `if "last_events" in self.state`)."""
    state, cfg, xp, corr = plant_world(n=8)      # пустой мир, расти нечему
    step(state, cfg, xp, corr)
    assert "last_events" in state
    assert state["last_events"] == {}            # ничего не произошло


def test_step_reports_dissolve_event():
    state, cfg, xp, corr = plant_world(n=10, p_dissolve=1.0)
    state["stone"][5, 5, 2] = True                # камень под будущим ростком
    put_plant = state["species"]
    put_plant[5, 5, 3] = 1                         # росток стоит на камне
    step(state, cfg, xp, corr)
    ev = state["last_events"].get("dissolve")
    assert ev is not None and ev["n"] >= 1


def test_step_reports_shock_event():
    state, cfg, xp, corr = plant_world(n=10, p_shock=1.0)
    state["species"][5, 5, 3] = 1
    state["stone"][5, 5, 2] = True
    step(state, cfg, xp, corr)
    ev = state["last_events"].get("shock")
    assert ev is not None and ev["n"] == 1


def test_step_reports_birth_plant_event():
    state, cfg, xp, corr = plant_world(n=12)
    for x in range(4, 8):                          # заросли, чтобы точно родилось
        for y in range(4, 8):
            put(state, x, y, 3, 1, 0.9)
    for _ in range(6):
        step(state, cfg, xp, corr)
        if "birth_plant" in state["last_events"]:
            break
    assert "birth_plant" in state["last_events"]


def test_animals_step_reports_kill_event():
    state, cfg, xp, corr = animal_world([GRASS, HERB, PRED], n=14)
    put(state, 7, 7, 3, 2, 5.0)
    put(state, 8, 7, 3, 3, 5.0)
    killed = False
    for _ in range(15):
        step(state, cfg, xp, corr)
        ev = state["last_events"].get("kill")
        if ev:
            killed = True
            assert ev["n"] >= 1
            break
    assert killed


def test_animals_step_reports_starve_event():
    state, cfg, xp, corr = animal_world([GRASS, PRED], n=10)
    put(state, 5, 5, 3, 2, 1.0)                    # хищник без добычи вокруг
    starved = False
    for _ in range(20):
        step(state, cfg, xp, corr)
        ev = state["last_events"].get("starve")
        if ev:
            starved = True
            assert ev["n"] == 1
            break
    assert starved
    assert int((state["species"] == 2).sum()) == 0


def test_animals_step_reports_birth_animal_event():
    breeder = genome(hunt=0.0, trophic=1, speed=1, sense=1, metabolism=0.0, repro=4.0)
    state, cfg, xp, corr = animal_world([GRASS, breeder], n=10)
    put(state, 5, 5, 3, 2, 10.0)
    step(state, cfg, xp, corr)
    ev = state["last_events"].get("birth_animal")
    assert ev is not None and ev["n"] == 1


# --------------------------------------------------------------------------
# 3. Engine.publish: события отдаются РОВНО один раз на поколение
# --------------------------------------------------------------------------

def make_shock_engine(n=10):
    cfg = Config(n=n, genomes=np.array([GRASS], np.float32), p_shock=1.0,
                 p_dissolve=0.0, p_mutate=0.0, seed_density=0.05, animal_share=0.0)
    e = Engine(cfg, rate=0, components=False)
    e.state["species"][:] = 0                    # только одна известная клетка
    e.state["stone"][:] = False
    e.state["stone"][:, :, :3] = True
    e.state["species"][5, 5, 3] = 1
    e.state["stone"][5, 5, 2] = True
    return e


def test_publish_pops_events_exactly_once_per_generation():
    e = make_shock_engine()
    e.advance()
    snap1 = e.publish(force=True)
    snap2 = e.publish(force=True)                 # тот же gen, вторичный такт
    assert snap1.events is not None and snap1.events.get("shock")
    assert snap2.events is None                   # уже забрано snap1'ом


def test_publish_events_advance_to_new_generation():
    e = make_shock_engine()
    e.advance()
    snap1 = e.publish(force=True)
    e.advance()
    snap2 = e.publish(force=True)
    assert snap1.events is not None
    assert snap2.events is not None                # новое поколение — новые события


# --------------------------------------------------------------------------
# 4. WebViewer: копится между отправками, раздаётся строго на отправке
# --------------------------------------------------------------------------

def test_merge_events_accumulates_counts_and_caps_pan():
    e = make_shock_engine(n=10)
    viewer = WebViewer(e)
    viewer._merge_events({"shock": {"n": 3, "x": [1.0, 2.0, 3.0]}})
    viewer._merge_events({"shock": {"n": 2, "x": [4.0]}})
    acc = viewer._events_acc["shock"]
    assert acc["n"] == 5
    assert len(acc["pan"]) == 4
    assert all(-1.0 <= p <= 1.0 for p in acc["pan"])
    # копится и не обрезается раньше 32
    for _ in range(20):
        viewer._merge_events({"shock": {"n": 1, "x": [0.0]}})
    assert len(viewer._events_acc["shock"]["pan"]) <= 32


def test_merge_events_keeps_separate_kinds():
    e = make_shock_engine(n=10)
    viewer = WebViewer(e)
    viewer._merge_events({"kill": {"n": 1, "x": [0.0]}, "shock": {"n": 1, "x": [1.0]}})
    assert set(viewer._events_acc.keys()) == {"kill", "shock"}


def test_on_snapshot_merges_snapshot_events():
    """_on_snapshot должен подхватывать snap.events, если Engine их выставил
    (а не только через publish()/broadcaster, что и тестируют другие кейсы)."""
    e = make_shock_engine(n=10)
    viewer = WebViewer(e)
    e.advance()
    snap = e.publish(force=True)
    assert snap.events and snap.events.get("shock")
    viewer._on_snapshot(snap)
    assert "shock" in viewer._events_acc
