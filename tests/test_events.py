"""Поток событий: Engine считает диф species между поколениями, копит его до
кадра, кадр несёт события и номера организмов по проводу.

Это будущий протокол между модулями «Куб» и «Голос»: звук делает зёрна из
событий, а не из случайных клеток, поэтому события обязаны быть полными
(каждое изменение), с правильным типом и с координатами той самой клетки.
"""

import numpy as np

from life_cube.engine import Engine
from life_cube.engines import get_rules
from life_cube.viewers.web.server import decode_snapshot, encode_snapshot


def make(n=20, variant=1):
    R = get_rules("iron")
    cfg = R.make_config(n=n, variant=variant)
    return Engine(cfg, rules="iron", rate=0, components=True)


def test_events_equal_species_diff():
    e = make()
    e.publish(force=True)                       # опустошить очередь
    before = np.asarray(e.state["species"]).copy()
    e.advance()
    after = np.asarray(e.state["species"]).copy()
    snap = e.publish(force=True)
    ev = snap.events
    assert snap.event_gens == 1
    births = (before == 0) & (after > 0)
    deaths = (before > 0) & (after == 0)
    change = (before > 0) & (after > 0) & (before != after)
    assert len(ev) == int(births.sum() + deaths.sum() + change.sum())
    got = np.zeros_like(before, dtype=np.uint8)
    got[ev[:, 1], ev[:, 2], ev[:, 3]] = ev[:, 0]
    assert np.array_equal(got == Engine.EV_BIRTH, births)
    assert np.array_equal(got == Engine.EV_DEATH, deaths)
    assert np.array_equal(got == Engine.EV_CHANGE, change)
    # вид: у рождения — новый, у гибели — старый
    for t, x, y, z, sp in ev:
        assert sp == (after[x, y, z] if t != Engine.EV_DEATH else before[x, y, z])


def test_events_accumulate_between_frames_and_drain():
    e = make()
    e.publish(force=True)
    for _ in range(5):
        e.advance()
    snap = e.publish(force=True)
    assert snap.event_gens == 5 and len(snap.events) > 0
    again = e.publish(force=True)
    assert snap.event_gens != 0 and again.event_gens == 0 and len(again.events) == 0


def test_events_carry_organism_ids_and_mobile_species():
    e = make()
    for _ in range(8):
        e.advance()
    e.publish(force=True, components=True)      # разметка → id организмов
    for _ in range(3):
        e.advance()
    snap = e.publish(force=True)
    assert len(snap.event_orgs) == len(snap.events)
    # гибель клетки известного организма несёт его номер
    deaths = snap.events[:, 0] == Engine.EV_DEATH
    assert deaths.any() and (snap.event_orgs[deaths] > 0).any()
    assert snap.mobile == [4, 5]


def test_events_survive_the_wire():
    e = make()
    for _ in range(4):
        e.advance()
    snap = e.publish(force=True)
    header, *_ = decode_snapshot(encode_snapshot(snap, first=True))
    assert header["e"] == len(snap.events) and header["e_gens"] == snap.event_gens
    assert np.array_equal(header["events"], snap.events)
    assert np.array_equal(header["event_orgs"], snap.event_orgs)
    assert header["mobile"] == [4, 5]


def test_events_are_capped_not_unbounded():
    e = make(n=16, variant=2)                   # волны: тысячи перекрасок за поколение
    e.publish(force=True)
    for _ in range(40):
        e.advance()
    snap = e.publish(force=True)
    assert 0 < len(snap.events) <= Engine.EVENTS_KEEP + Engine.EVENTS_PER_STEP


def test_other_engines_also_stream_events():
    cfg = get_rules("slope").make_config(n=24)
    e = Engine(cfg, rules="slope", rate=0, components=True)
    e.publish(force=True)
    for _ in range(3):
        e.advance()
    snap = e.publish(force=True)
    assert len(snap.events) > 0
    assert snap.mobile == [4, 5]
