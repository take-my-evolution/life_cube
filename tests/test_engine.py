import threading
import time

import numpy as np

from life_cube import Config
from life_cube.engine import Engine
from life_cube.viewers.web.server import encode_snapshot, decode_snapshot


def make_engine(rate=0.0, **kw):
    return Engine(Config(n=16, seed_density=0.05), rate=rate, **kw)


def test_engine_step_and_pause():
    e = make_engine()
    seen = []
    e.on_snapshot(lambda s: seen.append(s.gen))
    e.pause()
    th = threading.Thread(target=e.run, daemon=True); th.start()
    time.sleep(0.3)
    assert e.gen == 0                        # на паузе не шагает
    e.step_once(); time.sleep(0.3)
    assert e.gen == 1 and seen[-1] == 1      # одиночный шаг и снимок
    e.resume(); time.sleep(0.3)
    assert e.gen > 1
    e.stop(); th.join(2)
    assert not th.is_alive()


def test_engine_rate_limit():
    e = make_engine(rate=10.0)
    th = threading.Thread(target=e.run, daemon=True); th.start()
    time.sleep(1.0)
    e.stop(); th.join(2)
    assert 7 <= e.gen <= 13, e.gen           # ~10 пок/с
    assert 6 <= e.measured_rate <= 14


def test_engine_matches_plain_run():
    from life_cube import run
    cfg = Config(n=16, gens=12, seed_density=0.05)
    e = Engine(cfg, rate=0, components=False)
    e.run(max_gens=12)
    res = run(cfg, verbose=False)
    assert np.array_equal(e.state["species"], res["species"])
    assert np.array_equal(np.array(e.hist), res["hist"])


def test_engine_reset_changes_world():
    e = make_engine(components=False)
    e.run(max_gens=5)
    e.reset(Config(n=16, seed_density=0.05, seed_world=99))
    assert e.gen == 0 and e.hist == []
    assert e.last_snapshot.gen == 0


def test_protocol_roundtrip():
    e = make_engine()
    e.run(max_gens=3)
    snap = e.publish(force=True)
    buf = encode_snapshot(snap, first=True)
    hdr, coords, species, labels, soil = decode_snapshot(buf)
    assert hdr["gen"] == 3 and hdr["n"] == 16 and hdr["k"] == len(snap.coords)
    assert np.array_equal(coords, snap.coords) and np.array_equal(species, snap.species)
    assert np.array_equal(labels, snap.labels)
    assert len(hdr["relief"]) == 16 and hdr["species_names"][0] == "корка"
    assert len(hdr["components"]) == len(snap.components)
    # во втором кадре рельефа нет — экономим
    assert "relief" not in decode_snapshot(encode_snapshot(snap))[0]
