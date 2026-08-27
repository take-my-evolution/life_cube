"""Ответвление вида (форк) — правка генома, не стирающая эволюцию."""

import numpy as np
import pytest

from life_cube.backend import to_cpu
from life_cube.engine import Engine
from life_cube.engines import get_rules


def _terra(gens=60, n=48):
    e = Engine(cfg=get_rules("terra").Config(n=n), rules="terra", rate=0, snapshot_every=0)
    for _ in range(gens):
        e.advance()
    return e


def test_fork_creates_child_and_keeps_parent():
    e = _terra()
    sid = sorted(e.state["registry"])[0]
    before = int((to_cpu(e.state["species"]) == sid).sum())
    g = np.asarray(e.cfg.genomes[sid - 1]).copy()
    g[3] = 1.0
    new, k = e.fork_species(sid, g, share=0.4)
    sp = to_cpu(e.state["species"])
    assert new != sid and k > 0
    assert e.state["registry"][new]["parent"] == sid
    assert e.state["registry"][new]["born"] == e.gen
    assert int((sp == new).sum()) == k                 # потомок реально живёт
    assert 0 < int((sp == sid).sum()) < before         # родитель жив, но поредел
    assert float(e.cfg.genomes[new - 1][3]) == pytest.approx(1.0)
    assert float(e.cfg.genomes[sid - 1][3]) != pytest.approx(1.0)


def test_fork_child_survives_a_few_generations():
    """Потомок с нулевым населением был бы съеден сборщиком видов сразу же."""
    e = _terra()
    sid = sorted(e.state["registry"])[0]
    new, _k = e.fork_species(sid, np.asarray(e.cfg.genomes[sid - 1]).copy(), share=0.5)
    for _ in range(5):
        e.advance()
    assert new in e.state["registry"]


def test_fork_rejects_nonsense():
    e = _terra()
    sid = sorted(e.state["registry"])[0]
    with pytest.raises(ValueError):
        e.fork_species(999, np.zeros(12, np.float32))          # такого вида нет
    with pytest.raises(ValueError):
        e.fork_species(sid, np.zeros(3, np.float32))           # геном не той длины


def test_fork_refused_when_no_free_ids():
    e = _terra()
    sid = sorted(e.state["registry"])[0]
    e.state["free_ids"] = []
    with pytest.raises(ValueError):
        e.fork_species(sid, np.asarray(e.cfg.genomes[sid - 1]).copy())


def test_ecology_cannot_fork():
    e = Engine(cfg=get_rules("ecology").Config(n=32), rules="ecology", rate=0,
               snapshot_every=0)
    assert get_rules("ecology").can_fork is False
    with pytest.raises(ValueError):
        e.fork_species(1, np.zeros(14, np.float32))


def test_lichen_can_fork():
    e = Engine(cfg=get_rules("lichen").Config(n=48), rules="lichen", rate=0,
               snapshot_every=0)
    for _ in range(40):
        e.advance()
    sid = sorted(e.state["registry"])[0]
    new, k = e.fork_species(sid, np.asarray(e.cfg.genomes[sid - 1]).copy(), share=0.3)
    assert k > 0 and e.state["registry"][new]["parent"] == sid


def test_server_fork_command():
    pytest.importorskip("aiohttp")
    from life_cube.viewers.web.server import WebViewer
    e = _terra()
    v = WebViewer(e)
    sid = sorted(e.state["registry"])[0]
    g = np.asarray(e.cfg.genomes[sid - 1]).tolist()
    out = v.handle({"cmd": "fork", "id": sid, "value": g, "share": 0.25})
    assert out["forked"] != sid and out["cells"] > 0
    j = v._config_json()
    assert j["can_fork"] is True and out["forked"] in j["ids"]
    assert j["gene_docs"] and all(len(v) > 20 for v in j["gene_docs"].values())
