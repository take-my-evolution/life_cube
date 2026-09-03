"""Звук «Организмы» (сэмпл = организм, зерно = событие) — headless Chromium.

Урок v0.8.2: тесты на узлы не ловят «работает, но не слышно». Поэтому первым
делом — замер реального сигнала на шине через AnalyserNode. Дальше: таблицы
детерминированы и различают организмы; события действительно превращаются в
зёрна; ковёр режется на плитки, а компактный организм остаётся собой.
"""

import asyncio
import socket
import threading

import pytest

pytest.importorskip("aiohttp")
pw = pytest.importorskip("playwright.sync_api")

from life_cube.engine import Engine                # noqa: E402
from life_cube.engines import get_rules            # noqa: E402
from life_cube.viewers.web.server import WebViewer  # noqa: E402


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def server():
    from aiohttp import web
    cfg = get_rules("iron").make_config(n=24, variant=1)
    engine = Engine(cfg, rules="iron", rate=15, components=True)
    engine.run(max_gens=40)
    viewer = WebViewer(engine, fps=15, components_hz=3)
    port = _free_port()
    runner = web.AppRunner(viewer.make_app())
    loop = asyncio.new_event_loop()

    async def start():
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
    loop.run_until_complete(start())
    th = threading.Thread(target=loop.run_forever, daemon=True); th.start()
    sim = threading.Thread(target=engine.run, daemon=True); sim.start()
    yield {"url": f"http://127.0.0.1:{port}/", "engine": engine}
    engine.stop()
    loop.call_soon_threadsafe(loop.stop)


@pytest.fixture(scope="module")
def page(server):
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader",
                                          "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 900, "height": 600})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server["url"])
        page.wait_for_function("window.viewer && window.viewer.S.connected", timeout=15000)
        assert not errors, errors
        yield page
        browser.close()


def _audio_on(page):
    if not page.evaluate("viewer.Audio.on"):
        page.click("#btnAudio")
    if page.evaluate("viewer.Audio.ctx.state") != "running":
        pytest.skip("AudioContext не запустился в headless — мерить нечего")


def test_frames_carry_events_and_mobile_species(page):
    page.wait_for_function("viewer.S.events && viewer.S.events.length > 0", timeout=10000)
    ev = page.evaluate("Array.from(viewer.S.events.slice(0, 10))")
    assert len(ev) == 10 and all(0 < t <= 3 for t in ev[0::5])
    assert page.evaluate("viewer.S.mobile") == [4, 5]


def test_org_bus_is_actually_audible(page):
    _audio_on(page)
    page.check("#orgOn")
    page.wait_for_function("viewer.Audio.Org.ready", timeout=10000)
    page.evaluate("viewer.Audio.busH.gain.value=0; viewer.Audio.busV.gain.value=0; viewer.Audio.busN.gain.value=0")
    page.evaluate("""() => {
        const A = viewer.Audio; clearInterval(window.__poll);
        const an = A.ctx.createAnalyser(); an.fftSize = 2048; A.busO.connect(an); window.__an = an; window.__peak = 0;
        window.__poll = setInterval(() => { const b = new Float32Array(2048); an.getFloatTimeDomainData(b);
            let m = 0; for (let i=0;i<b.length;i++) m = Math.max(m, Math.abs(b[i])); if (m > window.__peak) window.__peak = m; }, 20);
    }""")
    page.wait_for_timeout(2500)
    peak = page.evaluate("window.__peak")
    sent = page.evaluate("viewer.Audio.Org.sent")
    stats = page.evaluate("viewer.Audio.Org.stats")
    assert sent > 20, sent
    assert stats["played"] > 0, stats
    assert peak > 0.08, (peak, sent, stats)
    # много разных голосов, а не пять
    assert page.evaluate("viewer.Audio.Org.tables.size") >= 8
    page.evaluate("clearInterval(window.__poll)")


def test_tables_are_deterministic_and_differ_between_organisms(page):
    js = """(args) => { const O = viewer.Audio.Org; const t = O.buildTables(args.key, args.sp, args.st);
        let da=0, db=0; for (let i=0;i<1024;i++){ da += t.a[i]*t.a[i]; db += t.b[i]*t.b[i]; }
        return {a: Array.from(t.a.slice(0, 64)), b: Array.from(t.b.slice(0, 64)), ra: Math.sqrt(da/1024), rb: Math.sqrt(db/1024)}; }"""
    young = page.evaluate(js, {"key": "c7", "sp": 2, "st": {"size": 10, "age": 0, "cx": 5, "cy": 5, "cz": 3}})
    again = page.evaluate(js, {"key": "c7", "sp": 2, "st": {"size": 10, "age": 0, "cx": 5, "cy": 5, "cz": 3}})
    other = page.evaluate(js, {"key": "c8", "sp": 2, "st": {"size": 10, "age": 0, "cx": 5, "cy": 5, "cz": 3}})
    old = page.evaluate(js, {"key": "c7", "sp": 2, "st": {"size": 10, "age": 300, "cx": 5, "cy": 5, "cz": 3}})
    tree = page.evaluate(js, {"key": "c7", "sp": 3, "st": {"size": 10, "age": 0, "cx": 5, "cy": 5, "cz": 3}})
    assert young == again                          # тот же организм — та же таблица
    assert young["a"] != other["a"]                # другой организм того же вида — другой голос
    assert young["a"] != old["a"]                  # состарился — таблица изменилась
    assert young["a"] != tree["a"]                 # другой вид — другой тембр
    assert young["a"] != young["b"]                # начало и конец «сэмпла» различаются
    for t in (young, other, old, tree):
        assert 0.2 < t["ra"] < 0.3 and 0.2 < t["rb"] < 0.3, t   # нормировка по RMS


def test_events_become_grains_and_step_differs_from_birth(page):
    _audio_on(page)
    page.check("#orgOn")
    page.wait_for_function("viewer.Audio.Org.ready", timeout=10000)
    res = page.evaluate("""() => {
        const O = viewer.Audio.Org; const msgs = [];
        const orig = O.node.port.postMessage.bind(O.node.port);
        O.node.port.postMessage = (m) => { msgs.push(m); };
        const n = viewer.S.n;
        // рождение травы, гибель мха, шаг травоядного (рождение подвижного вида)
        const ev = new Uint16Array([1, 3, 4, 5, 2,   2, 10, 11, 4, 1,   1, 20, 2, 6, 4]);
        const orgs = new Uint32Array([0, 0, 0]);
        const made = O.onEvents(ev, orgs, 1, null, 0.1);
        O.node.port.postMessage = orig;
        const grains = msgs.filter(m => m.t === 'grain');
        return {made, grains: grains.map(g => ({key: g.key, dur: g.dur, hz: g.hz, pan: g.pan, pos: g.pos, bright: g.bright})),
                tables: msgs.filter(m => m.t === 'table').length};
    }""")
    assert res["made"] == 3 and len(res["grains"]) == 3
    birth, death, step = res["grains"]
    assert birth["key"].startswith("t2_") and death["key"].startswith("t1_") and step["key"].startswith("t4_")
    assert step["dur"] < birth["dur"] < death["dur"]       # щелчок шага < вспышка рождения < тёмный хвост гибели
    assert death["bright"] < birth["bright"]
    assert abs(birth["pan"] - (2 * 3 / 24 - 1)) < 1e-6 and abs(birth["pos"] - 4 / 24) < 1e-6   # x → панорама, y → точка чтения


def test_carpet_splits_into_tiles_but_compact_organism_keeps_identity(page):
    keys = page.evaluate("""() => { const O = viewer.Audio.Org; const big = new Set([99]);
        return [O.key(2, 3, 4, 7, big), O.key(2, 30, 4, 7, big), O.key(2, 3, 4, 99, big), O.key(2, 12, 4, 99, big), O.key(2, 5, 5, 0, big)]; }""")
    assert keys[0] == keys[1] == "c7"              # компактный организм — один голос, где бы ни была клетка
    assert keys[2] == "t2_0_0" and keys[3] == "t2_1_0"   # ковёр — по плиткам
    assert keys[4] == "t2_0_0"                     # неизвестно чей — тоже по плитке


def test_switching_off_silences_and_frees_tables(page):
    _audio_on(page)
    page.check("#orgOn")
    page.wait_for_function("viewer.Audio.Org.ready && viewer.Audio.Org.tables.size > 0", timeout=10000)
    page.uncheck("#orgOn")
    assert page.evaluate("viewer.Audio.Org.tables.size") == 0
    before = page.evaluate("viewer.Audio.Org.sent")
    page.wait_for_timeout(600)
    assert page.evaluate("viewer.Audio.Org.sent") == before
