"""Скриншот-тесты клиента: headless Chromium через Playwright.

Сервер поднимается на случайном порту с крошечным миром на паузе; клиент
опрашивается через window.viewer — считаем пиксели по цветам видов.
Пропускаются, если playwright/aiohttp не установлены.
"""

import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("aiohttp")
pw = pytest.importorskip("playwright.sync_api")

from life_cube import Config                       # noqa: E402
from life_cube.engine import Engine                # noqa: E402
from life_cube.viewers.web.server import WebViewer  # noqa: E402


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope="module")
def server():
    from aiohttp import web
    cfg = Config(n=24, seed_density=0.05)
    engine = Engine(cfg, rate=0, components=True)
    engine.run(max_gens=25)          # немного жизни, потом на паузу
    engine.pause()
    viewer = WebViewer(engine)
    port = _free_port()
    runner = web.AppRunner(viewer.make_app())

    import asyncio
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
                                          "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 900, "height": 600})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(server["url"])
        page.wait_for_function("window.viewer && window.viewer.S.connected", timeout=15000)
        page.evaluate("viewer.draw()")
        assert not errors, errors
        yield page
        browser.close()


def hist(page):
    return page.evaluate("viewer.histogram()")


def test_page_loads_and_shows_life(page, server):
    S = page.evaluate("({gen: viewer.S.gen, k: viewer.S.k, n: viewer.S.n, comps: viewer.S.comps.length})")
    assert S["gen"] == 25 and S["n"] == 24 and S["k"] > 0 and S["comps"] > 0
    h = hist(page)
    live = h["s1"] + h["s2"] + h["s3"] + h["s4"] + h["s5"]
    assert live > 200, h
    assert h["stone"] > 200, h


def test_species_filter(page, server):
    """Оставляем один вид — на экране остаются пиксели только его цвета."""
    pops = server["engine"].last_snapshot.pops
    s = int(np.argmax(pops)) + 1                     # самый многочисленный вид
    mask = [1 if i == s - 1 else 0 for i in range(8)]
    page.evaluate(f"viewer.set('ghost', false); viewer.set('showStone', false);"
                  f" viewer.set('showSoil', false); viewer.set('speciesMask', {mask})")
    h = hist(page)
    mine = h[f"s{s}"] if f"s{s}" in h else 0
    others = sum(v for k, v in h.items() if k.startswith("s") and k != f"s{s}")
    assert mine > 100, h
    assert others < mine * 0.15, h
    page.evaluate("viewer.set('speciesMask', [1,1,1,1,1,1,1,1]); viewer.set('showStone', true); viewer.set('showSoil', true)")


def test_clip_z_removes_upper_cells(page, server):
    page.evaluate("viewer.set('ghost', false); viewer.set('showStone', false); viewer.setView('front')")
    full = hist(page)
    sp = server["engine"].state["species"]
    zs = np.argwhere(sp > 0)[:, 2]
    mid = int((zs.min() + zs.max()) // 2)
    page.evaluate(f"viewer.set('clipZ', {mid})")
    cut = hist(page)
    page.evaluate("viewer.set('clipZ', 1e9); viewer.set('showStone', true)")
    live = lambda h: h["s1"] + h["s2"] + h["s3"] + h["s4"] + h["s5"]
    assert 0 < live(cut) < live(full), (full, cut)


def test_top_view_scales_with_population(page, server):
    """Вид сверху ортографикой: сколько пикселей закрашено — пропорционально
    числу столбцов с жизнью (ловит перепутанные оси и битую проекцию)."""
    page.evaluate("viewer.set('proj','ortho'); viewer.set('ghost', false); viewer.set('showStone', false); viewer.set('showSoil', false); viewer.setView('top')")
    h = hist(page)
    sp = server["engine"].state["species"]
    cols = int((sp > 0).any(axis=2).sum())
    live = sum(v for k, v in h.items() if k.startswith("s"))
    px_per_cell = live / max(cols, 1)
    assert 20 < px_per_cell < 4000, (live, cols)
    # оставим один вид — площадь должна упасть примерно во столько же раз
    pops = server["engine"].last_snapshot.pops
    s = int(np.argmax(pops)) + 1
    cols1 = int((sp == s).any(axis=2).sum())
    mask = [1 if i == s - 1 else 0 for i in range(8)]
    page.evaluate(f"viewer.set('speciesMask',{mask})")
    live1 = sum(v for k, v in hist(page).items() if k.startswith("s"))
    assert live1 < live and live1 / max(live, 1) < min(1.0, 2.5 * cols1 / max(cols, 1))
    page.evaluate("viewer.set('speciesMask',[1,1,1,1,1,1,1,1]); viewer.set('proj','persp'); viewer.setView('iso'); viewer.set('showStone', true)")


def test_select_organism(page, server):
    comps = server["engine"].last_snapshot.components
    big = comps[0]
    page.evaluate(f"viewer.set('ghost', false); viewer.set('showStone', false); viewer.set('selected', {big.cid})")
    h = hist(page)
    key = f"s{big.species}"
    others = sum(h[f"s{s}"] for s in (1, 2, 3, 4, 5) if s != big.species)
    assert h[key] > 0 and others < 30, h
    page.evaluate("viewer.set('selected', 0); viewer.set('showStone', true)")


def test_controls_reach_engine(page, server):
    e = server["engine"]
    g0 = e.gen
    page.click("#btnStep")
    time.sleep(0.5)
    assert e.gen == g0 + 1
    page.wait_for_function(f"viewer.S.gen == {g0 + 1}", timeout=5000)
    page.click("#btnPause")          # -> resume
    time.sleep(0.4)
    assert e.gen > g0 + 1
    page.click("#btnPause")          # -> pause
    time.sleep(0.3)
    g = e.gen; time.sleep(0.3); assert e.gen == g


def test_sound_frame_reaches_client_and_audio_graph(page, server):
    gen = server["engine"].gen
    page.wait_for_function(f"viewer.S.gen == {gen}", timeout=10000)
    sf = page.evaluate("viewer.S.sound")
    assert sf and len(sf["harmonics"]) == 64 and sf["gen"] == gen
    assert max(sf["harmonics"]) > 0
    # включаем звук (в headless контекст может быть suspended — граф всё равно строится)
    page.click("#btnAudio")
    st = page.evaluate("({on: viewer.Audio.on, harm: viewer.Audio.harm.length, voices: viewer.Audio.voices.size, state: viewer.Audio.ctx.state})")
    assert st["on"] and st["harm"] == 64 and st["voices"] == len(sf["voices"])
    # гармоники с населением получили ненулевой gain-target, пустые — нулевой
    gains = page.evaluate("viewer.Audio.harm.map(x => x.g.gain.value)")
    live = [i for i, a in enumerate(sf["harmonics"]) if a > 0]
    dead = [i for i, a in enumerate(sf["harmonics"]) if a == 0]
    page.wait_for_timeout(600)
    gains = page.evaluate("viewer.Audio.harm.map(x => x.g.gain.value)")
    if page.evaluate("viewer.Audio.ctx.state") == "running":
        assert all(gains[i] > 0 for i in live) and all(gains[i] == 0 for i in dead)
    # водопад что-то нарисовал
    px = page.evaluate("(() => { const c=viewer.WF.ctx.getImageData(0,0,240,64).data; let s=0; for (let i=0;i<c.length;i+=4) s+=c[i+1]; return s; })()")
    assert px > 0
    page.click("#btnAudio")
    assert not page.evaluate("viewer.Audio.on")


def test_constructor_panel_edits_genomes(page, server):
    """Панель конструктора приходит с конфигом и правит геномы на сервере."""
    cfg = page.evaluate("viewer.CFG.data")
    assert cfg and len(cfg["fields"]) == 14 and len(cfg["genomes"]) == len(cfg["names"])
    assert "травоядное" in cfg["names"] and "хищник" in cfg["names"]
    # ползунки нарисованы для выбранного вида
    assert page.locator("#genes input[type=range]").count() == 14
    assert page.locator("#speciesPick button").count() == len(cfg["names"])
    # правим ген через API панели и применяем
    gi = cfg["fields"].index("metabolism")
    page.evaluate(f"viewer.CFG.edited[6][{gi}] = 0.123")
    page.click("#btnApplyGenes")
    page.wait_for_timeout(600)
    assert abs(float(server["engine"].cfg.genomes[6][gi]) - 0.123) < 1e-4
    # сервер разослал новый конфиг обратно
    page.wait_for_function(f"Math.abs(viewer.CFG.data.genomes[6][{gi}] - 0.123) < 1e-4", timeout=5000)


def test_world_panel_reseeds(page, server):
    e = server["engine"]
    page.evaluate("viewer.CFG.world.seed_world = 12345; viewer.CFG.world.stone_fraction = 0.5")
    page.click("#btnApplyWorld")
    page.wait_for_timeout(1200)
    assert e.cfg.seed_world == 12345
    assert abs(e.cfg.stone_fraction - 0.5) < 1e-6
    assert e.gen <= 3          # мир пересоздан, счётчик сброшен
    stone_share = float((e.state["stone"]).mean())
    assert 0.4 < stone_share < 0.6, stone_share


def test_server_stays_alive_during_heavy_reseed(page, server):
    """Регрессия на зависание: пересев большого мира не должен блокировать
    сервер. Раньше команда исполнялась в цикле событий, и страница умирала."""
    import time as _t
    import urllib.request
    url = server["url"]
    page.evaluate("viewer.CFG.world.n = 128; viewer.CFG.world.seed_density = 0.02")
    page.click("#btnApplyWorld")
    # сразу после команды сервер обязан отвечать по HTTP
    ok, t0 = 0, _t.time()
    for _ in range(6):
        started = _t.time()
        with urllib.request.urlopen(url, timeout=3) as r:
            assert r.status == 200
        ok += 1
        assert _t.time() - started < 3
        _t.sleep(0.2)
    assert ok == 6
    page.wait_for_function("viewer.S.n == 128", timeout=25000)
    assert server["engine"].cfg.n == 128


def test_oversized_world_is_refused_with_message(page, server):
    n_before = server["engine"].cfg.n
    page.evaluate("viewer.CFG.world.n = 4096")
    page.click("#btnApplyWorld")
    page.wait_for_function("document.getElementById('status').textContent.includes('ошибка')", timeout=8000)
    assert server["engine"].cfg.n == n_before
    txt = page.inner_text("#status")
    assert "предел" in txt


def test_engine_switch_from_browser(page, server):
    """Переключение движка из выпадающего списка: новый мир, новые гены в панели,
    имена видов приходят каждый кадр (динамические виды)."""
    e = server["engine"]
    page.wait_for_function("viewer.S.engines && viewer.S.engines.length >= 2", timeout=5000)
    page.evaluate("viewer.CFG.world.n = 32")          # предыдущие тесты могли раздуть мир
    page.select_option("#engineSel", "lichen")
    page.wait_for_function("viewer.CFG.data && viewer.CFG.data.engine === 'lichen'", timeout=20000)
    assert e.rules.name == "lichen"
    cfg = page.evaluate("viewer.CFG.data")
    assert "substrate" in cfg["fields"] and cfg["ids"] == [1]
    # пара шагов — и кадр с именами видов и рельефом
    page.click("#btnStep"); page.click("#btnStep")
    page.wait_for_function("viewer.S.gen >= 2 && viewer.S.names && viewer.S.names[0] === 'лишайник'", timeout=10000)
    assert page.evaluate("viewer.S.relief !== null")
    # обратно на экологию
    page.evaluate("viewer.CFG.world.n = 32")
    page.select_option("#engineSel", "ecology")
    page.wait_for_function("viewer.CFG.data && viewer.CFG.data.engine === 'ecology'", timeout=20000)
    assert e.rules.name == "ecology"


def test_randomize_and_restart_buttons(page, server):
    e = server["engine"]
    g_before = e.cfg.genomes.copy()
    page.click("#btnRandomRestart")
    page.wait_for_function("viewer.S.gen <= 1", timeout=10000)
    page.wait_for_timeout(500)
    assert not np.array_equal(e.cfg.genomes, g_before)     # гены поменялись
    assert e.gen <= 3                                       # мир пересоздан
    # «Перезапуск» — те же гены, мир с нуля
    page.click("#btnStep"); page.wait_for_timeout(300)
    g2 = e.cfg.genomes.copy()
    page.click("#btnRestart"); page.wait_for_timeout(800)
    assert np.array_equal(e.cfg.genomes, g2) and e.gen <= 2
