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
    # rate=60, а не 0 (без предела): test_controls_reach_engine снимает паузу
    # на 0.4 с, и на неограниченной скорости под нагрузкой это успевало
    # прокрутить тысячи поколений — мир вымирал (с появлением дождя это стало
    # возможно), и следующий тест звука видел пустой мир. Плавающее падение
    # ровно в полном прогоне, поодиночке тест всегда проходил.
    engine = Engine(cfg, rate=60, components=True)
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


@pytest.fixture(autouse=True)
def _lab_closed(page):
    """Лаборатория генома — модальное окно поверх всего: забытое открытым, оно
    перехватывает клики следующего теста. Закрываем до и после каждого."""
    page.evaluate("window.viewer && viewer.labOpen(false)")
    yield
    page.evaluate("window.viewer && viewer.labOpen(false)")


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
    """Лаборатория генома приходит с конфигом и правит геномы на сервере."""
    page.evaluate("viewer.labOpen(true)")
    cfg = page.evaluate("viewer.CFG.data")
    assert cfg and len(cfg["fields"]) == 14 and len(cfg["genomes"]) == len(cfg["names"])
    assert "травоядное" in cfg["names"] and "хищник" in cfg["names"]
    # карточка на каждый ген: значок, ползунок, число и описание
    assert page.locator("#genes .gcard").count() == 14
    assert page.locator("#genes input[type=range]").count() == 14
    assert page.locator("#genes .gcard svg.icon").count() == 14
    assert page.locator("#speciesPick .sprow").count() == len(cfg["names"])
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


def test_starters_and_reseed_panel(page, server):
    """Панель «Кем заселять» строится из конфига движка, а галочка повторного
    засева открывает свои настройки."""
    k = page.evaluate("document.querySelectorAll('#starters .gene').length")
    assert k == 8, k                              # восемь видов «экологии»
    names = page.evaluate("Array.from(document.querySelectorAll('#starters .gene span'))"
                          ".map(e=>e.textContent.trim())")
    assert "мох" in names and "травоядное" in names
    assert page.evaluate("getComputedStyle(document.getElementById('reseedBox')).display") == "none"
    page.evaluate("document.getElementById('reseedOn').checked = true;"
                  " document.getElementById('reseedOn').onchange()")
    page.wait_for_timeout(300)
    assert page.evaluate("getComputedStyle(document.getElementById('reseedBox')).display") != "none"
    assert server["engine"].cfg.reseed is True
    page.evaluate("document.getElementById('reseedOn').checked = false;"
                  " document.getElementById('reseedOn').onchange()")
    page.wait_for_timeout(300)


def test_genome_lab_opens_and_describes_genes(page, server):
    """Окно открывается кнопкой и клавишей, у каждого гена есть описание,
    а в «экологии» ответвление недоступно — виды здесь фиксированы."""
    page.evaluate("viewer.labOpen(false)")
    assert page.locator("#lab").is_hidden()
    page.click("#btnLab")
    assert page.locator("#lab").is_visible()
    docs = page.evaluate("viewer.CFG.data.gene_docs")
    fields = page.evaluate("viewer.CFG.data.fields")
    assert all(f in docs and len(docs[f]) > 20 for f in fields), docs
    texts = page.evaluate("Array.from(document.querySelectorAll('#genes .gc-doc')).map(e=>e.textContent)")
    assert sum(bool(t.strip()) for t in texts) == len(fields)
    assert page.evaluate("document.getElementById('btnForkGenes').disabled") is True
    # выбор вида в списке меняет карточку
    page.locator("#speciesPick .sprow").nth(6).click()
    assert page.evaluate("viewer.CFG.pick") == 7
    assert "травоядное" in page.locator("#labHead").inner_text()
    # поиск фильтрует список
    page.fill("#labSearch", "дерев")
    assert page.locator("#speciesPick .sprow").count() == 1
    page.fill("#labSearch", "")
    page.keyboard.press("Escape")
    assert page.locator("#lab").is_hidden()


def test_genome_lab_forks_species_in_terra(page, server):
    """В «терре» виды рождаются сами — там кнопка «Ответвить» работает и
    создаёт потомка с родословной, не трогая родителя."""
    e = server["engine"]
    page.evaluate("viewer.CFG.world.n = 32")
    page.evaluate("viewer.send({cmd:'engine', value:'terra', n:32})")
    page.wait_for_function("viewer.CFG.data && viewer.CFG.data.engine === 'terra'", timeout=20000)
    try:
        for _ in range(30):
            e.advance()
        e.publish(force=True)
        page.evaluate("viewer.send({cmd:'config'})")
        page.wait_for_timeout(400)
        page.evaluate("viewer.labOpen(true)")
        assert page.evaluate("document.getElementById('btnForkGenes').disabled") is False
        before = sorted(e.state["registry"])
        page.evaluate("viewer.CFG.pick = 1")
        page.evaluate("viewer.geneSliders()")
        page.click("#btnForkGenes")
        page.wait_for_timeout(800)
        after = sorted(e.state["registry"])
        assert len(after) > len(before)
        new = [s for s in after if s not in before][0]
        assert e.state["registry"][new]["parent"] == before[0]
        assert "ответвлён вид" in page.locator("#labStatus").inner_text()
        page.evaluate("viewer.labOpen(false)")
    finally:
        page.evaluate("viewer.CFG.world.n = 32")
        page.evaluate("viewer.send({cmd:'engine', value:'ecology', n:32})")
        page.wait_for_function("viewer.CFG.data && viewer.CFG.data.engine === 'ecology'", timeout=20000)


def test_voice_pitch_glides_instead_of_freezing(page, server):
    """У голоса организма раньше частота застывала навсегда в момент
    рождения: обновлялись только громкость, пан и вибрато, а высота — нет,
    даже когда организм вырастал и его гармоника (v.harmonic) менялась.
    На разреженном пересчёте организмов это было незаметно, но качественно
    неверно и способствовало ощущению «дискретных нот» вместо плавного звука."""
    page.click("#btnAudio")
    page.wait_for_timeout(200)
    page.evaluate("""() => {
        const sf1 = {gen: 1, harmonics: new Array(64).fill(0), noise: new Array(64).fill(0),
                     base_hz: 55, voices: [{vid: 777, harmonic: 6, amp: 1, pan: 0, vib: 0}]};
        viewer.Audio.apply(sf1);
    }""")
    h1 = page.evaluate("viewer.Audio.voices.get(777).h")
    f1 = page.evaluate("viewer.Audio.voices.get(777).o.frequency.value")
    assert h1 == 6
    page.evaluate("""() => {
        const sf2 = {gen: 2, harmonics: new Array(64).fill(0), noise: new Array(64).fill(0),
                     base_hz: 55, voices: [{vid: 777, harmonic: 2, amp: 1, pan: 0, vib: 0}]};
        viewer.Audio.apply(sf2);
    }""")
    # тот же голос (тот же vid) — значит организм вырос и стал звучать ниже.
    # node.h обязан отразить новую гармонику немедленно (это и планирует
    # скольжение частоты), даже если сам сигнал доедет до цели по рампе.
    h2 = page.evaluate("viewer.Audio.voices.get(777).h")
    assert h2 == 2, "частота голоса застыла на исходной гармонике"
    assert page.evaluate("viewer.Audio.voices.size") == 1, "должен быть тот же голос, не новый"


def test_colorful_mode_recolors_bands_and_opens_reverb_send(page, server):
    """"Кристалл" — опциональный красочный режим (checkbox #colorful, выкл по
    умолчанию): полосы должны звучать не плоским целочисленным рядом
    гармоник (baseHz*h), а аккордом, окрашенным доминирующим видом полосы
    (sf.band_species), и должен открыться посыл в реверб. Выключение режима
    обязано вернуть полосы на плоский ряд и закрыть посыл.

    page — фикстура модульного масштаба и мир мог быть не на паузе (другие
    тесты его возобновляют) — значит настоящие кадры от сервера продолжают
    прилетать параллельно. apply() и чтение результата объединены в один
    evaluate(), чтобы между "применили" и "прочитали" не мог вклиниться
    реальный кадр (JS однопоточный — внутри одного evaluate() это атомарно).
    Заодно звук включаем явно, а не слепым кликом (тот бы просто выключил
    уже включённый предыдущим тестом звук, и apply() молча вышел бы по
    !this.on)."""
    if not page.evaluate("viewer.Audio.on"):
        page.click("#btnAudio")
    page.wait_for_timeout(200)
    assert page.evaluate("viewer.Audio.on") is True
    assert page.evaluate("viewer.Audio.colorful") is False   # по умолчанию выключен

    def sound_frame(dominant_species):
        band_species = [0] * 64
        band_species[5] = dominant_species
        harmonics = [0] * 64
        harmonics[5] = 1.0
        return {"gen": 1, "harmonics": harmonics, "noise": [0] * 64, "base_hz": 55,
                "voices": [], "band_species": band_species}

    apply_and_read = "(sf) => { viewer.Audio.apply(sf); return viewer.Audio.harm[5]._f; }"

    flat_f = page.evaluate(apply_and_read, sound_frame(3))
    assert flat_f == pytest.approx(55 * 6)   # обычный режим — плоская 6-я гармоника

    page.check("#colorful")
    assert page.evaluate("viewer.Audio.colorful") is True
    assert page.evaluate("viewer.Audio.sendH._amt") > 0
    assert page.evaluate("viewer.Audio.sendV._amt") > 0
    colorful_f = page.evaluate(apply_and_read, sound_frame(3))
    assert colorful_f != pytest.approx(flat_f), "частота полосы не изменилась в красочном режиме"
    # тот же номер полосы, но другой доминирующий вид -> другая нота
    other_species_f = page.evaluate(apply_and_read, sound_frame(5))
    assert other_species_f != pytest.approx(colorful_f), "разные виды должны звучать разными нотами"

    page.uncheck("#colorful")
    assert page.evaluate("viewer.Audio.sendH._amt") == 0
    back_f = page.evaluate(apply_and_read, sound_frame(3))
    assert back_f == pytest.approx(flat_f), "выключение режима должно вернуть плоскую гармонику"

    page.click("#btnAudio")   # звук выключен для следующих тестов, как и раньше в файле
    assert not page.evaluate("viewer.Audio.on")


def test_voice_pitch_updates_are_staggered_not_synchronized(page, server):
    """Баг-жалоба: "циклические ноты... на разной высоте, но интервал
    подозрительно одинаковый, будто нота нажимается через равный
    промежуток". Причина: сервер пересчитывает организмов (а с ними —
    высоту голосов) не каждый кадр, а пачкой, раз в components_hz (и того
    реже на населённом мире — сама разметка стоит дороже интервала). Когда
    пачка приходит, ВСЕ голоса, чья гармоника изменилась, получали новую
    цель ровно в один и тот же момент t — на слух синхронный "щелчок"/аккорд
    на подозрительно ровном интервале, никак не связанный с тем, что
    происходит с клетками между пересчётами. voiceJitter должен развести
    старт скольжения высоты по времени (детерминированно, по vid), так
    чтобы одновременное изменение гармоники у нескольких голосов не
    стартовало одним кликом."""
    if not page.evaluate("viewer.Audio.on"):
        page.click("#btnAudio")
    page.wait_for_timeout(200)
    assert page.evaluate("viewer.Audio.on") is True

    def sf_with_voices(voices):
        return {"gen": 1, "harmonics": [0] * 64, "noise": [0] * 64, "base_hz": 55,
                "voices": voices, "band_species": [0] * 64}

    # рождаем два голоса разных id
    page.evaluate("(sf) => viewer.Audio.apply(sf)", sf_with_voices([
        {"vid": 501, "harmonic": 6, "amp": 1, "pan": 0, "vib": 0, "species": 1},
        {"vid": 502, "harmonic": 8, "amp": 1, "pan": 0, "vib": 0, "species": 2},
    ]))
    page.wait_for_timeout(50)

    # шпион на setTargetAtTime — фиксируем момент, на который сервер
    # (сервер тут — сама тестовая функция) планирует старт новой кривой
    page.evaluate("""() => {
        window.__calls = [];
        const orig = AudioParam.prototype.setTargetAtTime;
        AudioParam.prototype.setTargetAtTime = function(value, startTime, tc){
            window.__calls.push({value, startTime, ctxTime: viewer.Audio.ctx.currentTime});
            return orig.call(this, value, startTime, tc);
        };
    }""")
    # оба голоса меняют гармонику ОДНИМ и тем же кадром — именно так и
    # приходит пачка с сервера раз в components_hz
    page.evaluate("(sf) => viewer.Audio.apply(sf)", sf_with_voices([
        {"vid": 501, "harmonic": 2, "amp": 1, "pan": 0, "vib": 0, "species": 1},
        {"vid": 502, "harmonic": 4, "amp": 1, "pan": 0, "vib": 0, "species": 2},
    ]))
    calls = page.evaluate("window.__calls")
    # 501 -> harmonic 2 => 55*2=110; 502 -> harmonic 4 => 55*4=220 (различимы по value)
    delay_501 = next(c["startTime"] - c["ctxTime"] for c in calls if c["value"] == pytest.approx(110))
    delay_502 = next(c["startTime"] - c["ctxTime"] for c in calls if c["value"] == pytest.approx(220))
    assert delay_501 != pytest.approx(delay_502), \
        "два голоса меняют высоту синхронно — тот самый синхронный щелчок"
    assert 0 <= delay_501 < 0.4 and 0 <= delay_502 < 0.4

    page.click("#btnAudio")
    assert not page.evaluate("viewer.Audio.on")


def test_granular_sample_is_built_from_genome(page, server):
    """Сэмпл вида синтезируется ИЗ ГЕНОМА (Audio.grainBuffer) — в этом вся
    суть режима. Проверяем три вещи, без которых фича — просто шумелка:
    один и тот же геном даёт побитово тот же сэмпл (детерминизм — иначе
    тембр вида плавал бы от кадра к кадру), разные геномы дают РАЗНЫЕ
    сэмплы, и правка гена в лаборатории пересобирает сэмпл."""
    page.click("#btnAudio")
    try:
        # детерминизм: два вызова подряд — один и тот же буфер (из кэша) и
        # одинаковое содержимое даже после сброса кэша
        same = page.evaluate("""() => {
            const A = viewer.Audio;
            A.grainBufs.clear();
            const a = A.grainBuffer(1).getChannelData(0).slice(0, 4096);
            A.grainBufs.clear();
            const b = A.grainBuffer(1).getChannelData(0).slice(0, 4096);
            let diff = 0; for (let i=0;i<a.length;i++) diff += Math.abs(a[i]-b[i]);
            return diff;
        }""")
        assert same == 0, same

        # разные виды (мох/хищник — заведомо разные геномы) звучат по-разному
        diff_species = page.evaluate("""() => {
            const A = viewer.Audio;
            const a = A.grainBuffer(1).getChannelData(0);
            const b = A.grainBuffer(8).getChannelData(0);
            let d = 0; for (let i=0;i<4096;i++) d += Math.abs(a[i]-b[i]);
            return d / 4096;
        }""")
        assert diff_species > 0.05, diff_species

        # сэмпл не статичен: начало и конец различаются, иначе точка чтения
        # (глубина клетки по y) была бы не слышна вообще
        morph = page.evaluate("""() => {
            const d = viewer.Audio.grainBuffer(1).getChannelData(0);
            const n = d.length;
            const rms = (from) => { let s=0; for (let i=from;i<from+4096;i++) s+=d[i]*d[i];
                                    return Math.sqrt(s/4096); };
            const head = rms(0), tail = rms(n-8192);
            return Math.abs(head - tail) / Math.max(head, tail, 1e-9);
        }""")
        assert morph > 0.02, morph

        # правка гена -> новый сэмпл (кэш инвалидируется по геному, а не по id)
        changed = page.evaluate("""() => {
            const A = viewer.Audio;
            const a = A.grainBuffer(1).getChannelData(0).slice(0, 4096);
            const before = viewer.CFG.edited[0][0];
            viewer.CFG.edited[0][0] = before > 0.5 ? 0.05 : 0.95;
            const b = A.grainBuffer(1).getChannelData(0).slice(0, 4096);
            viewer.CFG.edited[0][0] = before;
            let d = 0; for (let i=0;i<a.length;i++) d += Math.abs(a[i]-b[i]);
            return d / a.length;
        }""")
        assert changed > 0.01, changed
    finally:
        page.click("#btnAudio")


def test_granular_schedules_grains_only_when_enabled(page, server):
    """Зёрна реально сыплются, когда режим включён, и не сыплются, когда
    выключен. Заодно ловим немой чекбокс: без планировщика grainsPlayed
    остался бы нулевым при любом положении галочки."""
    page.click("#btnAudio")
    try:
        if page.evaluate("viewer.Audio.ctx.state") != "running":
            pytest.skip("AudioContext не запустился в headless — планировщику не по чему тикать")
        # выключено — счётчик стоит
        page.evaluate("viewer.Audio.grainsPlayed = 0; viewer.Audio.setGrain(false)")
        page.wait_for_timeout(300)
        assert page.evaluate("viewer.Audio.grainsPlayed") == 0

        page.check("#grainOn")
        assert page.evaluate("viewer.Audio.grainOn") is True
        page.wait_for_function("viewer.Audio.grainsPlayed > 0", timeout=5000)
        played = page.evaluate("viewer.Audio.grainsPlayed")
        assert played > 0

        # сэмплы собрались только для тех видов, что реально живут в мире
        live = {s + 1 for s, p in enumerate(page.evaluate("viewer.S.pops")) if p > 0}
        built = set(page.evaluate("[...viewer.Audio.grainBufs.keys()]"))
        assert built and built <= live, (built, live)

        # выключили — счётчик перестал расти
        page.uncheck("#grainOn")
        page.wait_for_timeout(200)
        stopped = page.evaluate("viewer.Audio.grainsPlayed")
        page.wait_for_timeout(400)
        assert page.evaluate("viewer.Audio.grainsPlayed") == stopped
    finally:
        page.evaluate("viewer.Audio.setGrain(false)")
        page.click("#btnAudio")


def test_granular_grain_maps_cell_axes_to_sound(page, server):
    """Раскладка по осям — то, ради чего клетка вообще становится зерном:
    x -> панорама, y -> точка чтения сэмпла, z -> высота тона. Подсовываем
    конкретные клетки и смотрим, что получил узел зерна."""
    page.click("#btnAudio")
    try:
        got = page.evaluate("""() => {
            const A = viewer.Audio, n = viewer.S.n;
            const spy = [];
            const origBuf = A.ctx.createBufferSource.bind(A.ctx);
            const origPan = A.ctx.createStereoPanner.bind(A.ctx);
            let last = {};
            A.ctx.createStereoPanner = () => { const p = origPan(); last.panner = p; return p; };
            A.ctx.createBufferSource = () => {
                const s = origBuf();
                const start = s.start.bind(s);
                s.start = (when, off, dur) => { last.off = off; last.rate = s.playbackRate.value;
                                                return start(when, off, dur); };
                return s;
            };
            const buf = A.grainBuffer(1);
            // подменяем клетки на две крайние — и возвращаем как было, иначе
            // подделку увидит следующий тест (page/S общие на весь модуль)
            const keepC = viewer.S.coords, keepS = viewer.S.species;
            viewer.S.coords = new Uint16Array([0, 0, 0,  n-1, n-1, n-1]);
            viewer.S.species = new Uint8Array([1, 1]);
            A.playGrain(buf, 0, 1, A.ctx.currentTime + 0.05, 0.05, 40);
            const lo = {pan: last.panner.pan.value, off: last.off, rate: last.rate};
            A.playGrain(buf, 1, 1, A.ctx.currentTime + 0.05, 0.05, 40);
            const hi = {pan: last.panner.pan.value, off: last.off, rate: last.rate};
            A.ctx.createBufferSource = origBuf; A.ctx.createStereoPanner = origPan;
            viewer.S.coords = keepC; viewer.S.species = keepS;
            return {lo, hi, dur: buf.duration};
        }""")
        lo, hi = got["lo"], got["hi"]
        # x: левый край -> панорама влево, правый -> вправо
        assert lo["pan"] == pytest.approx(-1, abs=0.05), lo
        assert hi["pan"] > 0.8, hi
        # y: глубина куба -> точка чтения сэмпла (в начале vs в конце)
        assert lo["off"] == pytest.approx(0, abs=1e-6), lo
        assert hi["off"] > got["dur"] * 0.5, (hi, got["dur"])
        # z: высота -> высота тона (верхняя клетка звучит выше нижней)
        assert hi["rate"] > lo["rate"] * 1.5, (lo, hi)
    finally:
        page.click("#btnAudio")


def test_granular_bus_is_actually_audible_and_faders_move_it(page, server):
    """САМАЯ ВАЖНАЯ проверка режима, и та, которой сначала не было: зёрна не
    просто создаются — на шине гранул есть РЕАЛЬНЫЙ СИГНАЛ слышимого уровня,
    и ползунки его двигают.

    Первая версия режима все проверки на узлах проходила, но звучала на
    ~14 дБ тише полос (0.04 против 0.20 по пику) и полностью тонула в
    миксе — «включаю, кручу ползунки, звук не меняется». Тесты на создание
    узлов такое пропускают по построению; ловит только замер выхода."""
    page.click("#btnAudio")
    try:
        if page.evaluate("viewer.Audio.ctx.state") != "running":
            pytest.skip("AudioContext не запустился в headless — мерить нечего")
        page.check("#grainOn")
        # глушим остальные слои: меряем именно гранулы
        page.evaluate("viewer.Audio.busH.gain.value=0; viewer.Audio.busV.gain.value=0;"
                      " viewer.Audio.busN.gain.value=0")

        tap = """() => {
            const A = viewer.Audio;
            clearInterval(window.__poll);
            if (window.__an) { try { A.busG.disconnect(window.__an); } catch(e){} }
            const an = A.ctx.createAnalyser(); an.fftSize = 2048;
            A.busG.connect(an); window.__an = an; window.__peak = 0;
            window.__poll = setInterval(() => {
                const b = new Float32Array(2048); an.getFloatTimeDomainData(b);
                let m = 0; for (let i=0;i<b.length;i++) m = Math.max(m, Math.abs(b[i]));
                if (m > window.__peak) window.__peak = m;
            }, 20);
        }"""

        page.evaluate(tap)
        page.wait_for_timeout(1500)
        loud = page.evaluate("window.__peak")
        # до фикса тут было ~0.04 — формально «работает», на слух нет
        assert loud > 0.12, loud

        # ползунок микса реально двигает уровень
        page.evaluate("viewer.Audio.busG.gain.value = 0.15")
        page.evaluate(tap)
        page.wait_for_timeout(1500)
        quiet = page.evaluate("window.__peak")
        assert quiet < loud * 0.5, (loud, quiet)

        # ползунок плотности реально меняет поток зёрен
        page.evaluate("viewer.Audio.busG.gain.value = 0.7")
        rates = {}
        for want in (4, 140):
            page.evaluate(f"() => {{ document.getElementById('grainRate').value = {want}; }}")
            page.wait_for_timeout(250)
            n0 = page.evaluate("viewer.Audio.grainsPlayed")
            page.wait_for_timeout(1200)
            rates[want] = (page.evaluate("viewer.Audio.grainsPlayed") - n0) / 1.2
        assert rates[140] > rates[4] * 4, rates
    finally:
        page.evaluate("""() => {
            clearInterval(window.__poll);
            const A = viewer.Audio;
            document.getElementById('grainRate').value = 40;
            A.busG.gain.value = 0.7;
            A.busH.gain.value = 0.7; A.busV.gain.value = 0.6; A.busN.gain.value = 0.3;
            A.setGrain(false);
        }""")
        page.uncheck("#grainOn")
        page.click("#btnAudio")


# признаки тембра, замеренные ИЗ ЗВУКА (не из параметров): спектр по
# гармоникам опорных 110 Гц, центроид, баланс чётных/нечётных, шумность
# (энергия вне гармоник) и глубина пульсации огибающей
_TIMBRE_FEATURES_JS = """() => {
  const A=viewer.Audio, sr=A.ctx.sampleRate, NH=16, out=[];
  for (let sp=1; sp<=8; sp++){
    const b=A.grainBuffer(sp), d=b.getChannelData(0);
    const N=8192, off=(d.length/3)|0;
    const mag=[];
    for(let k=1;k<=NH;k++){
      const w=2*Math.PI*110*k/sr; let re=0,im=0;
      for(let i=0;i<N;i++){ const v=d[off+i]; re+=v*Math.cos(w*i); im+=v*Math.sin(w*i); }
      mag.push(Math.sqrt(re*re+im*im)/N);
    }
    const tot=mag.reduce((a,b)=>a+b,0)||1e-9;
    let cen=0; for(let k=0;k<NH;k++) cen += (k+1)*mag[k]/tot;
    let ev=0, od=0; for(let k=0;k<NH;k++){ if((k+1)%2) od+=mag[k]; else ev+=mag[k]; }
    let tot2=0; for(let i=0;i<N;i++) tot2 += d[off+i]*d[off+i];
    const harm = mag.reduce((a,b)=>a+b*b,0)*N*2;
    const W=512, env=[];
    for(let i=0;i+W<d.length;i+=W){ let s=0; for(let j=0;j<W;j++) s+=d[i+j]*d[i+j]; env.push(Math.sqrt(s/W)); }
    const em=env.reduce((a,b)=>a+b,0)/env.length;
    let va=0; for(const v of env) va+=Math.pow(v-em,2);
    out.push({ centroid:cen, evenOdd:ev/Math.max(od,1e-9),
               noise:Math.max(0,1-harm/Math.max(tot2,1e-9)),
               trem:Math.sqrt(va/env.length)/Math.max(em,1e-9) });
  }
  return out;
}"""


def test_species_timbre_axes_are_all_gene_driven(page, server):
    """«Все виды звучат одинаково»: тембр брался по ФИКСИРОВАННЫМ индексам
    гена — шум из гена №7 (trophic), негармоничность из №13 (armor). У всех
    шести растений «экологии» звериные гены нулевые, поэтому обе оси у них
    совпадали до последнего знака (шум 0.050, негармоничность 0.0040), а
    растения — это почти весь мир.

    Теперь каждая ось — знаковая проекция ВСЕГО генома в z-оценках
    относительно разброса генов по видам этого мира. Требование: ни одна ось
    не смеет быть константой по видам."""
    page.click("#btnAudio")
    try:
        T = page.evaluate("[...Array(8)].map((_,i)=>viewer.Audio.grainTimbre(i+1))")
        assert all(T), T
        plants = T[:6]          # шесть растений: именно они были клонами
        for axis in ("tilt", "odd", "inh", "nz", "fPos", "morph", "tremHz"):
            vals = [t[axis] for t in plants]
            rng = max(vals) - min(vals)
            rel = rng / max(max(abs(v) for v in vals), 1e-9)
            assert rel > 0.15, (axis, vals)

        # виды не должны сталкиваться в одну ноту: PENTA сворачивалась в
        # октаву (%12) и на восемь видов приходилось всего пять высот
        degs = page.evaluate("[...Array(8)].map((_,i)=>viewer.Audio.PENTA[i %"
                             " viewer.Audio.PENTA.length])")
        assert len(set(degs)) == 8, degs
    finally:
        page.click("#btnAudio")


def test_species_timbres_are_audibly_apart(page, server):
    """И это слышно в самом звуке, а не только в параметрах: замеряем спектр
    сэмплов и требуем широкого разброса по яркости и по пульсации."""
    page.click("#btnAudio")
    try:
        F = page.evaluate(_TIMBRE_FEATURES_JS)
        cen = [f["centroid"] for f in F]
        trem = [f["trem"] for f in F]
        eo = [f["evenOdd"] for f in F]
        # яркость: до фикса центроид укладывался в 1.6x, теперь ~2.5x
        assert max(cen) / min(cen) > 2.0, cen
        # пульсация: до фикса разброс был ~0.15 (побочный артефакт), теперь ~0.6
        assert max(trem) - min(trem) > 0.35, trem
        assert max(eo) - min(eo) > 0.25, eo
        # и каждая пара видов различается хотя бы по одной оси
        def sep(a, b):
            return max(abs(a["centroid"] - b["centroid"]) / max(a["centroid"], b["centroid"]),
                       abs(a["trem"] - b["trem"]) / 3.0,
                       abs(a["evenOdd"] - b["evenOdd"]) / 0.5,
                       abs(a["noise"] - b["noise"]) / 0.3)
        worst = min(sep(F[i], F[j]) for i in range(8) for j in range(8) if i < j)
        assert worst > 0.09, worst
    finally:
        page.click("#btnAudio")


def test_note_mode_plays_audible_notes_only_when_enabled(page, server):
    """Режим «Ноты»: отдельные ноты по сетке вместо сплошного дрона гранул.
    Проверяем не только счётчик, но и РЕАЛЬНЫЙ сигнал на шине — урок
    гранулярного режима, который «работал», но был на 14 дБ тише всех."""
    page.click("#btnAudio")
    try:
        if page.evaluate("viewer.Audio.ctx.state") != "running":
            pytest.skip("AudioContext не запустился в headless")
        page.evaluate("viewer.Audio.notesPlayed = 0; viewer.Audio.setNotes(false)")
        page.wait_for_timeout(300)
        assert page.evaluate("viewer.Audio.notesPlayed") == 0

        page.evaluate("viewer.Audio.busH.gain.value=0; viewer.Audio.busV.gain.value=0;"
                      " viewer.Audio.busN.gain.value=0; viewer.Audio.busG.gain.value=0")
        page.check("#noteOn")
        page.evaluate("""() => {
            const A=viewer.Audio; const an=A.ctx.createAnalyser(); an.fftSize=2048;
            A.busNo.connect(an); window.__nan=an; window.__peak=0;
            clearInterval(window.__poll);
            window.__poll=setInterval(()=>{ const b=new Float32Array(2048);
                an.getFloatTimeDomainData(b);
                let m=0; for(let i=0;i<b.length;i++) m=Math.max(m,Math.abs(b[i]));
                if(m>window.__peak) window.__peak=m; }, 20);
        }""")
        page.wait_for_function("viewer.Audio.notesPlayed > 0", timeout=6000)
        page.wait_for_timeout(1800)
        assert page.evaluate("window.__peak") > 0.10, page.evaluate("window.__peak")

        page.uncheck("#noteOn")
        page.wait_for_timeout(250)
        stopped = page.evaluate("viewer.Audio.notesPlayed")
        page.wait_for_timeout(500)
        assert page.evaluate("viewer.Audio.notesPlayed") == stopped
    finally:
        page.evaluate("""() => { clearInterval(window.__poll); const A=viewer.Audio;
            A.setNotes(false);
            A.busH.gain.value=0.7; A.busV.gain.value=0.6; A.busN.gain.value=0.3;
            A.busG.gain.value=0.7; }""")
        page.click("#btnAudio")


def test_adsr_differs_per_species_and_follows_the_simulation(page, server):
    """Форма ADSR завязана на симуляцию отдельно по каждому виду.

    Две проверки. Первая — что огибающая вообще различает виды: если брать
    только именованные гены (metabolism/lifespan), у всех шести растений
    «экологии» они нулевые и ADSR совпал бы до знака — ровно та ловушка,
    из-за которой «все виды звучат одинаково». Вторая — что огибающая
    реагирует на ЖИВОЕ состояние мира: растущий вид бьёт резче, вымирающий
    звенит дольше."""
    page.click("#btnAudio")
    try:
        E = page.evaluate("[...Array(8)].map((_,i)=>viewer.Audio.speciesADSR(i+1))")
        plants = E[:6]
        for axis in ("a", "d", "s", "r"):
            vals = [e[axis] for e in plants]
            assert (max(vals) - min(vals)) / max(vals) > 0.08, (axis, vals)

        # взрывной рост -> атака резче; вымирание -> атака мягче, релиз длиннее
        base = page.evaluate("(()=>{viewer.Audio._growth[0]=0; return viewer.Audio.speciesADSR(1);})()")
        boom = page.evaluate("(()=>{viewer.Audio._growth[0]=0.3; return viewer.Audio.speciesADSR(1);})()")
        dying = page.evaluate("(()=>{viewer.Audio._growth[0]=-0.3; return viewer.Audio.speciesADSR(1);})()")
        page.evaluate("viewer.Audio._growth[0]=0")
        assert boom["a"] < base["a"] * 0.5, (base, boom)
        assert dying["a"] > base["a"] * 1.4, (base, dying)
        assert dying["r"] > base["r"] * 1.5, (base, dying)
    finally:
        page.click("#btnAudio")


def test_euclidean_rhythm_tracks_species_share(page, server):
    """Ритм вида — евклидов рисунок, плотность которого задаёт доля вида в
    населении: доминирующий бьёт часто, редкий изредка. Без этого сетка была
    бы метрономом, одинаковым для всех и не связанным с экологией."""
    page.click("#btnAudio")
    try:
        hits = page.evaluate("""() => {
            const A=viewer.Audio, n=16, out={};
            for (const [name, share] of [['rare',0.05],['mid',0.4],['dom',0.95]]){
                const k=Math.max(1, Math.min(n, Math.round(1 + share*(n-1)*0.8)));
                let c=0; for(let i=0;i<n;i++) if (A.euclid(k,n,i)) c++;
                out[name]=c;
            }
            // и сам рисунок не вырожденный: E(5,16) даёт ровно 5 ударов,
            // распределённых, а не слипшихся в начало
            const pat=[]; for(let i=0;i<16;i++) pat.push(A.euclid(5,16,i)?1:0);
            out.pat=pat;
            return out;
        }""")
        assert hits["rare"] < hits["mid"] < hits["dom"], hits
        assert sum(hits["pat"]) == 5, hits["pat"]
        # удары не слиплись в первые пять шагов
        assert sum(hits["pat"][:5]) < 4, hits["pat"]
    finally:
        page.click("#btnAudio")
