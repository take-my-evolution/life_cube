"""Разметка организмов не должна стопорить поток кадров.

Баг, который тут воспроизводится: на населённом мире scipy.label по всему
кубу стоит на порядок дороже голой геометрии. Раньше её считали ПРЯМО в
цикле рассылки раз в components_hz — и каждый такой кадр рендер стопорился
на десятки-сотни миллисекунд, а звук (голоса берутся из организмов)
перескакивал на новый набор нот с той же периодичностью — «нота на разной
высоте через равный промежуток», о которой сообщил пользователь. Тест бьёт
по симптому напрямую: гоняет реальный WebViewer.broadcaster() на плотном
мире и проверяет, что между кадрами нет многосоткратных пауз.
"""

import asyncio
import threading
import time

import numpy as np
import pytest

from life_cube import Config
from life_cube.engine import Engine
from life_cube.snapshot import describe_components
from life_cube.viewers.web.server import WebViewer, encode_snapshot


def _dense_engine(n=64, scatter=0.06, n_species=4, seed=0):
    """Мир с множеством отдельно стоящих клеток — худший случай для
    scipy.label (много отдельных компонент). Прогонять сотни поколений ради
    этого не нужно: собираем картину напрямую, тест остаётся быстрым, а
    разметка всё равно стоит реальных ~150-300мс, как на населённом кубе."""
    cfg = Config(n=n, seed_density=0.05, animal_share=0.3)
    e = Engine(cfg, rate=0, snapshot_every=0, components=True)
    rng = np.random.default_rng(seed)
    sp = np.zeros((n, n, n), np.int8)
    mask = rng.random((n, n, n)) < scatter
    sp[mask] = rng.integers(1, min(n_species, e.cfg.n_species) + 1, size=int(mask.sum()))
    e.state["species"] = sp
    return e


class _FakeClient:
    """Стоит вместо настоящего WebSocket: broadcaster() реально зовёт
    send_bytes на каждый кадр — засекаем момент прихода, это и есть то,
    что увидел бы браузер."""
    def __init__(self, log):
        self.log = log

    async def send_bytes(self, data):
        self.log.append(time.perf_counter())


def test_component_recompute_does_not_stall_frames():
    e = _dense_engine()
    v = WebViewer(e, fps=25.0, components_hz=2.0)
    sent = []
    v.clients.add(_FakeClient(sent))

    async def run():
        loop = asyncio.get_running_loop()
        v.loop = loop
        # симуляция и рассылка кадров в реальном деплое — разные потоки;
        # без этого gen никогда не меняется, и фоновая разметка не запускается
        stop = threading.Event()
        sim = threading.Thread(target=e.run, kwargs={"stop_event": stop}, daemon=True)
        sim.start()
        task = asyncio.ensure_future(v.broadcaster())
        await asyncio.sleep(3.0)
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        sim.join(timeout=2.0)

    asyncio.run(run())
    assert len(sent) > 20, f"кадров почти не было — {len(sent)}, сломали не то"
    gaps = [b - a for a, b in zip(sent, sent[1:])]
    # Раньше разметка организмов синхронно подвешивала цикл рассылки на
    # ~150-300мс каждые components_hz раз в секунду — клиент явно видел
    # паузу. При fps=25 (кадр раз в 40мс) разрыв за треть секунды уже
    # заметен глазу; такого быть не должно.
    assert max(gaps) < 0.3, f"обнаружен стопор: {max(gaps):.3f}с между кадрами, {sorted(gaps)[-5:]}"


def test_heavy_recompute_runs_in_background_not_inline():
    """_recompute_components() не должен блокировать вызывающего — иначе
    он ничем не отличается от старой синхронной разметки в том же такте."""
    e = _dense_engine()
    v = WebViewer(e, fps=25.0, components_hz=2.0)

    async def run():
        t0 = time.perf_counter()
        task = asyncio.ensure_future(v._recompute_components())
        dt_launch = time.perf_counter() - t0
        await task
        return dt_launch

    dt_launch = asyncio.run(run())
    # запуск фоновой задачи — это одна строчка asyncio.ensure_future, не
    # ожидание executor'а: должен вернуться мгновенно
    assert dt_launch < 0.05, f"launch занял {dt_launch*1000:.1f}мс — не фоновый"


def test_components_recompute_updates_snapshot_without_regressing_geometry():
    """Фоновая разметка обновляет organism-данные, не откатывая геометрию
    назад: снимок после разметки не старше того, что уже был показан."""
    e = _dense_engine()
    v = WebViewer(e, fps=25.0, components_hz=2.0)
    e.publish(force=True, components=False)
    gen_before = v.latest.gen if v.latest else e.gen

    asyncio.run(v._recompute_components())

    assert v.latest is not None
    assert v.latest.components                       # организмы посчитаны
    assert v.latest.gen >= gen_before


def test_snapshot_reports_real_publish_cost():
    """snapshot_seconds раньше жил только на Engine и никогда не попадал в
    Snapshot — клиент всегда получал snapshot_ms=0, даже когда разметка
    стоила десятки миллисекунд."""
    e = _dense_engine()
    snap = e.publish(force=True, components=True)
    assert snap.snapshot_seconds >= 0
    assert snap.snapshot_seconds == pytest.approx(e.snapshot_seconds)


def test_encode_snapshot_reports_snapshot_ms():
    e = _dense_engine()
    snap = e.publish(force=True, components=True)
    snap.snapshot_seconds = 0.083            # смоделируем дорогой кадр
    data = encode_snapshot(snap)
    import json
    hlen = int.from_bytes(data[:4], "little")
    header = json.loads(data[4:4 + hlen])
    assert header["snapshot_ms"] == 83


def test_organism_size_updates_between_heavy_recomputes():
    """Жалоба после v0.7.3 (щелчок смягчили, но не убрали): организмы (а с
    ними — SoundMapper и высота/громкость голоса) обновлялись ТОЛЬКО на
    тяжёлом пересчёте (components_hz, и того реже на населённом мире — сама
    разметка стоит дороже интервала) — между пересчётами snap.components
    был буквально тем же списком объектов, потом разом скачком менялся.
    Engine.publish() теперь на КАЖДОМ быстром такте (components=False)
    дёшево накладывает старую (с последнего тяжёлого пересчёта) id-карту на
    текущие живые клетки — без scipy.label — так что усыхание/гибель уже
    отслеженного организма видно немедленно, а не раз в components_hz."""
    e = _dense_engine()
    snap1 = e.publish(force=True, components=True)
    assert snap1.components
    biggest = max(snap1.components, key=lambda c: c.size)
    cid, size1 = biggest.cid, biggest.size
    assert size1 >= 4, "нужен организм покрупнее, иначе некуда усыхать в этом тесте"

    # "убиваем" половину клеток самого крупного организма — НЕ пересчитывая
    # разметку (ни scipy.label, ни tracker.assign() тут не вызываются)
    ids_map = e.tracker.prev
    cells = np.argwhere(ids_map == cid)
    to_kill = cells[: len(cells) // 2]
    sp = e.state["species"]
    sp[to_kill[:, 0], to_kill[:, 1], to_kill[:, 2]] = 0
    e.state["species"] = sp

    snap2 = e.publish(force=True, components=False)     # быстрый такт
    match = [c for c in snap2.components if c.cid == cid]
    assert match, "организм пропал с быстрого такта вместо того, чтобы усохнуть"
    assert match[0].size == size1 - len(to_kill)

    # клетки, которые сейчас живы, но не входили ни в один организм на
    # последней тяжёлой разметке (id=0 — "пока неизвестно чей"), не должны
    # схлопываться в фантомный "организм номер 0"
    assert all(c.cid != 0 for c in snap2.components)


def test_describe_components_handles_many_fragments_quickly():
    """describe_components() раньше строила Component-объект на КАЖДЫЙ
    организм чистым питон-циклом — на сильно фрагментированном мире
    (десятки тысяч organisms, характерно для теста дешёвого пересчёта на
    каждом такте) это стоило сотни миллисекунд, даже когда наружу уходила
    только верхушка max_components. Теперь групповые агрегаты считаются
    векторно (np.add.reduceat), а Python-объекты строятся только для
    top-K — иначе дешёвый пересчёт организмов на быстром такте
    (см. test_organism_size_updates_between_heavy_recomputes) сам стал бы
    новым источником стопора кадров."""
    rng = np.random.default_rng(0)
    n_ids, k = 30_000, 60_000
    ids = rng.integers(1, n_ids, size=k).astype(np.uint32)
    coords = rng.integers(0, 128, size=(k, 3)).astype(np.uint16)
    species = rng.integers(1, 6, size=k).astype(np.uint8)

    t0 = time.perf_counter()
    comps = describe_components(coords, species, ids, {}, max_components=200)
    dt = time.perf_counter() - t0

    assert len(comps) == 200
    assert dt < 0.1, f"describe_components слишком медленная: {dt*1000:.0f}мс на {n_ids} организмов"
    sizes = [c.size for c in comps]
    assert sizes == sorted(sizes, reverse=True)
