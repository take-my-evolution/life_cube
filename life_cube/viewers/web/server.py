"""Веб-рендер: aiohttp отдаёт статичный клиент и гонит снимки по WebSocket.

Протокол (один бинарный кадр на снимок):
    uint32 header_len, header JSON (utf-8), затем тела подряд:
      coords   uint16 [k,3]
      species  uint8  [k]
      labels   uint32 [k]
      soil     uint16 [m,3]
    В header: gen, n, k, m, pops, rate, measured_rate, paused, components,
    hist_tail (последние 400 значений), relief (только в первом кадре).

Клиент шлёт JSON-команды: {"cmd": "pause"|"resume"|"step"|"rate", "value": ..}
                          {"cmd": "reset", "seed_world":..,"seed_mut":..}
                          {"cmd": "seeding", "value": {...}, "restart": bool}
                          {"cmd": "fork", "id": вид, "value": геном, "share": доля}
"""

import asyncio
import json
import pathlib
import struct
import threading

import numpy as np

from ...config import Config
from ...engine import Engine
from ...engines import get_rules, list_engines
from ...snapshot import Snapshot
from ...sound import SoundMapper

STATIC = pathlib.Path(__file__).parent / "static"


def _digest(obj):
    import hashlib
    return hashlib.blake2b(repr(obj).encode(), digest_size=8).hexdigest()


def encode_snapshot(snap: Snapshot, first=False, sound=None, sent=None) -> bytes:
    """sent — словарь состояния клиента: что ему уже отправлено (рельеф, имена).
    Всё, что не изменилось, в кадр не попадает: на большом мире рельеф и имена
    64 видов в JSON стоили больше самой геометрии."""
    header = {
        "sound": sound.to_dict() if sound is not None else None,
        "gen": snap.gen, "n": snap.n, "k": int(len(snap.coords)),
        "m": int(len(snap.soil_coords)) if snap.soil_coords is not None else 0,
        "pops": snap.pops,
        "rate": getattr(snap, "rate", 0.0),
        "measured_rate": round(getattr(snap, "measured_rate", 0.0), 2),
        "paused": getattr(snap, "paused", False),
        "stride": getattr(snap, "stride", 1),
        "components_on": getattr(snap, "components_on", True),
        "snapshot_ms": round(getattr(snap, "snapshot_seconds", 0.0) * 1000),
        "components": [[c.cid, c.species, c.size, *c.center, c.zmin, c.zmax, c.born]
                       for c in snap.components],
        "hist_tail": [list(map(int, h)) for h in getattr(snap, "hist", [])],
        # метки организмов не шлём, когда их не считают: это 4 байта на клетку
        "labels": bool(snap.components) and len(snap.labels) == len(snap.coords),
    }
    sent = sent if sent is not None else {}
    names = list(getattr(snap, "species_names", []))[:len(snap.pops)]
    colors = list(getattr(snap, "species_colors", []))[:len(snap.pops)]
    nd = _digest((names, colors))
    if first or nd != sent.get("names"):
        header["species_names"] = names
        header["species_colors"] = colors
        sent["names"] = nd
    if first:
        header["config"] = getattr(snap, "config_json", None)
        header["engines"] = list_engines()
    # карты высот (камень/почва/вода) — бинарно и только когда изменились
    maps = {}
    if snap.stone_h is not None:
        header["heightmaps"] = True
        for key in ("stone_h", "soil_h", "water_h"):
            a = getattr(snap, key, None)
            if a is None:
                continue
            d = _digest(a.tobytes())
            if first or d != sent.get(key):
                maps[key] = True
                sent[key] = d
        header["maps"] = maps
    else:
        relief = getattr(snap, "relief", None)
        if relief is not None:
            rd = _digest(relief.tobytes())
            if first or rd != sent.get("relief"):
                header["relief"] = relief.astype(int).tolist()
                sent["relief"] = rd
    hb = json.dumps(header, ensure_ascii=False).encode()
    soil = snap.soil_coords if snap.soil_coords is not None else np.zeros((0, 3), np.uint16)
    parts = [struct.pack("<I", len(hb)), hb,
             np.ascontiguousarray(snap.coords, dtype=np.uint16).tobytes(),
             np.ascontiguousarray(snap.species, dtype=np.uint8).tobytes()]
    if header["labels"]:
        parts.append(np.ascontiguousarray(snap.labels, dtype=np.uint32).tobytes())
    parts.append(np.ascontiguousarray(soil, dtype=np.uint16).tobytes())
    for key in ("stone_h", "soil_h", "water_h"):
        a = getattr(snap, key, None)
        if a is not None and header.get("maps", {}).get(key):
            parts.append(np.ascontiguousarray(a, dtype=np.uint16).tobytes())
    return b"".join(parts)


def decode_snapshot(buf: bytes):
    """Для тестов: обратное преобразование."""
    (hl,) = struct.unpack_from("<I", buf, 0)
    header = json.loads(buf[4:4 + hl].decode())
    off = 4 + hl
    k, m = header["k"], header["m"]
    coords = np.frombuffer(buf, np.uint16, k * 3, off).reshape(k, 3); off += k * 6
    species = np.frombuffer(buf, np.uint8, k, off); off += k
    if header.get("labels"):
        labels = np.frombuffer(buf, np.uint32, k, off); off += k * 4
    else:
        labels = np.zeros(k, np.uint32)
    soil = np.frombuffer(buf, np.uint16, m * 3, off).reshape(m, 3); off += m * 6
    n = header["n"]
    for key in ("stone_h", "soil_h", "water_h"):
        if header.get("maps", {}).get(key):
            header[key] = np.frombuffer(buf, np.uint16, n * n, off).reshape(n, n); off += n * n * 2
    return header, coords, species, labels, soil


class WebViewer:
    MAX_N = 256          # выше этого куб не даём создать из браузера

    def __init__(self, engine: Engine, fps=25.0, components_hz=2.0, max_n=256):
        self.engine = engine
        self.MAX_N = int(max_n)
        self.fps = fps                 # верхний предел снимков в секунду
        self.components_hz = components_hz   # как часто пересчитывать организмы
        self.loop = None
        self.clients = set()
        self.latest = None
        self._new = asyncio.Event()
        self.mapper = SoundMapper()
        self.latest_sound = None
        self._sent = {}            # что клиентам уже отправлено (рельеф, имена)
        engine.on_snapshot(self._on_snapshot)

    # вызывается из потока симуляции
    def _on_snapshot(self, snap):
        try:
            self.latest_sound = self.mapper.map(snap)
        except Exception:               # звук не должен ронять симуляцию
            self.latest_sound = None
        self.latest = snap
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._new.set)

    async def broadcaster(self):
        """Кадры уходят не чаще fps и только зрителям. Если движок не
        публикует сам (snapshot_every=0), снимок делаем здесь, в пуле потоков,
        чтобы ни симуляция, ни event loop не ждали разметки организмов."""
        pull = self.engine.snapshot_every <= 0
        last_gen = -1
        every = max(1, int(round(self.fps / max(self.components_hz, 1e-6))))
        tick = 0
        while True:
            if pull:
                # если прошлый снимок был долгим, снимаем реже: иначе очередь
                # снимков забивает и симуляцию, и сервер (так вешался большой куб)
                slow = self.engine.snapshot_seconds
                await asyncio.sleep(max(1.0 / self.fps, min(slow * 1.5, 5.0)))
                if not self.clients or self.engine.gen == last_gen or self.engine.busy:
                    continue
                last_gen = self.engine.gen
                tick += 1
                # разметка организмов дороже самой геометрии — делаем её реже
                comps = (tick % every == 0)
                self.engine.busy = True
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: self.engine.publish(force=True, components=comps))
                finally:
                    self.engine.busy = False
            else:
                await self._new.wait()
                self._new.clear()
            snap = self.latest
            if snap is None or not self.clients:
                continue
            # Кодирование (включая json.dumps заголовка) — В ПОТОКЕ: на
            # большом мире это десятки мс, и в цикле событий оно душило сервер
            data = await asyncio.get_running_loop().run_in_executor(
                None, lambda: encode_snapshot(snap, sound=self.latest_sound, sent=self._sent))
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_bytes(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

    async def ws_handler(self, request):
        from aiohttp import web, WSMsgType
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024)
        await ws.prepare(request)
        snap = self.latest or self.engine.publish(force=True)
        snap.config_json = self._config_json()
        await ws.send_bytes(encode_snapshot(snap, first=True, sound=self.latest_sound))
        self._sent = {}            # новый зритель — следующий общий кадр полный
        self.clients.add(ws)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    cmd = json.loads(msg.data)
                except Exception as e:
                    await ws.send_str(json.dumps({"error": f"не разобрал команду: {e}"}))
                    continue
                # ВАЖНО: команды исполняем в отдельном потоке. Пересев большого
                # мира занимает секунды, и в цикле сервера он вешал ВСЁ — ни
                # страница, ни другие клиенты не отвечали (наступали).
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, self.handle, cmd)
                except Exception as e:
                    await ws.send_str(json.dumps({"error": str(e)}))
        finally:
            self.clients.discard(ws)
        return ws

    def handle(self, cmd):
        e = self.engine
        c = cmd.get("cmd")
        if c == "pause":
            e.pause()
        elif c == "resume":
            e.resume()
        elif c == "step":
            e.step_once()
        elif c == "rate":
            e.set_rate(float(cmd.get("value", 0)))
        elif c == "reset":
            e.set_world(seed_world=int(cmd.get("seed_world", e.cfg.seed_world)),
                        seed_mut=int(cmd.get("seed_mut", e.cfg.seed_mut)))
            e.reset()
        elif c == "snapshot":
            e.publish(force=True)
        elif c == "reset_sound":
            self.mapper = SoundMapper()
        elif c == "genomes":
            e.set_genomes(cmd["value"], ids=cmd.get("ids"))
            self._push_config()
        elif c == "randomize":
            e.randomize(seed=cmd.get("seed"))
            self.mapper = SoundMapper()
            self._push_config()
        elif c == "restart":
            e.reset()
            self.mapper = SoundMapper()
            self._push_config()
        elif c == "engine":
            name = str(cmd.get("value"))
            n = int(cmd.get("n", e.cfg.n))
            if n > self.MAX_N:
                raise ValueError(f"предел {self.MAX_N}³")
            rules = get_rules(name)
            e.switch_rules(name, rules.Config(n=n))
            self.mapper = SoundMapper()
            self._push_config()
        elif c == "world":
            params = dict(cmd.get("value") or {})
            n = int(params.get("n", e.cfg.n))
            if n > self.MAX_N:
                raise ValueError(f"куб {n}³ не поднять: предел {self.MAX_N}³ "
                                 f"(память и скорость). Уменьши размер.")
            if n < 16:
                raise ValueError("куб меньше 16³ бессмысленен")
            reseed = bool(cmd.get("reseed", True))
            e.set_world(**params)
            if reseed:
                e.reset()
            self.mapper = SoundMapper()
            self._push_config()
        elif c == "fork":
            # ответвить новый вид от живущего, не трогая родителя
            sid, k = e.fork_species(int(cmd["id"]), cmd["value"],
                                    share=float(cmd.get("share", 0.3)))
            self._push_config()
            return {"forked": sid, "cells": k}
        elif c == "seeding":
            # кем заселять мир и нужен ли повторный засев
            v = dict(cmd.get("value") or {})
            params = {}
            if "start_species" in v:
                params["start_species"] = tuple(int(x) for x in v["start_species"])
            for k in ("reseed", "reseed_on_extinction"):
                if k in v:
                    params[k] = bool(v[k])
            for k in ("reseed_every", "reseed_count"):
                if k in v:
                    params[k] = max(1, int(v[k]))
            if params.get("start_species") == () and "start_species" in params:
                raise ValueError("выбери хотя бы один стартовый вид")
            e.set_world(**params)
            if bool(cmd.get("restart", False)):
                e.reset()
                self.mapper = SoundMapper()
            self._push_config()
        elif c == "config":
            self._push_config()
        else:
            raise ValueError(f"неизвестная команда {c!r}")

    def _config_json(self):
        """Конфиг движка + то, что знает о движке оркестровка: умеет ли он
        ответвлять виды и как объясняются его гены (для лаборатории генома)."""
        rules = self.engine.rules
        j = rules.to_json(self.engine.cfg, self.engine.state)
        j["can_fork"] = bool(getattr(rules, "can_fork", False))
        j["gene_docs"] = rules.gene_docs()
        return j

    def _push_config(self):
        """Разослать клиентам актуальный конфиг (геномы, мир) отдельным
        текстовым сообщением — он меняется редко и в бинарный кадр не входит."""
        if self.loop is None:
            return
        payload = json.dumps({"config": self._config_json()}, ensure_ascii=False)

        async def send():
            for ws in list(self.clients):
                try:
                    await ws.send_str(payload)
                except Exception:
                    self.clients.discard(ws)
        asyncio.run_coroutine_threadsafe(send(), self.loop)

    def make_app(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_get("/ws", self.ws_handler)
        async def index(request):
            return web.FileResponse(STATIC / "index.html")
        app.router.add_get("/", index)
        app.router.add_static("/static", STATIC)

        async def on_startup(app):
            self.loop = asyncio.get_running_loop()
            app["bcast"] = asyncio.create_task(self.broadcaster())
        app.on_startup.append(on_startup)
        return app


def serve(cfg=None, use_gpu=False, host="0.0.0.0", port=8765, rate=0.0,
          snapshot_every=0, components=True, autostart=True, fps=25.0,
          components_hz=2.0, yield_ms=0.5, max_n=256, max_cells=400_000,
          rules="ecology"):
    """Поднять движок в фоновом потоке и веб-сервер в текущем."""
    from aiohttp import web
    engine = Engine(cfg, use_gpu=use_gpu, rate=rate,
                    snapshot_every=snapshot_every, components=components,
                    yield_ms=yield_ms, max_cells=max_cells, rules=rules)
    if not autostart:
        engine.pause()
    viewer = WebViewer(engine, fps=fps, components_hz=components_hz, max_n=max_n)
    th = threading.Thread(target=engine.run, daemon=True, name="life-cube-sim")
    th.start()
    print(f"life-cube web viewer: http://{host}:{port}/  "
          f"(движок {engine.rules.name}, куб {engine.cfg.n}³, {'GPU' if engine.on_gpu else 'CPU'})", flush=True)
    try:
        web.run_app(viewer.make_app(), host=host, port=port, print=None)
    finally:
        engine.stop()
