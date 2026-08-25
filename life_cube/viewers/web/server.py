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
"""

import asyncio
import json
import pathlib
import struct
import threading

import numpy as np

from ...config import Config, SPECIES_NAMES
from ...engine import Engine
from ...snapshot import Snapshot
from ...sound import SoundMapper

STATIC = pathlib.Path(__file__).parent / "static"


def encode_snapshot(snap: Snapshot, first=False, sound=None) -> bytes:
    header = {
        "sound": sound.to_dict() if sound is not None else None,
        "gen": snap.gen, "n": snap.n, "k": int(len(snap.coords)),
        "m": int(len(snap.soil_coords)) if snap.soil_coords is not None else 0,
        "pops": snap.pops,
        "rate": getattr(snap, "rate", 0.0),
        "measured_rate": round(getattr(snap, "measured_rate", 0.0), 2),
        "paused": getattr(snap, "paused", False),
        "components": [[c.cid, c.species, c.size, *c.center, c.zmin, c.zmax, c.born]
                       for c in snap.components],
        "hist_tail": [list(map(int, h)) for h in getattr(snap, "hist", [])[-400:]],
    }
    if first:
        header["species_names"] = list(SPECIES_NAMES)
        relief = getattr(snap, "relief", None)
        header["relief"] = relief.astype(int).tolist() if relief is not None else None
    hb = json.dumps(header, ensure_ascii=False).encode()
    soil = snap.soil_coords if snap.soil_coords is not None else np.zeros((0, 3), np.uint16)
    parts = [struct.pack("<I", len(hb)), hb,
             np.ascontiguousarray(snap.coords, dtype=np.uint16).tobytes(),
             np.ascontiguousarray(snap.species, dtype=np.uint8).tobytes(),
             np.ascontiguousarray(snap.labels, dtype=np.uint32).tobytes(),
             np.ascontiguousarray(soil, dtype=np.uint16).tobytes()]
    return b"".join(parts)


def decode_snapshot(buf: bytes):
    """Для тестов: обратное преобразование."""
    (hl,) = struct.unpack_from("<I", buf, 0)
    header = json.loads(buf[4:4 + hl].decode())
    off = 4 + hl
    k, m = header["k"], header["m"]
    coords = np.frombuffer(buf, np.uint16, k * 3, off).reshape(k, 3); off += k * 6
    species = np.frombuffer(buf, np.uint8, k, off); off += k
    labels = np.frombuffer(buf, np.uint32, k, off); off += k * 4
    soil = np.frombuffer(buf, np.uint16, m * 3, off).reshape(m, 3)
    return header, coords, species, labels, soil


class WebViewer:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.loop = None
        self.clients = set()
        self.latest = None
        self._new = asyncio.Event()
        self.mapper = SoundMapper()
        self.latest_sound = None
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
        while True:
            await self._new.wait()
            self._new.clear()
            snap = self.latest
            if snap is None or not self.clients:
                continue
            data = encode_snapshot(snap, sound=self.latest_sound)
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
        await ws.send_bytes(encode_snapshot(snap, first=True, sound=self.latest_sound))
        self.clients.add(ws)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    self.handle(json.loads(msg.data))
                except Exception as e:      # плохая команда не роняет сервер
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
            old = e.cfg
            cfg = Config(n=old.n, gens=old.gens,
                         seed_world=int(cmd.get("seed_world", old.seed_world)),
                         seed_mut=int(cmd.get("seed_mut", old.seed_mut)),
                         seed_density=old.seed_density, genomes=old.genomes)
            e.reset(cfg)
        elif c == "snapshot":
            e.publish(force=True)
        elif c == "reset_sound":
            self.mapper = SoundMapper()
        else:
            raise ValueError(f"неизвестная команда {c!r}")

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


def serve(cfg: Config, use_gpu=False, host="0.0.0.0", port=8765, rate=10.0,
          snapshot_every=1, components=True, autostart=True):
    """Поднять движок в фоновом потоке и веб-сервер в текущем."""
    from aiohttp import web
    engine = Engine(cfg, use_gpu=use_gpu, rate=rate,
                    snapshot_every=snapshot_every, components=components)
    if not autostart:
        engine.pause()
    viewer = WebViewer(engine)
    th = threading.Thread(target=engine.run, daemon=True, name="life-cube-sim")
    th.start()
    print(f"life-cube web viewer: http://{host}:{port}/  "
          f"(куб {cfg.n}³, {'GPU' if engine.on_gpu else 'CPU'})", flush=True)
    try:
        web.run_app(viewer.make_app(), host=host, port=port, print=None)
    finally:
        engine.stop()
