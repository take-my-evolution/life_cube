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
        "biomass": getattr(snap, "biomass", None),
        "organisms": getattr(snap, "organisms", None),
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
        for key in ("stone_h", "soil_h", "water_h", "log_h"):
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
    for key in ("stone_h", "soil_h", "water_h", "log_h"):
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
    for key in ("stone_h", "soil_h", "water_h", "log_h"):
        if header.get("maps", {}).get(key):
            header[key] = np.frombuffer(buf, np.uint16, n * n, off).reshape(n, n); off += n * n * 2
    return header, coords, species, labels, soil


class WebViewer:
    MAX_N = 256          # выше этого куб не даём создать из браузера

    def __init__(self, engine: Engine = None, fps=25.0, components_hz=2.0, max_n=256,
                 rules="slope", cfg=None, sim_kw=None):
        # Симуляция — управляемый ресурс, а не то, что живёт всегда: пока она
        # остановлена, `self.engine` равен None, мира нет и видеопамять не
        # занята. «Рецепт» мира (движок + Config) живёт ОТДЕЛЬНО от самого
        # мира: Config ничего не выделяет, куб выделяет только init_state
        # внутри Engine. Поэтому панель конструктора работает и на
        # остановленной симуляции, а Запустить собирает мир по этому рецепту.
        self.engine = engine
        self.rules_name = engine.rules.name if engine is not None else str(rules)
        self.cfg = engine.cfg if engine is not None else cfg
        self._sim_kw = dict(sim_kw or {})
        self._sim_thread = None     # поток симуляции, если его завели МЫ
        self._simlock = threading.Lock()
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
        self._heavy_task = None    # фоновая разметка организмов (см. broadcaster)
        if engine is not None:
            engine.on_snapshot(self._on_snapshot)

    # --- жизненный цикл симуляции -------------------------------------------
    @property
    def running(self):
        return self.engine is not None

    def sim_state(self):
        e = self.engine
        return {"running": e is not None,
                "engine": self.rules_name,
                "n": int(self.cfg.n) if self.cfg is not None else 0,
                "gpu": bool(e.on_gpu) if e is not None else None,
                "gen": int(e.gen) if e is not None else 0}

    def start_sim(self):
        """Собрать мир и запустить поток симуляции. Идемпотентно: одновременно
        живёт РОВНО ОДНА симуляция — второй вызов ничего не делает."""
        with self._simlock:
            if self.engine is not None:
                return self.sim_state()
            if self.cfg is None:
                self.cfg = get_rules(self.rules_name).Config()
            eng = Engine(self.cfg, rules=self.rules_name, **self._sim_kw)
            eng.on_snapshot(self._on_snapshot)
            self.engine = eng
            self.mapper = SoundMapper()
            self._sent = {}
            th = threading.Thread(target=eng.run, daemon=True, name="life-cube-sim")
            self._sim_thread = th
            th.start()
        self._push_sim()
        self._push_config()
        return self.sim_state()

    def stop_sim(self):
        """Остановить симуляцию и отпустить мир вместе с видеопамятью.
        Настройки (движок, Config с геномами) переживают остановку — Запустить
        соберёт мир по ним же."""
        with self._simlock:
            eng, th = self.engine, self._sim_thread
            if eng is None:
                return self.sim_state()
            self.rules_name = eng.rules.name
            self.cfg = eng.cfg
            self.engine = None
            self._sim_thread = None
            eng.stop()
        if th is not None:
            th.join(timeout=3.0)
        eng.release()
        self.latest = None
        self.latest_sound = None
        self._sent = {}
        self._push_sim()
        return self.sim_state()

    def _push_sim(self):
        """Разослать зрителям состояние симуляции (идёт / остановлена)."""
        if self.loop is None or not self.clients:
            return
        msg = json.dumps({"sim": self.sim_state()}, ensure_ascii=False)

        async def send():
            for ws in list(self.clients):
                try:
                    await ws.send_str(msg)
                except Exception:
                    self.clients.discard(ws)
        try:
            asyncio.run_coroutine_threadsafe(send(), self.loop)
        except Exception:
            pass

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
        чтобы ни симуляция, ни event loop не ждали разметки организмов.

        Разметка организмов — самая дорогая часть кадра: scipy.label по всему
        кубу, отдельно на каждый вид. На населённом мире (десятки тысяч живых
        клеток) это ~80мс против ~7мс на голую геометрию — на порядок дороже.
        Раньше её считали ПРЯМО в этом цикле раз в components_hz — и каждый
        такой кадр рендер стопорился на десятки-сотни миллисекунд (рывок), а
        звук (голоса берутся из организмов, snapshot.describe_components)
        разом перескакивал на новый набор нот ровно с той же периодичностью —
        отсюда «нота на разной высоте через равный промежуток» (наступали).
        Теперь разметка считается в фоновой задаче, не блокируя выдачу
        кадров: геометрия идёт каждый тик, организмы (а с ними и голоса)
        подтягиваются, когда фон досчитает — обычно за один-два кадра, без
        видимого стопора. Ценой того, что кадр с организмами может на пару
        поколений отстать от геометрии — на населении, которое меняется на
        доли процента за 80мс, это незаметно."""
        last_gen = -1
        last_sent = None
        every = max(1, int(round(self.fps / max(self.components_hz, 1e-6))))
        tick = 0
        while True:
            e = self.engine
            # симуляция остановлена: мира нет, кадрам взяться неоткуда —
            # просто ждём, пока её запустят из браузера
            if e is None:
                await asyncio.sleep(0.2)
                last_gen = -1
                continue
            if e.snapshot_every <= 0:
                await asyncio.sleep(1.0 / self.fps)
                e = self.engine
                if e is None:
                    continue
                if self.clients and not e.busy and e.gen != last_gen:
                    last_gen = e.gen
                    tick += 1
                    e.busy = True
                    try:
                        await asyncio.get_running_loop().run_in_executor(
                            None, lambda: e.publish(force=True, components=False))
                    finally:
                        e.busy = False
                    # организмы — своим фоновым темпом, не в этом такте
                    if (tick % every == 0
                            and (self._heavy_task is None or self._heavy_task.done())):
                        self._heavy_task = asyncio.ensure_future(self._recompute_components())
            else:
                await self._new.wait()
                self._new.clear()
            snap = self.latest
            # ничего нового не появилось (симуляция медленнее fps, а фон ещё
            # не досчитал) — незачем кодировать и слать тот же кадр повторно
            if snap is None or not self.clients or snap is last_sent:
                continue
            last_sent = snap
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

    async def _recompute_components(self):
        """Фоновая разметка организмов — см. broadcaster(). Ошибку глотаем:
        одна неудачная разметка не должна ронять поток кадров, следующая
        попытка придёт через components_hz."""
        e = self.engine
        if e is None:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: e.publish(force=True, components=True))
        except Exception:
            pass

    async def ws_handler(self, request):
        from aiohttp import web, WSMsgType
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024)
        await ws.prepare(request)
        # состояние симуляции — первым делом: на остановленной симуляции кадров
        # не будет вовсе, и без этого зритель не узнал бы, что происходит
        await ws.send_str(json.dumps({"sim": self.sim_state()}, ensure_ascii=False))
        e = self.engine
        snap = self.latest or (e.publish(force=True) if e is not None else None)
        if snap is not None:
            snap.config_json = self._config_json()
            await ws.send_bytes(encode_snapshot(snap, first=True, sound=self.latest_sound))
        else:
            # мира нет, но панель конструктора нужна и на остановленной
            # симуляции: по этому же рецепту её и запустят
            cj = self._config_json()
            if cj:
                await ws.send_str(json.dumps({"config": cj}, ensure_ascii=False))
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
                    out = await asyncio.get_running_loop().run_in_executor(
                        None, self.handle, cmd)
                    # команды, которым есть что сказать в ответ (например форк:
                    # какой id получился), отвечают словарём — шлём его автору
                    if isinstance(out, dict):
                        await ws.send_str(json.dumps({"ok": out}, ensure_ascii=False))
                except Exception as e:
                    await ws.send_str(json.dumps({"error": str(e)}))
        finally:
            self.clients.discard(ws)
        return ws

    def handle(self, cmd):
        c = cmd.get("cmd")
        # --- жизненный цикл: единственные команды, которым мир не нужен -----
        if c == "sim_start":
            return self.start_sim()
        if c == "sim_stop":
            return self.stop_sim()
        if c == "sim_state":
            return self.sim_state()
        e = self.engine
        if e is None:
            # Пока симуляция остановлена, менять можно только РЕЦЕПТ мира —
            # это чистая правка Config, она ничего не выделяет. Всё остальное
            # требует живого мира и честно об этом говорит.
            if c == "engine":
                name = str(cmd.get("value"))
                n = int(cmd.get("n", self.cfg.n if self.cfg is not None else 128))
                if n > self.MAX_N:
                    raise ValueError(f"предел {self.MAX_N}³")
                rules = get_rules(name)
                self.rules_name = name
                self.cfg = rules.Config(n=n)
                self._push_config()
                return {"engine": name, "n": n, "running": False}
            if c == "world" and self.cfg is not None:
                for k, v in dict(cmd.get("value") or {}).items():
                    if hasattr(self.cfg, k) and k != "genomes":
                        setattr(self.cfg, k, type(getattr(self.cfg, k))(v))
                self._push_config()
                return {"world": True, "running": False}
            if c == "config":
                self._push_config()
                return {"running": False}
            raise ValueError("симуляция остановлена — нажми «Запустить»")
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
            # рецепт вьюера должен смотреть на ТОТ ЖЕ Config, что и движок:
            # switch_rules подменяет e.cfg новым объектом, и без этой строки
            # self.cfg оставался прежним — Запустить после Остановить собрал
            # бы мир по устаревшему движку и размеру
            self.rules_name, self.cfg = e.rules.name, e.cfg
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
            self.cfg = e.cfg
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
        e = self.engine
        rules = e.rules if e is not None else get_rules(self.rules_name)
        cfg = e.cfg if e is not None else self.cfg
        if cfg is None:
            return None
        try:
            # на остановленной симуляции состояния мира нет — движки, которые
            # умеют обойтись одним Config, всё равно отдадут панель
            j = rules.to_json(cfg, e.state if e is not None else None)
        except Exception:
            return None
        j["can_fork"] = bool(getattr(rules, "can_fork", False))
        j["gene_docs"] = rules.gene_docs()
        # Список движков раньше уходил ТОЛЬКО в первом бинарном кадре. На
        # остановленной симуляции кадров нет вовсе (v0.10.0), и выпадающий
        # список движков оставался пустым навсегда — выбрать движок было
        # нечем. Шлём его вместе с конфигом: конфиг приходит и без мира.
        j["engines"] = list_engines()
        # предел размера куба у ЭТОГО сервера (--max-n): без него ползунок в
        # браузере доходил до 256, сервер молча отказывал, и со стороны это
        # выглядело как «увеличение куба не работает»
        j["max_n"] = int(self.MAX_N)
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
          snapshot_every=0, components=True, autostart=False, fps=25.0,
          components_hz=2.0, yield_ms=0.5, max_n=256, max_cells=400_000,
          rules="slope"):
    """Поднять веб-сервер. Мир по умолчанию НЕ создаётся: сервис поднимается
    мгновенно и не держит видеокарту, пока симуляцию не запустят из браузера
    (`autostart=True` / `--autostart` возвращает старое поведение)."""
    from aiohttp import web
    sim_kw = dict(use_gpu=use_gpu, rate=rate, snapshot_every=snapshot_every,
                  components=components, yield_ms=yield_ms, max_cells=max_cells)
    viewer = WebViewer(None, fps=fps, components_hz=components_hz, max_n=max_n,
                       rules=rules, cfg=cfg, sim_kw=sim_kw)
    if autostart:
        viewer.start_sim()
    n = viewer.cfg.n if viewer.cfg is not None else get_rules(rules).Config().n
    print(f"life-cube web viewer: http://{host}:{port}/  "
          f"(движок {rules}, куб {n}³, симуляция "
          f"{'запущена' if autostart else 'остановлена — запусти из браузера'})", flush=True)
    try:
        web.run_app(viewer.make_app(), host=host, port=port, print=None)
    finally:
        viewer.stop_sim()
