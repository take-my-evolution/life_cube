"""Engine — управляемый цикл симуляции.

Ничего не знает о рендерах: рендер (matplotlib, web, что угодно) подписывается
на снимки через on_snapshot и дёргает pause()/resume()/step_once()/set_rate().
Сам Engine синхронный; веб-сервер крутит его в отдельном потоке.
"""

import threading
import time
from collections import deque

import numpy as np

from .backend import get_backend, to_cpu
from .config import Config
from .engines import get_rules
from .snapshot import Tracker, make_snapshot, describe_components


class Engine:
    # выше этого числа клеток разметка организмов (scipy.label на CPU) стоит
    # секунды и душит и симуляцию, и веб-сервер: отключаем её автоматически
    COMPONENTS_CELL_LIMIT = 6_000_000
    HIST_KEEP = 600          # сколько последних поколений держим поштучно
    HIST_LONG = 400          # столько точек прореженной длинной истории
    HIST_EVERY = 50          # каждое k-е поколение попадает в длинную историю

    def __init__(self, cfg: Config = None, use_gpu=False, rate=10.0,
                 snapshot_every=1, components=True, yield_ms=0.5,
                 max_cells=400_000, rules="ecology"):
        self.rules = get_rules(rules) if isinstance(rules, str) else rules
        self.cfg = cfg if cfg is not None else self.rules.Config()
        self.xp, self.correlate, self.on_gpu = get_backend(use_gpu)
        self.state, self.relief = self.rules.init_state(self.cfg, self.xp)
        self.gen = 0
        self.rate = float(rate)          # целевых поколений/с; <=0 — без предела
        # без предела скорости поток симуляции не отдаёт GIL и душит веб-сервер:
        # уступаем ему немного времени на каждом шаге
        self.yield_ms = float(yield_ms)
        self.snapshot_every = int(snapshot_every)
        self.components = components
        self.tracker = Tracker() if components else None
        self.paused = False
        self.running = False
        # История: последние HIST_KEEP поколений точно + прореженная длинная.
        # Раньше это был безграничный список: на 2 млн поколений он занимал
        # сотни мегабайт и копировался ПОД ЗАМКОМ на каждый кадр (рывки).
        self.hist = deque(maxlen=self.HIST_KEEP)
        self.hist_long = deque(maxlen=self.HIST_LONG)   # каждое HIST_EVERY-е
        self.listeners = []
        self._step_request = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.last_snapshot = None
        self.measured_rate = 0.0
        self.max_cells = int(max_cells)     # столько живых клеток шлём зрителю
        self.snapshot_seconds = 0.0         # сколько занял последний снимок
        self.busy = False                   # снимок уже считается
        self.reseeds = 0                    # сколько раз мир подсевали
        self.last_reseed_gen = None
        self.publish(force=True)

    # --- управление ---------------------------------------------------------
    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self._wake.set()

    def step_once(self):
        with self._lock:
            self._step_request += 1
        self._wake.set()

    def set_genomes(self, genomes, ids=None):
        """Заменить геномы на лету, не трогая мир."""
        with self._lock:
            if ids is not None:
                self.rules.apply_genomes(self.cfg, self.state, genomes, self.xp, ids=ids)
            else:
                self.rules.apply_genomes(self.cfg, self.state, genomes, self.xp)
        self.publish(force=True)

    def fork_species(self, sid, genome, share=0.3):
        """Ответвить новый вид от живущего (движки с динамическими видами).
        -> (новый id, сколько клеток перекрашено)."""
        with self._lock:
            out = self.rules.fork_species(self.cfg, self.state, int(sid), genome,
                                          self.xp, gen=self.gen, share=float(share))
        self.publish(force=True)
        return out

    def randomize(self, seed=None):
        """Случайные гены (по правилам движка) + пересоздание мира."""
        rng = np.random.default_rng(seed)
        with self._lock:
            self.cfg.genomes = self.rules.randomize(self.cfg, rng)
        self.reset()

    def switch_rules(self, name, cfg=None):
        """Сменить движок: новый Config по умолчанию (или переданный) и новый мир."""
        rules = get_rules(name)
        with self._lock:
            self.rules = rules
            self.cfg = cfg if cfg is not None else rules.Config(n=self.cfg.n)
        self.reset()

    def set_world(self, **params):
        """Изменить параметры мира. Те, что влияют на рельеф/засев, требуют
        пересоздания — вызывающий решает через reset()."""
        with self._lock:
            for k, v in params.items():
                if hasattr(self.cfg, k) and k != "genomes":
                    setattr(self.cfg, k, type(getattr(self.cfg, k))(v))
        self.publish(force=True)

    def set_rate(self, rate):
        self.rate = float(rate)
        self._wake.set()

    def on_snapshot(self, fn):
        self.listeners.append(fn)

    def reset(self, cfg=None):
        with self._lock:
            if cfg is not None:
                self.cfg = cfg
            # освобождаем видеопамять прошлого мира до создания нового,
            # иначе на больших кубах два мира не влезают
            self.state = None
            if self.on_gpu:
                try:
                    self.xp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            self.state, self.relief = self.rules.init_state(self.cfg, self.xp)
            self.gen = 0
            self.hist.clear()
            self.hist_long.clear()
            self.reseeds = 0
            self.last_reseed_gen = None
            self.tracker = Tracker() if self.components else None
        self.publish(force=True)

    # --- шаг ----------------------------------------------------------------
    def advance(self):
        with self._lock:
            pops = self.rules.step(self.state, self.cfg, self.xp, self.correlate, self.gen)
            self.gen += 1
            self.hist.append(pops)
            if self.gen % self.HIST_EVERY == 0:
                self.hist_long.append(pops)
        self.maybe_reseed(pops)
        return pops

    # --- повторный засев ----------------------------------------------------
    def maybe_reseed(self, pops):
        """Спасательный круг: вымерший мир больше не остаётся пустым навсегда.

        Включается галочкой (`cfg.reseed`). По умолчанию срабатывает только при
        полном вымирании и не чаще, чем раз в `reseed_every` поколений — иначе
        подсев затирает результат эволюции."""
        cfg = self.cfg
        if not getattr(cfg, "reseed", False) or not getattr(self.rules, "can_seed", False):
            return 0
        every = max(int(getattr(cfg, "reseed_every", 200)), 1)
        last = self.state.get("last_reseed", -10 ** 9) if isinstance(self.state, dict) else 0
        if self.gen - last < every:
            return 0
        extinct = sum(pops) == 0
        if getattr(cfg, "reseed_on_extinction", True) and not extinct:
            return 0
        with self._lock:
            k = self.rules.seed(self.state, cfg, self.xp,
                                self.state.get("rng"), count=None, gen=self.gen)
            self.state["last_reseed"] = self.gen
        if k:
            self.reseeds = getattr(self, "reseeds", 0) + 1
            self.last_reseed_gen = self.gen
        return k

    def publish(self, force=False, components=None):  # noqa: C901
        """Снимок текущего состояния. snapshot_every=0 — из цикла не зовётся,
        снимки делает наблюдатель в своём темпе (см. viewers/web)."""
        if not force and (self.snapshot_every <= 0 or self.gen % self.snapshot_every):
            return None
        # копируем массивы под замком (быстро), тяжёлую разметку делаем без него,
        # чтобы симуляция не ждала CPU
        t_start = time.perf_counter()
        heightmaps = getattr(self.rules, "heightmaps", False)
        with self._lock:
            gen = self.gen
            cpu = {"species": to_cpu(self.state["species"]).copy(),
                   # при картах высот почва не нужна поклеточно
                   "soil": (np.zeros((1, 1, 1), bool) if heightmaps
                            else to_cpu(self.state["soil"]).copy())}
            n_species = self.rules.n_species(self.cfg)
        want = self.components if components is None else (components and self.components)
        if want and self.cfg.n ** 3 > self.COMPONENTS_CELL_LIMIT:
            want = False                    # слишком большой мир — без разметки
        snap = make_snapshot(cpu, gen, self.cfg, self.tracker,
                             with_components=want, n_species=n_species)
        if not want:
            # Полную разметку (scipy.label) на этом такте не считаем — дорого
            # (см. WebViewer.broadcaster). Но список организмов не должен
            # висеть неизменным до следующего тяжёлого пересчёта: тогда
            # размер/положение/громкость голоса (SoundMapper берёт их из
            # snap.components) стоят на месте кадров десять-двадцать подряд
            # и потом разом скачут — на слух это звучит синхронным щелчком
            # на подозрительно ровном интервале components_hz, никак не
            # связанным с тем, что реально непрерывно происходит с
            # клетками (жалоба пользователя после v0.7.3).
            #
            # Дешёвый компромисс: берём СТАРУЮ (с последнего тяжёлого
            # пересчёта) устойчивую id-карту (tracker.prev, тот же плотный
            # массив, что использует сам Tracker для сопоставления кадров)
            # и накладываем её на ТЕКУЩИЕ живые клетки — обычное
            # индексирование массива, на порядки дешевле scipy.label.
            # Организмы, которые с последнего пересчёта выросли за старые
            # границы или родились заново, в эту карту не попадают (id=0)
            # и появятся только на следующем тяжёлом пересчёте — это
            # редкое дискретное событие (действительно новый организм), а
            # не постоянная пульсация. Зато размер/центр уже отслеженных
            # организмов теперь меняются каждый кадр, а не раз в
            # components_hz.
            ids_map = self.tracker.prev if self.tracker is not None else None
            done = False
            if (ids_map is not None and len(snap.coords)
                    and len(snap.coords) <= self.max_cells):
                c = snap.coords.astype(np.intp)
                ids = ids_map[c[:, 0], c[:, 1], c[:, 2]]
                snap.labels = ids.astype(np.uint32)
                # id=0 — клетка сейчас жива, но там, где на прошлой тяжёлой
                # разметке было пусто (новый рост/новый организм): это НЕ
                # организм "номер 0", а "пока неизвестно чей" — учитывать
                # его как один гигантский компонент нельзя (раньше был
                # именно такой баг здесь: несвязанные новые клетки со всего
                # мира схлопывались в один "organism 0" размером с тысячи
                # клеток). Такие клетки просто не попадают в список
                # организмов до следующего тяжёлого пересчёта.
                known = ids > 0
                if known.any():
                    snap.components = describe_components(
                        snap.coords[known], snap.species[known], ids[known],
                        self.tracker.born, max_components=200)
                    done = True
            if not done and self.last_snapshot is not None:
                # id-карты ещё нет (разметки не было ни разу), мир слишком
                # населён для дешёвого пересчёта, или все живые клетки —
                # неопознанный новый рост: как раньше, переиспользуем
                # прошлую разметку целиком
                snap.components = self.last_snapshot.components
        # движки с картами высот отдают подложку картами, а не поклеточно
        if getattr(self.rules, "heightmaps", False):
            with self._lock:
                snap.stone_h = to_cpu(self.state["stone_h"]).astype("uint16")
                snap.soil_h = to_cpu(self.state["soil_h"]).astype("uint16")
                w = self.state.get("water_h")
                snap.water_h = to_cpu(w).astype("uint16") if w is not None else None
            snap.soil_coords = None
            self.relief = snap.stone_h
        snap.relief = self.relief
        # график строится по прореженной истории + хвосту: это ~200 точек
        # вместо тысяч, кадр не раздувается
        snap.hist = self.history_series()
        snap.species_names = self.rules.species_names(self.cfg, self.state) \
            if getattr(self.rules, "dynamic_species", False) else self.rules.species_names(self.cfg)
        snap.species_colors = self.rules.species_colors(self.cfg, self.state) \
            if getattr(self.rules, "dynamic_species", False) else self.rules.species_colors(self.cfg)
        snap.dynamic_species = getattr(self.rules, "dynamic_species", False)
        # для показа прореживаем: миллион кубиков браузеру всё равно не нужен
        k = len(snap.coords)
        if self.max_cells and k > self.max_cells:
            stride = int(k // self.max_cells) + 1
            snap.coords = snap.coords[::stride]
            snap.species = snap.species[::stride]
            snap.labels = snap.labels[::stride]
            snap.stride = stride
        else:
            snap.stride = 1
        snap.rate = self.rate
        snap.measured_rate = self.measured_rate
        snap.paused = self.paused
        snap.components_on = want
        self.snapshot_seconds = time.perf_counter() - t_start
        snap.snapshot_seconds = self.snapshot_seconds   # для диагностики в клиенте
        self.last_snapshot = snap
        for fn in self.listeners:
            fn(snap)
        return snap

    def history_series(self, points=160):
        """Компактная история для графика: длинная прореженная + хвост."""
        long_part = list(self.hist_long)
        tail = list(self.hist)
        if len(tail) > points:
            step_t = len(tail) // points + 1
            tail = tail[::step_t]
        if len(long_part) > points:
            step_l = len(long_part) // points + 1
            long_part = long_part[::step_l]
        return long_part + tail

    def run(self, max_gens=None, stop_event=None):
        """Цикл: держит целевую скорость, уважает паузу и одиночные шаги."""
        self.running = True
        t_prev = None          # момент начала прошлого шага
        ema = None
        try:
            while self.running and not (stop_event and stop_event.is_set()):
                if max_gens is not None and self.gen >= max_gens:
                    break
                want_step = False
                with self._lock:
                    if self._step_request > 0:
                        self._step_request -= 1
                        want_step = True
                if self.paused and not want_step:
                    self._wake.wait(0.1)
                    self._wake.clear()
                    t_prev = None
                    continue
                t0 = time.perf_counter()
                if t_prev is not None and t0 > t_prev:
                    r = 1.0 / (t0 - t_prev)
                    ema = r if ema is None else 0.8 * ema + 0.2 * r
                    self.measured_rate = ema
                t_prev = t0
                self.advance()
                self.publish()
                if self.rate > 0 and not want_step:
                    budget = 1.0 / self.rate - (time.perf_counter() - t0)
                    if budget > 0:
                        self._wake.wait(budget)
                        self._wake.clear()
                elif self.yield_ms > 0:
                    time.sleep(self.yield_ms / 1000.0)
        finally:
            self.running = False

    def stop(self):
        self.running = False
        self._wake.set()
