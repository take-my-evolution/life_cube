"""Engine — управляемый цикл симуляции.

Ничего не знает о рендерах: рендер (matplotlib, web, что угодно) подписывается
на снимки через on_snapshot и дёргает pause()/resume()/step_once()/set_rate().
Сам Engine синхронный; веб-сервер крутит его в отдельном потоке.
"""

import threading
import time

from .backend import get_backend
from .config import Config
from .sim import init_state
from .snapshot import Tracker, make_snapshot
from .step import step


class Engine:
    def __init__(self, cfg: Config, use_gpu=False, rate=10.0,
                 snapshot_every=1, components=True):
        self.cfg = cfg
        self.xp, self.correlate, self.on_gpu = get_backend(use_gpu)
        self.state, self.relief = init_state(cfg, self.xp)
        self.gen = 0
        self.rate = float(rate)          # целевых поколений/с; <=0 — без предела
        self.snapshot_every = int(snapshot_every)
        self.components = components
        self.tracker = Tracker() if components else None
        self.paused = False
        self.running = False
        self.hist = []
        self.listeners = []
        self._step_request = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.last_snapshot = None
        self.measured_rate = 0.0
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

    def set_rate(self, rate):
        self.rate = float(rate)
        self._wake.set()

    def on_snapshot(self, fn):
        self.listeners.append(fn)

    def reset(self, cfg: Config = None):
        with self._lock:
            if cfg is not None:
                self.cfg = cfg
            self.state, self.relief = init_state(self.cfg, self.xp)
            self.gen = 0
            self.hist = []
            self.tracker = Tracker() if self.components else None
        self.publish(force=True)

    # --- шаг ----------------------------------------------------------------
    def advance(self):
        with self._lock:
            pops = step(self.state, self.cfg, self.xp, self.correlate, self.gen)
            self.gen += 1
            self.hist.append(pops)
        return pops

    def publish(self, force=False):
        if not force and self.gen % self.snapshot_every:
            return None
        snap = make_snapshot(self.state, self.gen, self.cfg, self.tracker,
                             with_components=self.components)
        snap.relief = self.relief
        snap.hist = list(self.hist)
        snap.rate = self.rate
        snap.measured_rate = self.measured_rate
        snap.paused = self.paused
        self.last_snapshot = snap
        for fn in self.listeners:
            fn(snap)
        return snap

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
        finally:
            self.running = False

    def stop(self):
        self.running = False
        self._wake.set()
