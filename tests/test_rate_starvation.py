"""Снимок не должен голодать под замком симуляции.

Жалоба: «поставил 10 поколений в секунду — всё зависло, интерфейс не
отвечает». Симуляция при этом шла: поколения тикали, а браузер стоял на
давнем кадре. Причина не в скорости как таковой, а в том, что шаг оказался
ДОЛЬШЕ бюджета скорости: ждать было нечего, цикл забирал замок обратно
мгновенно, и наблюдатель со своим снимком не мог вклиниться вовсе.
"""

import threading
import time

import pytest

from life_cube.engine import Engine
from life_cube.engines import get_rules

STEP = 0.12          # шаг заведомо дольше бюджета 10 пок/с


class SlowRules:
    """Движок «склон» с искусственно долгим шагом."""

    def __init__(self):
        self._r = get_rules("slope")

    def __getattr__(self, k):
        return getattr(self._r, k)

    def step(self, *a, **kw):
        time.sleep(STEP)
        return self._r.step(*a, **kw)


@pytest.fixture
def engine():
    rules = SlowRules()
    e = Engine(cfg=rules.make_config(n=32), rules=rules, rate=10.0,
               snapshot_every=0, components=False)
    th = threading.Thread(target=e.run, daemon=True)
    th.start()
    time.sleep(0.3)
    yield e
    e.stop()
    th.join(timeout=5)


def test_snapshot_is_not_starved_by_a_slow_step(engine):
    worst = 0.0
    for _ in range(6):
        t = time.perf_counter()
        engine.publish(force=True, components=False)
        worst = max(worst, time.perf_counter() - t)
        time.sleep(0.01)
    # один шаг подождать нормально; секунды — это и есть «интерфейс завис»
    assert worst < 4 * STEP, f"снимок ждал {worst * 1000:.0f} мс"


def test_simulation_still_runs_while_yielding(engine):
    gen0 = engine.gen
    for _ in range(4):
        engine.publish(force=True, components=False)
        time.sleep(0.05)
    assert engine.gen > gen0, "уступая снимку, симуляция встала совсем"
