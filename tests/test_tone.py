"""Основной тон из населения: он обязан ДВИГАТЬСЯ и знать свою шкалу.

Жалоба: «тон застывает, и почему шкала от 20 до 220».

Застывал он не по ошибке: октава считалась ступенчато (три корзины по
населению), и всё отображение имело от силы полтора десятка возможных
значений. Тон подползал к одному из них и стоял там, хотя население
колебалось в разы.
"""

import numpy as np
import pytest

from life_cube.engine import Engine
from life_cube.engines import get_rules
from life_cube.sound.features import SoundMapper


class FakeSnap:
    def __init__(self, pops):
        self.pops = list(pops)


def settle(m, total, steps=400):
    for _ in range(steps):
        hz = m.base_from_world(FakeSnap([total, 0, 0, 0, 0]))
    return hz


def test_tone_follows_the_population():
    """Вдвое гуще населённый мир звучит заметно выше — и наоборот."""
    m = SoundMapper()
    settle(m, 3000)                      # мир обжился, опора установилась
    low = settle(m, 1000, steps=60)
    m2 = SoundMapper()
    settle(m2, 3000)
    high = settle(m2, 9000, steps=60)
    assert high > low * 1.15, (low, high)


def test_tone_does_not_freeze_while_the_world_keeps_changing():
    """Ядро жалобы: население ходит в разы, а тон стоит.

    Держим доминирующий вид неизменным и гоняем население внутри ОДНОЙ старой
    корзины (2000..40000). Прежнее отображение давало там ровно одну цель, тон
    сходился к ней и замирал; новое ведёт тон непрерывно по логарифму
    населения.
    """
    m = SoundMapper()
    settle(m, 3000)                      # опора установилась
    wave = []
    for i in range(240):
        total = int(2200 + 1600 * (i % 60) / 59.0)     # 2200 → 3800 и снова
        wave.append(m.base_from_world(FakeSnap([total, 0, 0, 0, 0])))
    tail = np.array(wave[120:])
    assert tail.max() - tail.min() > 2.0, f"тон замер на {tail[-1]:.2f} Гц"


def test_the_tone_still_moves_on_a_living_world():
    """И то же самое на настоящем мире, без искусственной волны."""
    R = get_rules("slope")
    e = Engine(cfg=R.make_config(n=40, seed_world=7), rules="slope", components=False)
    m = SoundMapper()
    tones = []
    for g in range(1, 601):
        e.advance()
        if g % 10:
            continue
        tones.append(m.base_from_world(e.publish(force=True, components=False)))
    tail = np.array(tones[len(tones) // 2:])
    assert tail.max() - tail.min() > 3.0, f"тон замер на {tail[-1]:.1f} Гц"


def test_the_scale_is_configurable_and_respected():
    """Границы шкалы задаются, тон из них не выходит и едет клиенту."""
    m = SoundMapper(base_min=110.0, base_max=440.0)
    for total in (10, 500, 5000, 200000):
        hz = settle(m, total, steps=200)
        assert 110.0 <= hz <= 440.0, (total, hz)
    snap = _slope_snapshot()
    frame = m.map(snap)
    assert frame.base_min == 110.0 and frame.base_max == 440.0


def test_the_scale_does_not_depend_on_world_size():
    """Опора считается по самому миру, поэтому маленький и большой куб дают
    сопоставимый тон — иначе на 128³ он всегда упирался бы в потолок."""
    tones = []
    for total in (400, 40000):
        m = SoundMapper()
        tones.append(settle(m, total))
    assert abs(tones[0] - tones[1]) < 1.0, tones


def _slope_snapshot():
    R = get_rules("slope")
    e = Engine(cfg=R.make_config(n=32, seed_world=3), rules="slope", components=False)
    for _ in range(30):
        e.advance()
    return e.publish(force=True, components=False)


def test_server_accepts_a_new_scale_without_touching_the_world():
    from life_cube.viewers.web.server import WebViewer

    R = get_rules("slope")
    v = WebViewer(None, rules="slope", cfg=R.make_config(n=32))
    v.start_sim()
    try:
        for _ in range(20):
            v.engine.advance()
        gen, k = v.engine.gen, int((np.asarray(v.engine.state["species"]) > 0).sum())
        ans = v.handle({"cmd": "sound", "value": {"base_min": 200, "base_max": 800}})
        assert ans["sound"]["base_min"] == 200 and ans["sound"]["base_max"] == 800
        assert v.mapper.base_min == 200 and v.mapper.base_max == 800
        assert v.engine.gen == gen, "мир пересоздался от настройки звука"
        assert int((np.asarray(v.engine.state["species"]) > 0).sum()) == k
        with pytest.raises(ValueError):
            v.handle({"cmd": "sound", "value": {"base_min": 800, "base_max": 200}})
    finally:
        v.stop_sim()
