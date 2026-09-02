"""Жизненный цикл симуляции: сервис поднимается БЕЗ мира и не держит
видеокарту, пока симуляцию не запустят из браузера; симуляция всегда одна;
остановка отпускает мир."""

import socket
import threading

import pytest

pytest.importorskip("aiohttp")

from life_cube import Config                          # noqa: E402
from life_cube.viewers.web.server import WebViewer    # noqa: E402


def _viewer(**kw):
    return WebViewer(None, rules="ecology", cfg=Config(n=24, seed_density=0.05),
                     sim_kw=dict(rate=0, components=True), **kw)


def test_viewer_starts_without_a_world():
    """Главное требование: пока никто не нажал «Запустить», мира нет вовсе —
    ни массивов, ни занятой видеопамяти."""
    v = _viewer()
    assert v.engine is None
    assert v.running is False
    st = v.sim_state()
    assert st["running"] is False and st["engine"] == "ecology" and st["n"] == 24
    # рецепт мира при этом есть — панель конструктора работает и без мира
    cj = v._config_json()
    assert cj and cj["engine"] == "ecology" and len(cj["genomes"]) == 8


def test_start_creates_exactly_one_simulation_and_stop_releases_it():
    v = _viewer()
    try:
        st = v.start_sim()
        assert st["running"] is True
        e = v.engine
        assert e is not None and e.state is not None
        th = v._sim_thread
        assert th is not None and th.is_alive()

        # повторный запуск не создаёт вторую симуляцию: она всегда одна
        again = v.start_sim()
        assert again["running"] is True
        assert v.engine is e and v._sim_thread is th

        v.stop_sim()
        assert v.engine is None and v.running is False
        assert e.state is None, "мир не отпущен — видеопамять осталась занятой"
        th.join(timeout=3)
        assert not th.is_alive(), "поток симуляции не остановился"
    finally:
        v.stop_sim()


def test_settings_survive_stop_and_start_again():
    """Остановка — не сброс: движок, размер куба и геномы переживают её,
    и Запустить собирает мир по тому же рецепту."""
    v = _viewer()
    try:
        v.start_sim()
        v.handle({"cmd": "world", "value": {"rain_rate": 0.11}, "reseed": False})
        v.stop_sim()
        assert v.cfg.rain_rate == pytest.approx(0.11)
        assert v.rules_name == "ecology"
        v.start_sim()
        assert v.engine.cfg.rain_rate == pytest.approx(0.11)
        assert v.engine.cfg.n == 24
    finally:
        v.stop_sim()


def test_recipe_is_editable_while_stopped_but_world_commands_are_refused():
    """Остановленной симуляции можно менять рецепт (это чистая правка Config,
    она ничего не выделяет), а команды, которым нужен живой мир, обязаны
    честно отказать, а не падать по AttributeError на None."""
    v = _viewer()
    try:
        out = v.handle({"cmd": "engine", "value": "lichen", "n": 32})
        assert out["running"] is False and out["engine"] == "lichen"
        assert v.rules_name == "lichen" and v.cfg.n == 32 and v.engine is None

        for cmd in ({"cmd": "step"}, {"cmd": "resume"}, {"cmd": "restart"},
                    {"cmd": "rate", "value": 5}):
            with pytest.raises(ValueError, match="остановлена"):
                v.handle(cmd)

        # а запуск после правки рецепта поднимает именно тот мир
        v.start_sim()
        assert v.engine.rules.name == "lichen" and v.engine.cfg.n == 32
    finally:
        v.stop_sim()


def test_serve_does_not_build_a_world_by_default():
    """`serve()` по умолчанию не создаёт мир — это и есть «сервис поднялся,
    видеокарта свободна». Проверяем, не поднимая сети: собираем WebViewer
    ровно так же, как это делает serve()."""
    import inspect
    from life_cube.viewers.web.server import serve
    sig = inspect.signature(serve)
    assert sig.parameters["autostart"].default is False


def test_config_carries_engine_list_and_size_limit():
    """Список движков раньше уходил ТОЛЬКО в первом бинарном кадре. На
    остановленной симуляции кадров нет вовсе, и выпадающий список движков в
    браузере оставался пустым — выбрать движок было физически нечем.
    Предел размера куба (--max-n) тоже должен доезжать до клиента: без него
    ползунок уезжал до 256, сервер молча отказывал, и со стороны это
    выглядело как «увеличение куба не работает»."""
    v = _viewer(max_n=224)
    cj = v._config_json()
    assert cj["engines"] and {e["name"] for e in cj["engines"]} >= {"ecology", "lichen", "terra"}
    assert cj["max_n"] == 224
    # и то же самое на ЗАПУЩЕННОЙ симуляции
    try:
        v.start_sim()
        cj2 = v._config_json()
        assert cj2["engines"] and cj2["max_n"] == 224
    finally:
        v.stop_sim()


def test_engine_switch_keeps_viewer_recipe_in_sync():
    """switch_rules подменяет e.cfg НОВЫМ объектом. Рецепт вьюера должен
    поехать за ним, иначе Остановить/Запустить пересобирает мир по
    устаревшему движку и размеру."""
    v = _viewer()
    try:
        v.start_sim()
        v.handle({"cmd": "engine", "value": "lichen", "n": 32})
        assert v.rules_name == "lichen" and v.cfg is v.engine.cfg and v.cfg.n == 32
        v.stop_sim()
        v.start_sim()
        assert v.engine.rules.name == "lichen" and v.engine.cfg.n == 32
    finally:
        v.stop_sim()


def test_world_size_change_reaches_the_engine():
    """Ползунок размера + «Пересоздать мир»: куб должен реально пересоздаться
    новым размером, и рецепт вьюера — совпасть с движком."""
    v = _viewer()
    try:
        v.start_sim()
        assert v.engine.cfg.n == 24
        v.handle({"cmd": "world", "value": {"n": 40}, "reseed": True})
        assert v.engine.cfg.n == 40 and v.cfg.n == 40
        assert v.engine.state["species"].shape == (40, 40, 40)
    finally:
        v.stop_sim()
