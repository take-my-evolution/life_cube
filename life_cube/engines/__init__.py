"""Реестр движков (наборов правил).

Каждый движок — модуль с объектом `RULES`, реализующим интерфейс `Rules`:
свой Config, свой мир, свой шаг, свои гены. Оркестровка (Engine), снимки,
рендеры и звук — общие: они видят только массив `species` (0 = пусто) плюс
имена и цвета видов, которые движок отдаёт сам.

Добавить движок = положить модуль сюда, дать ему `RULES` и описать его в
docs/engines/<name>.md. Ниже — список известных; порядок = порядок в UI.
"""

import importlib

ENGINES = {
    "ecology": "life_cube.engines.ecology",
    "lichen": "life_cube.engines.lichen",
    "terra": "life_cube.engines.terra",
}

_cache = {}


def get_rules(name: str):
    if name not in ENGINES:
        raise KeyError(f"неизвестный движок {name!r}; есть: {', '.join(ENGINES)}")
    if name not in _cache:
        _cache[name] = importlib.import_module(ENGINES[name]).RULES
    return _cache[name]


def list_engines():
    """[{name, title, summary}] для UI."""
    out = []
    for name in ENGINES:
        r = get_rules(name)
        out.append({"name": name, "title": r.title, "summary": r.summary})
    return out


class Rules:
    """Интерфейс движка. Все методы — без состояния, состояние живёт в `state`."""

    name = "base"
    title = "Базовый"
    summary = ""
    doc = ""                    # путь к docs/engines/*.md

    Config = None               # dataclass конфигурации

    def make_config(self, **kw):
        return self.Config(**kw)

    def init_state(self, cfg, xp):
        """-> (state: dict, relief: numpy (n,n))"""
        raise NotImplementedError

    def step(self, state, cfg, xp, correlate, gen):
        """Одно поколение. -> список населения по видам (длина n_species)."""
        raise NotImplementedError

    def n_species(self, cfg):
        return cfg.n_species

    def species_names(self, cfg):
        raise NotImplementedError

    def species_colors(self, cfg):
        raise NotImplementedError

    def to_json(self, cfg, state=None):
        """Описание для панели конструктора: fields/labels/ranges/genomes/world."""
        raise NotImplementedError

    def apply_genomes(self, cfg, state, genomes, xp):
        """Заменить геномы на лету."""
        raise NotImplementedError

    def randomize(self, cfg, rng):
        """-> новая таблица геномов (случайная, но осмысленная)."""
        raise NotImplementedError

    def world_params(self):
        """Имена параметров мира, которые можно менять из панели."""
        return ()
