"""Реестр движков (наборов правил).

Каждый движок — модуль с объектом `RULES`, реализующим интерфейс `Rules`:
свой Config, свой мир, свой шаг, свои гены. Оркестровка (Engine), снимки,
рендеры и звук — общие: они видят только массив `species` (0 = пусто) плюс
имена и цвета видов, которые движок отдаёт сам.

Добавить движок = положить модуль сюда, дать ему `RULES` и описать его в
docs/engines/<name>.md. Ниже — список известных; порядок = порядок в UI.
"""

import importlib

import numpy as np

ENGINES = {
    "ecology": "life_cube.engines.ecology",
    "lichen": "life_cube.engines.lichen",
    "terra": "life_cube.engines.terra",
    "slope": "life_cube.engines.slope",
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


def fork_dynamic(state, cfg, xp, sid, genome, gen, share=0.3, rng=None):
    """Ответвить новый вид от живущего (для движков с динамическими видами).

    Правка генома «на месте» переписывает вид целиком и стирает то, что
    отобралось эволюцией. Форк вместо этого заводит НОВЫЙ id с родословной
    (parent = sid) и перекрашивает в него часть клеток родителя — иначе вид с
    нулевым населением тут же сочтут вымершим и id вернётся в пул.

    Возвращает (new_sid, сколько клеток перекрашено).
    """
    from ..backend import to_cpu

    reg = state.get("registry") or {}
    if sid not in reg:
        raise ValueError(f"вид #{sid} не живёт — ответвлять не от чего")
    if not state.get("free_ids"):
        raise ValueError("свободных номеров видов не осталось: подожди, пока кто-нибудь вымрет")
    g = np.asarray(genome, dtype=np.float32).ravel()
    width = int(np.asarray(cfg.genomes).shape[1])
    if g.size != width:
        raise ValueError(f"геном: ожидается {width} чисел, пришло {g.size}")

    species = to_cpu(state["species"])
    mine = np.argwhere(species == sid)
    if len(mine) == 0:
        raise ValueError(f"у вида #{sid} нет живых клеток")
    new_sid = state["free_ids"].pop(0)
    cfg.genomes[new_sid - 1] = g
    reg[new_sid] = {"parent": int(sid), "born": int(gen), "died": None,
                    "genome": g.tolist(), "peak": 1, "changed": "конструктор"}

    share = float(min(max(share, 0.0), 1.0))
    k = max(1, int(round(len(mine) * share)))
    rng_cpu = state.get("rng_cpu")
    if rng_cpu is None:
        rng_cpu = np.random.default_rng(gen)
    pick = mine[rng_cpu.choice(len(mine), size=min(k, len(mine)), replace=False)]
    species = species.copy()
    species[pick[:, 0], pick[:, 1], pick[:, 2]] = new_sid
    state["species"] = xp.asarray(species)
    state["genomes"] = xp.asarray(cfg.genomes)
    return int(new_sid), int(len(pick))


def seeding_json(cfg):
    """Настройки повторного засева для панели."""
    return {"on": bool(getattr(cfg, "reseed", False)),
            "on_extinction": bool(getattr(cfg, "reseed_on_extinction", True)),
            "every": int(getattr(cfg, "reseed_every", 200)),
            "count": int(getattr(cfg, "reseed_count", 200))}


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

    def species_mass(self, cfg):
        """Масса ОДНОЙ клетки каждого вида — для расчёта биомассы.
        По умолчанию все клетки одинаковы (единица); движки, у которых есть
        ген массы, переопределяют."""
        return [1.0] * int(self.n_species(cfg))

    def species_organisms(self, cfg, state=None):
        """Сколько ОРГАНИЗМОВ каждого вида. По умолчанию — None: у движка, где
        клетка и есть организм, отдельная колонка ничего не добавит. Движки, где
        организм собирается из клеток (дерево — ствол и крона), считают сами."""
        return None

    def to_json(self, cfg, state=None):
        """Описание для панели конструктора: fields/labels/ranges/genomes/world."""
        raise NotImplementedError

    def apply_genomes(self, cfg, state, genomes, xp):
        """Заменить геномы на лету."""
        raise NotImplementedError

    def randomize(self, cfg, rng):
        """-> новая таблица геномов (случайная, но осмысленная)."""
        raise NotImplementedError

    # --- конструктор ------------------------------------------------------
    # Движки с фиксированными видами правят геном «на месте»; движки, где виды
    # рождаются сами, умеют ещё и ответвлять новый вид от живущего.
    can_fork = False

    def fork_species(self, cfg, state, sid, genome, xp, gen=0, share=0.3):
        """-> (новый id, сколько клеток перекрашено)."""
        raise ValueError("этот движок не умеет ответвлять виды: "
                         "виды здесь фиксированы, правь геном на месте")

    def gene_docs(self):
        """{ген: человеческое описание} для карточки конструктора."""
        return {}

    # --- засев ------------------------------------------------------------
    # Движок может уметь подсаживать жизнь в уже готовый мир: этим пользуется
    # Engine, когда включён повторный засев (спасение вымершего мира).
    can_seed = False

    def seed(self, state, cfg, xp, rng, count=None, gen=0):
        """Подсадить `count` клеток стартовых видов. -> сколько посажено."""
        return 0

    def starters_json(self, cfg):
        """[{i, name, habitat, on}] — кем можно заселить мир (для панели)."""
        return []

    def world_params(self):
        """Имена параметров мира, которые можно менять из панели."""
        return ()
