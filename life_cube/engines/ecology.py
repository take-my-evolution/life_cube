"""Движок «экология» — исходная модель: камень, свет, вода, растения с
ветвлением, травоядные и хищники, фиксированный набор видов.

Код самой модели живёт в life_cube/{config,world,fields,step,motion}.py;
этот модуль только оборачивает его в интерфейс Rules.
Описание: docs/engines/ecology.md
"""

import numpy as np

from . import Rules
from .. import config as C
from ..sim import init_state as _init_state
from ..step import step as _step
from ..world import allowed_species
from . import seeding_json


class EcologyRules(Rules):
    name = "ecology"
    title = "Экология: растения, травоядные, хищники"
    summary = ("Свет сверху, вода снизу, ветвящиеся растения, подвижные "
               "травоядные и хищники. Виды фиксированы, гены крутятся в панели.")
    doc = "docs/engines/ecology.md"
    Config = C.Config

    can_seed = True

    WORLD_PARAMS = ("n", "seed_world", "seed_mut", "seed_density", "animal_share",
                    "stone_fraction", "relief_amp", "p_shock", "p_dissolve",
                    "plant_energy", "eat_efficiency", "move_noise",
                    "rain_rate", "rain_amount", "rain_decay")

    def init_state(self, cfg, xp):
        return _init_state(cfg, xp)

    def step(self, state, cfg, xp, correlate, gen):
        return _step(state, cfg, xp, correlate, gen)

    GENE_DOCS = {
        "absorb": "Сколько света клетка съедает по дороге вниз. Высокий absorb "
                  "кормит саму клетку и одновременно затеняет всё под ней — так "
                  "крона дерева душит подлесок.",
        "up": "Вес высоты при выборе места для рождения. 0 — вид стелется "
              "ковром, большие значения строят столбы и стволы.",
        "birth": "Порог: сколько взвешенных соседей нужно, чтобы в пустой клетке "
                 "появилась новая. Ниже порог — быстрее расползается, но рыхлее.",
        "need": "Минимум ресурса, при котором клетка ещё жива. Высокий need — "
                "вид занимает только лучшие места, зато конкурирует там жёстко.",
        "water": "Чем ближе к 1, тем больше вид кормится водой снизу и меньше "
                 "светом сверху. 0 — чистый фотосинтетик, 1 — чистый водолюб.",
        "branch": "Вес бокового роста, помноженный на свет в точке. Отсюда "
                  "берутся ветки и кроны: вбок растут туда, где светло.",
        "hunt": "И сила удара, и доля энергии жертвы, которая достанется едоку. "
                "Работает только у подвижных.",
        "trophic": "Уровень в цепи: 0 растение, 1 травоядное, 2 хищник. Едят "
                   "ровно на уровень ниже — хищник без травоядных вымрет, даже "
                   "если травы полно.",
        "speed": "Шагов за поколение. Ноль делает вид сидячим (растением), "
                 "любое положительное значение — подвижным существом.",
        "sense": "Радиус чутья: на сколько клеток вид чувствует запах еды. "
                 "Ноль — бродит вслепую.",
        "metabolism": "Трата энергии за поколение — и у животных, и у растений. "
                      "У растения это дыхание: чем выше, тем меньше остаётся на "
                      "прирост биомассы и тем быстрее оно гибнет в тени.",
        "repro": "Порог накопления для деления. Новая клетка стоит "
                 "repro × mass × growth_cost, поэтому именно этот ген (вместе с "
                 "массой) задаёт СКОРОСТЬ РОСТА вида: корка удваивается за "
                 "десяток поколений, дерево — почти за сотню.",
        "lifespan": "Предельный возраст в поколениях, 0 — не стареет. "
                    "Быстрые виды-пионеры живут мало, дерево — тысячи поколений.",
        "mass": "Масса одной клетки — плотность ткани. Клетка мха и клетка "
                "дерева не равны: масса задаёт цену постройки (плотное растёт "
                "дольше), пищевую ценность для травоядного и вклад вида в "
                "биомассу мира.",
        "armor": "Шанс отбиться от нападения. Броня травоядного — вторая "
                 "крутилка баланса цепи после hunt хищника.",
    }

    def gene_docs(self):
        return dict(self.GENE_DOCS)

    # --- засев --------------------------------------------------------------
    def starters_json(self, cfg):
        """Кем можно заселить мир: все виды движка, отмечены выбранные.
        Пустой выбор = все (так мир вёл себя всегда)."""
        want = set(allowed_species(cfg))
        mobile = cfg.mobile_mask()
        out = []
        for i, name in enumerate(list(C.SPECIES_NAMES)[: cfg.n_species]):
            tro = int(cfg.genomes[i][C.IDX["trophic"]])
            habitat = ("растение", "травоядное", "хищник")[min(tro, 2)] if mobile[i] or tro else "растение"
            out.append({"i": i + 1, "name": name, "habitat": habitat,
                        "on": (i + 1) in want})
        return out

    def seed(self, state, cfg, xp, rng, count=None, gen=0):
        """Подсев спор и существ на поверхность: те же виды, что и в начале."""
        from ..backend import to_cpu
        n = cfg.n
        relief = np.asarray(state.get("relief"))
        if relief is None or relief.shape != (n, n):
            return 0
        species = to_cpu(state["species"]).copy()
        energy = to_cpu(state["energy"]).copy()
        allowed = sorted(allowed_species(cfg))
        if not allowed:
            return 0
        mobile = cfg.mobile_mask()
        count = int(count if count is not None else getattr(cfg, "reseed_count", 200))
        rng_cpu = np.random.default_rng((cfg.seed_mut ^ 0x5bf03635) + gen)
        xs = rng_cpu.integers(0, n, count)
        ys = rng_cpu.integers(0, n, count)
        zs = np.clip(relief[xs, ys], 0, n - 1)
        pick = np.asarray(allowed)[rng_cpu.integers(0, len(allowed), count)]
        free = species[xs, ys, zs] == 0
        xs, ys, zs, pick = xs[free], ys[free], zs[free], pick[free]
        if len(xs) == 0:
            return 0
        species[xs, ys, zs] = pick.astype(species.dtype)
        is_anim = mobile[pick - 1]
        energy[xs, ys, zs] = np.where(is_anim, cfg.start_energy, cfg.plant_energy)
        state["species"] = xp.asarray(species)
        state["energy"] = xp.asarray(energy)
        state["last_reseed"] = gen
        return int(len(xs))

    def species_names(self, cfg):
        return list(C.SPECIES_NAMES)[: cfg.n_species]

    def species_colors(self, cfg):
        return list(C.SPECIES_COLORS)[: cfg.n_species]

    def species_mass(self, cfg):
        """Масса клетки — ген `mass` (плотность ткани). 0 читается как 1,
        чтобы старые геномы без этого гена вели себя как раньше."""
        i = C.IDX.get("mass")
        g = np.asarray(cfg.genomes)
        if i is None or g.shape[1] <= i:
            return [1.0] * cfg.n_species
        return [float(v) if v > 0 else 1.0 for v in g[:, i]]

    def world_params(self):
        return self.WORLD_PARAMS

    def to_json(self, cfg, state=None):
        j = cfg.to_json()
        j["engine"] = self.name
        j["starters"] = self.starters_json(cfg)
        j["reseed"] = seeding_json(cfg)
        return j

    def apply_genomes(self, cfg, state, genomes, xp):
        g = np.asarray(genomes, dtype=np.float32)
        if g.ndim != 2 or g.shape[1] != len(C.GENOME_FIELDS):
            raise ValueError(f"геномы: ожидается таблица вид × {len(C.GENOME_FIELDS)}")
        cfg.genomes = g
        state["genomes"] = xp.asarray(g)

    def randomize(self, cfg, rng):
        """Случайные гены в пределах ползунков, но структура сохраняется:
        растения остаются растениями, животные — животными своего уровня."""
        g = np.asarray(cfg.genomes, dtype=np.float32).copy()
        mobile = cfg.mobile_mask()
        for s in range(len(g)):
            for name, i in C.IDX.items():
                lo, hi, _st = C.GENOME_RANGES[name]
                if name in ("trophic", "speed"):
                    continue                     # роль и подвижность не трогаем
                if mobile[s] and name in ("absorb", "up", "birth", "need", "water", "branch"):
                    continue                     # у животных это не работает
                if not mobile[s] and name in ("sense", "armor", "hunt"):
                    continue        # это только про животных
                g[s, i] = rng.uniform(lo, hi)
            if not mobile[s]:
                # у растений держим ресурсные пороги и экономику роста в
                # живом диапазоне: иначе случайный вид либо не растёт вовсе,
                # либо мгновенно заливает мир
                g[s, C.IDX["need"]] = rng.uniform(0.15, 0.6)
                g[s, C.IDX["birth"]] = rng.uniform(1.3, 2.2)
                g[s, C.IDX["metabolism"]] = rng.uniform(0.005, 0.05)
                g[s, C.IDX["repro"]] = rng.uniform(1.0, 8.0)
                g[s, C.IDX["mass"]] = rng.uniform(0.4, 6.0)
                g[s, C.IDX["lifespan"]] = float(rng.choice([0, 300, 600, 1200, 2500]))
            else:
                g[s, C.IDX["metabolism"]] = rng.uniform(0.03, 0.15)
                g[s, C.IDX["repro"]] = rng.uniform(2.0, 6.0)
        return g


RULES = EcologyRules()
