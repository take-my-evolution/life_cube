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


class EcologyRules(Rules):
    name = "ecology"
    title = "Экология: растения, травоядные, хищники"
    summary = ("Свет сверху, вода снизу, ветвящиеся растения, подвижные "
               "травоядные и хищники. Виды фиксированы, гены крутятся в панели.")
    doc = "docs/engines/ecology.md"
    Config = C.Config

    WORLD_PARAMS = ("n", "seed_world", "seed_mut", "seed_density", "animal_share",
                    "stone_fraction", "relief_amp", "p_shock", "p_dissolve",
                    "plant_energy", "eat_efficiency", "move_noise")

    def init_state(self, cfg, xp):
        return _init_state(cfg, xp)

    def step(self, state, cfg, xp, correlate, gen):
        return _step(state, cfg, xp, correlate, gen)

    def species_names(self, cfg):
        return list(C.SPECIES_NAMES)[: cfg.n_species]

    def species_colors(self, cfg):
        return list(C.SPECIES_COLORS)[: cfg.n_species]

    def world_params(self):
        return self.WORLD_PARAMS

    def to_json(self, cfg, state=None):
        j = cfg.to_json()
        j["engine"] = self.name
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
                if not mobile[s] and name in ("sense", "metabolism", "repro", "lifespan", "armor", "hunt"):
                    continue
                g[s, i] = rng.uniform(lo, hi)
            if not mobile[s]:
                # у растений держим ресурсные пороги в живом диапазоне
                g[s, C.IDX["need"]] = rng.uniform(0.15, 0.6)
                g[s, C.IDX["birth"]] = rng.uniform(1.3, 2.2)
            else:
                g[s, C.IDX["metabolism"]] = rng.uniform(0.03, 0.15)
                g[s, C.IDX["repro"]] = rng.uniform(2.0, 6.0)
        return g


RULES = EcologyRules()
