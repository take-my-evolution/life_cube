"""Дождь: влажность подложки (`wet`) больше не застывший навсегда узор ниш —
сохнет каждое поколение и пополняется каплями. Без дождя водолюбивые виды
(например мох) со временем усыхают вместо бесконечного разрастания по всему
камню; дождь регулируется тремя ручками конфига."""

import numpy as np

from life_cube import Config, run
from life_cube.backend import get_backend
from life_cube.sim import init_state
from life_cube.step import step


def test_wet_field_decays_without_rain_and_holds_with_it():
    """`wet` больше не константа: без дождя высыхает почти до нуля, а с
    дождём остаётся на заметном уровне (регулярно пополняется)."""
    xp, corr, _ = get_backend(False)

    dry_cfg = Config(n=32, seed_density=0.02, rain_rate=0.0, rain_decay=0.9)
    dry_state, _ = init_state(dry_cfg, xp)
    w0 = float(dry_state["wet"].mean())
    for gen in range(80):
        step(dry_state, dry_cfg, xp, corr, gen)
    w_dry = float(dry_state["wet"].mean())
    assert w_dry < w0 * 0.01, (w0, w_dry)          # высохло почти дотла

    rain_cfg = Config(n=32, seed_density=0.02, rain_rate=0.05, rain_amount=0.5, rain_decay=0.995)
    rain_state, _ = init_state(rain_cfg, xp)
    for gen in range(80):
        step(rain_state, rain_cfg, xp, corr, gen)
    w_rain = float(rain_state["wet"].mean())
    assert w_rain > w0 * 0.5, (w0, w_rain)         # дождь держит влажность на плаву


def test_moss_population_lower_under_drought_than_with_rain():
    """Экологический эффект дождя: тот же мир, тот же сид, только дождь
    выключен или включён — без животных (изолируем растения от выедания),
    засухa должна заметно придушить популяцию мха относительно дождливого
    прогона. Раньше (статичная wet) дождь вообще ни на что не влиял."""
    common = dict(n=32, gens=150, seed_density=0.02, animal_share=0.0)
    dry = run(Config(rain_rate=0.0, rain_decay=0.9, **common), verbose=False)
    rain = run(Config(rain_rate=0.05, rain_amount=0.5, rain_decay=0.995, **common), verbose=False)
    moss_dry = dry["hist"][:, 0]
    moss_rain = rain["hist"][:, 0]
    assert moss_rain[-1] > moss_dry[-1] * 1.5, (moss_dry[-1], moss_rain[-1])
    assert moss_rain.max() > moss_dry.max() * 1.3, (moss_dry.max(), moss_rain.max())


def test_rain_config_exposed_to_world_json_and_ecology_engine():
    """Дождь регулируется из панели мира (см. WORLD_PARAMS движка «экология»
    и index.html #rainRate) — ручки должны реально долетать до cfg.to_json()."""
    from life_cube.engines.ecology import EcologyRules
    cfg = Config()
    j = cfg.to_json()
    assert {"rain_rate", "rain_amount", "rain_decay"} <= j["world"].keys()
    assert {"rain_rate", "rain_amount", "rain_decay"} <= set(EcologyRules.WORLD_PARAMS)
