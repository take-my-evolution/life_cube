"""Движок «терра»: горы, реки, почва — и жизнь, которая их переделывает.

Концепт (docs/engines/terra.md, кратко):
  * Мир сразу трёхсоставный: КАМЕНЬ (примерно треть высоты куба, многооктавный
    рельеф — хребты и долины, без одинокого холма), ПОЧВА (тонкий слой в
    низинах) и ВОДА (заливает впадины до уровня моря).
  * Гидрология каждый шаг: дождь (в горах чаще), сток к самому низкому соседу
    по полной высоте, эрозия и перенос почвы потоком, отложение на спокойной
    воде, испарение. Русла и озёра возникают сами.
  * Лишайник (белый) живёт на голом камне и грызёт его: одна клетка камня даёт
    `soil_per_stone` (по умолчанию 3) клетки почвы — обмен НЕРАВНОЦЕННЫЙ, поэтому
    мир со временем «зарастает» почвой, а горы срабатываются.
  * Ниша задаётся тремя генами: `stone`, `soil`, `water` — насколько хорошо вид
    кормится на камне, на почве и под водой. Лишайник умеет только камень.
  * `enrich` — бактерия: живёт на почве и «заражает» её. `symbiont` — вид,
    который без бактерий рядом почти не кормится: так растение может появиться
    только на почве, уже занятой бактериями.
  * `up` — рост вверх (стебель): клетка над землёй живёт только светом и только
    если под ней своя же клетка.
  * Мутация меняет один ген; сильнее всего мутируют те, кому плохо (стресс).
    Виды рождаются и вымирают сами, генеалогия пишется.

Цвет вида вычисляется из его генома (ниша + рост), а не из номера — картинка
читается: белое на камне, зелёное на почве, синее в воде, яркое и высокое —
растения.
"""

import colorsys
from dataclasses import dataclass, field

import numpy as np

from . import Rules, fork_dynamic, seeding_json
from ..fields import light_field

GENES = ("light", "stone", "soil", "water", "erode", "enrich", "symbiont",
         "up", "absorb", "metabolism", "repro", "mut")
LABELS = {
    "light": "усвоение света", "stone": "кормится на камне", "soil": "кормится на почве",
    "water": "кормится под водой", "erode": "грызёт камень (при воде)",
    "enrich": "обогащает почву (бактерия)", "symbiont": "нужны бактерии рядом",
    "up": "рост вверх (стебель)", "absorb": "тень (поглощение света)",
    "metabolism": "обмен веществ", "repro": "порог деления", "mut": "частота мутаций",
}
RANGES = {k: (0.0, 1.0, 0.01) for k in GENES}
RANGES["metabolism"] = (0.02, 0.4, 0.005)   # ниже — вид жил бы на одних объедках чужой ниши
RANGES["repro"] = (0.5, 6.0, 0.05)
RANGES["mut"] = (0.0, 0.3, 0.005)
IDX = {g: i for i, g in enumerate(GENES)}
MAX_SPECIES = 64

#                  light stone soil water erode enrich symb  up  absorb metab repro mut
LICHEN = np.array([0.85, 1.00, 0.05, 0.00, 0.55, 0.00, 0.00, 0.00, 0.35, 0.06, 1.8, 0.004],
                  np.float32)

# Кем можно заселить мир в начале. Каждый архетип — готовый геном; всё
# остальное (в том числе растения) должно появиться мутациями. Набор
# выбирается в панели: только лишайник, лишайник+бактерия, и так далее.
STARTERS = (
    ("лишайник", "камень", LICHEN),
    ("бактерия", "почва",
     np.array([0.80, 0.05, 1.00, 0.00, 0.05, 0.90, 0.00, 0.00, 0.30, 0.06, 1.8, 0.004], np.float32)),
    ("водоросль", "вода",
     np.array([0.85, 0.00, 0.10, 1.00, 0.00, 0.10, 0.00, 0.00, 0.25, 0.05, 1.6, 0.004], np.float32)),
    ("мох", "почва",
     np.array([0.75, 0.30, 0.80, 0.00, 0.10, 0.10, 0.00, 0.05, 0.40, 0.05, 1.6, 0.004], np.float32)),
)
STARTER_NAMES = tuple(x[0] for x in STARTERS)


@dataclass
class TerraConfig:
    n: int = 96
    gens: int = 400
    seed_world: int = 20260826
    seed_mut: int = 20260826

    # рельеф
    stone_fraction: float = 0.33    # средняя высота камня (доля куба)
    relief_amp: float = 0.55        # размах хребтов относительно средней высоты
    ridges: float = 0.6             # 0 — холмы, 1 — острые хребты
    soil_start: float = 0.35        # сколько низин уже покрыто почвой
    sea_level: float = 0.12         # уровень моря (доля площади под водой)

    # гидрология
    rain_rate: float = 0.035        # доля столбцов под каплей за поколение
    rain_mountains: float = 2.0     # во сколько раз чаще дождь на вершинах
    evaporate: float = 0.15         # доля воды, испаряющейся за поколение
    flow: float = 0.55              # доля воды, стекающей к нижнему соседу
    erode_by_water: float = 0.55    # вероятность смыть клетку почвы потоком
    deposit: float = 0.25           # вероятность осесть на спокойной воде

    # жизнь
    soil_per_stone: int = 3         # почвы из одной клетки камня (обмен неравноценный)
    start_energy: float = 1.0
    light_gain: float = 0.95
    niche_floor: float = 0.02       # доля корма в чужой нише
    symbiont_floor: float = 0.10    # доля корма симбионту на «пустой» почве
    fert_gain: float = 0.06         # насколько бактерия удобряет свой столбец за поколение
    fert_decay: float = 0.004       # плодородие само тает
    fert_bonus: float = 0.5         # прибавка почвенным видам на удобренной почве
    symbiont_floor_unused: float = 0.15    # доля корма симбионту без бактерий рядом
    water_light: float = 0.45       # сколько света доходит на клетку глубины
    erode_gain: float = 0.7
    erode_rate: float = 0.03
    lifespan: int = 400
    crowd_max: int = 5
    takeover: float = 0.35
    energy_cap: float = 2.5
    stress_mut: float = 0.5
    mut_scale: float = 0.22
    min_change: float = 0.06
    max_new_species: int = 4
    p_shock: float = 0.0004

    # кем заселяем в начале (номера архетипов из STARTERS) и повторный засев
    start_species: tuple = (0,)
    reseed: bool = False            # подсевать заново
    reseed_on_extinction: bool = True   # только когда всё вымерло
    reseed_every: int = 200         # не чаще, чем раз в столько поколений
    reseed_count: int = 40          # сколько клеток подсевать

    genomes: np.ndarray = field(
        default_factory=lambda: np.zeros((MAX_SPECIES, len(GENES)), np.float32))

    def __post_init__(self):
        if not self.genomes.any():
            self.genomes[0] = LICHEN

    @property
    def n_species(self):
        return MAX_SPECIES


# ---------------------------------------------------------------------------
# Рельеф: многооктавный шум, хребты и долины
# ---------------------------------------------------------------------------

def fractal_relief(cfg, rng):
    n = cfg.n
    field_ = np.zeros((n, n), np.float32)
    amp, total = 1.0, 0.0
    for octave in range(4):
        cells = max(2, int(2 ** (octave + 1)))
        g = rng.normal(0, 1, (cells + 1, cells + 1)).astype(np.float32)
        # билинейное растяжение решётки октавы до размера мира
        ys = np.linspace(0, cells, n, endpoint=False)
        i0 = np.floor(ys).astype(int); t = (ys - i0).astype(np.float32)
        t = t * t * (3 - 2 * t)                       # сглаживание
        a = g[i0][:, i0] * (1 - t)[:, None] * (1 - t)[None, :]
        b = g[i0 + 1][:, i0] * t[:, None] * (1 - t)[None, :]
        c = g[i0][:, i0 + 1] * (1 - t)[:, None] * t[None, :]
        d = g[i0 + 1][:, i0 + 1] * t[:, None] * t[None, :]
        oct_field = a + b + c + d
        if cfg.ridges > 0:                            # |шум| даёт острые гребни
            oct_field = (1 - cfg.ridges) * oct_field + cfg.ridges * (1.0 - 2.0 * np.abs(oct_field))
        field_ += amp * oct_field
        total += amp
        amp *= 0.5
    field_ /= max(total, 1e-6)
    f = (field_ - field_.min()) / max(field_.max() - field_.min(), 1e-6)
    base = cfg.stone_fraction * n
    h = base * (1.0 - cfg.relief_amp) + f * base * cfg.relief_amp * 2.0
    return np.clip(np.round(h), 1, n - 8).astype(np.int32)


def _fill_basins(ground, level):
    """Вода стоит везде, где земля ниже уровня моря."""
    return np.maximum(level - ground, 0).astype(np.int32)


def genome_color(g, sid=0):
    """Цвет вида — из его генома, а не из номера: ниша задаёт тон (камень —
    белёсый, почва — зелёный, вода — синий), симбиоз уводит в жёлто-зелёное,
    рост вверх делает цвет темнее и насыщеннее. Небольшой сдвиг по номеру
    разводит соседние виды, чтобы их было видно по отдельности."""
    stone, soil, water = float(g[IDX["stone"]]), float(g[IDX["soil"]]), float(g[IDX["water"]])
    up, sym, enr = float(g[IDX["up"]]), float(g[IDX["symbiont"]]), float(g[IDX["enrich"]])
    tot = stone + soil + water + 1e-6
    st_s, so_s, wa_s = stone / tot, soil / tot, water / tot
    hue = (0.13 * st_s + 0.30 * so_s + 0.58 * wa_s + 0.10 * sym - 0.05 * enr
           + 0.013 * ((sid * 7) % 5)) % 1.0
    sat = float(np.clip(0.25 + 0.65 * (1.0 - st_s) + 0.25 * up, 0.12, 0.95))
    light = float(np.clip(0.78 - 0.30 * (so_s + wa_s) - 0.18 * up, 0.3, 0.9))
    r, gg, b = colorsys.hls_to_rgb(hue, light, sat)
    return "#%02x%02x%02x" % (int(r * 255), int(gg * 255), int(b * 255))


class TerraRules(Rules):
    name = "terra"
    title = "Терра: горы, реки, почва, бактерии и растения"
    summary = ("Камень на треть куба, реки и озёра, почва мигрирует по руслам. "
               "Лишайник грызёт камень (1 камень → 3 почвы), на почве заводятся "
               "бактерии, а растения растут только там, где бактерии уже есть.")
    doc = "docs/engines/terra.md"
    Config = TerraConfig
    dynamic_species = True
    terrain_changes = True
    heightmaps = True
    can_seed = True
    can_fork = True

    WORLD_PARAMS = ("n", "seed_world", "seed_mut", "stone_fraction", "relief_amp",
                    "ridges", "soil_start", "sea_level", "rain_rate", "evaporate",
                    "erode_by_water", "deposit", "soil_per_stone", "erode_rate",
                    "stress_mut", "mut_scale", "max_new_species", "lifespan", "p_shock")
    WORLD_RANGES = {"n": [32, 224, 32], "stone_fraction": [0.1, 0.6, 0.01],
                    "relief_amp": [0.0, 1.0, 0.05], "ridges": [0.0, 1.0, 0.05],
                    "soil_start": [0.0, 1.0, 0.05], "sea_level": [0.0, 0.5, 0.01],
                    "rain_rate": [0.0, 0.3, 0.005], "evaporate": [0.0, 0.4, 0.01],
                    "erode_by_water": [0.0, 1.0, 0.05], "deposit": [0.0, 1.0, 0.05],
                    "soil_per_stone": [1, 6, 1], "erode_rate": [0.0, 1.0, 0.02],
                    "stress_mut": [0.0, 1.0, 0.02], "mut_scale": [0.02, 0.6, 0.02],
                    "max_new_species": [0, 16, 1], "lifespan": [20, 2000, 10],
                    "p_shock": [0, 0.01, 0.0001]}
    WORLD_LABELS = {"n": "размер куба", "seed_world": "сид мира", "seed_mut": "сид мутаций",
                    "stone_fraction": "камень: доля высоты куба", "relief_amp": "размах гор",
                    "ridges": "острота хребтов", "soil_start": "почвы в начале",
                    "sea_level": "уровень моря", "rain_rate": "дождь",
                    "evaporate": "испарение", "erode_by_water": "смыв почвы потоком",
                    "deposit": "осаждение почвы", "soil_per_stone": "почвы из клетки камня",
                    "erode_rate": "скорость проедания камня", "stress_mut": "мутации от стресса",
                    "mut_scale": "размер мутации", "max_new_species": "новых видов за поколение",
                    "lifespan": "предел возраста", "p_shock": "выветривание"}

    DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    DIRS3 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1))

    # ------------------------------------------------------------------ мир
    def init_state(self, cfg, xp):
        rng = np.random.default_rng(cfg.seed_world)
        n = cfg.n
        stone_h = fractal_relief(cfg, rng)
        # уровень моря — по рельефу, а не по абсолютной высоте: sea_level это
        # доля площади, которую заливает вода (иначе на «высоком» мире воды
        # не было бы вовсе)
        sea = 0
        # почва: в низинах толще, на вершинах нет
        rel = (stone_h - stone_h.min()) / max(stone_h.max() - stone_h.min(), 1)
        soil_h = np.round(cfg.soil_start * 4.0 * np.clip(1.0 - rel, 0, 1) ** 2
                          * rng.uniform(0.5, 1.5, (n, n))).astype(np.int32)
        ground = stone_h + soil_h
        sea = int(np.percentile(ground, 100.0 * np.clip(cfg.sea_level, 0, 0.9)))
        water_h = _fill_basins(ground, sea)

        species = np.zeros((n, n, n), np.uint8)
        energy = np.zeros((n, n, n), np.float32)
        starters = list(cfg.start_species) or [0]
        genomes = np.zeros((MAX_SPECIES, len(GENES)), np.float32)
        registry, free_ids = {}, list(range(1, MAX_SPECIES + 1))
        for si in starters:
            name, habitat, gen0 = STARTERS[int(si) % len(STARTERS)]
            sid = free_ids.pop(0)
            genomes[sid - 1] = gen0
            mask = self._habitat_mask(habitat, stone_h, soil_h, water_h)
            xs, ys = np.nonzero(mask)
            species[xs, ys, ground[xs, ys]] = sid
            energy[xs, ys, ground[xs, ys]] = cfg.start_energy
            registry[sid] = {"parent": 0, "born": 0, "died": None,
                             "genome": gen0.tolist(), "peak": int(mask.sum()),
                             "changed": None, "starter": name}
        cfg.genomes = genomes

        state = {
            "species": xp.asarray(species),
            "energy": xp.asarray(energy),
            "age": xp.zeros((n, n, n), dtype=xp.int32),
            "stone_h": xp.asarray(stone_h),
            "soil_h": xp.asarray(soil_h),
            "water_h": xp.asarray(water_h),
            "water_f": xp.asarray(water_h.astype(np.float32)),
            "flow": xp.zeros((n, n), dtype=xp.float32),
            # плодородие почвы: бактерии «заражают» её, растения-симбионты
            # селятся только на заражённой
            "fert": xp.zeros((n, n), dtype=xp.float32),
            "genomes": xp.asarray(cfg.genomes),
            "rng": xp.random.default_rng(cfg.seed_mut),
            "rng_cpu": np.random.default_rng(cfg.seed_mut ^ 0x9e3779b9),
            "registry": registry,
            "lineage": [], "free_ids": free_ids, "gen": 0,
            "last_reseed": 0,
        }
        self._sync_volumes(state, cfg, xp)
        return state, stone_h.copy()

    @staticmethod
    def _habitat_mask(habitat, stone_h, soil_h, water_h):
        """Где может сесть архетип: камень — сухой голый камень, почва —
        столбцы с почвой без глубокой воды, вода — залитые столбцы."""
        if habitat == "камень":
            return (soil_h == 0) & (water_h == 0)
        if habitat == "почва":
            return (soil_h > 0) & (water_h == 0)
        return water_h > 0

    def seed(self, state, cfg, xp, rng, count=None, gen=0):
        """Подсев: живые клетки стартовых архетипов в подходящие пустые места.
        Возвращает, сколько клеток посажено."""
        from ..backend import to_cpu
        n = cfg.n
        stone_h, soil_h, water_h = (to_cpu(state["stone_h"]), to_cpu(state["soil_h"]),
                                    to_cpu(state["water_h"]))
        ground = stone_h + soil_h
        species = to_cpu(state["species"])
        energy = to_cpu(state["energy"])
        rng_cpu = state["rng_cpu"]
        count = int(count or cfg.reseed_count)
        planted = 0
        for si in (list(cfg.start_species) or [0]):
            name, habitat, gen0 = STARTERS[int(si) % len(STARTERS)]
            # свой ли это вид уже живёт — подсаживаем его же, иначе заводим новый
            sid = next((k for k, r in state["registry"].items()
                        if r.get("starter") == name), None)
            if sid is None:
                if not state["free_ids"]:
                    continue
                sid = state["free_ids"].pop(0)
                cfg.genomes[sid - 1] = gen0
                state["registry"][sid] = {"parent": 0, "born": gen, "died": None,
                                          "genome": gen0.tolist(), "peak": 1,
                                          "changed": None, "starter": name}
            mask = self._habitat_mask(habitat, stone_h, soil_h, water_h)
            xs, ys = np.nonzero(mask)
            if len(xs) == 0:
                continue
            k = min(max(count // max(len(cfg.start_species) or 1, 1), 1), len(xs))
            pick = rng_cpu.choice(len(xs), size=k, replace=False)
            x, y = xs[pick], ys[pick]
            z = ground[x, y]
            free = species[x, y, z] == 0
            species[x[free], y[free], z[free]] = sid
            energy[x[free], y[free], z[free]] = cfg.start_energy
            planted += int(free.sum())
        state["species"] = xp.asarray(species)
        state["energy"] = xp.asarray(energy)
        state["genomes"] = xp.asarray(cfg.genomes)
        state["last_reseed"] = gen
        return planted

    def starters_json(self, cfg):
        return [{"i": i, "name": nm, "habitat": hb, "on": i in tuple(cfg.start_species)}
                for i, (nm, hb, _g) in enumerate(STARTERS)]

    def _sync_volumes(self, state, cfg, xp):
        n = cfg.n
        zz = xp.arange(n)[None, None, :]
        sh = state["stone_h"][:, :, None]
        so = state["soil_h"][:, :, None]
        state["stone"] = zz < sh
        state["soil"] = (zz >= sh) & (zz < sh + so)

    @staticmethod
    def _shift2(a, dx, dy, xp, fill=0):
        out = xp.full_like(a, fill)
        sx = slice(max(dx, 0), a.shape[0] + min(dx, 0))
        sy = slice(max(dy, 0), a.shape[1] + min(dy, 0))
        tx = slice(max(-dx, 0), a.shape[0] + min(-dx, 0))
        ty = slice(max(-dy, 0), a.shape[1] + min(-dy, 0))
        out[sx, sy] = a[tx, ty]
        return out

    # ------------------------------------------------------------ гидрология
    def _hydrology(self, state, cfg, xp, rng):
        """Дождь, сток к самому низкому соседу, перенос почвы, испарение."""
        stone_h, soil_h, water_h = state["stone_h"], state["soil_h"], state["water_h"]
        n = cfg.n
        ground = stone_h + soil_h
        # дождь: чаще на высоте
        rel = (ground.astype(xp.float32) - float(ground.min())) / max(float(ground.max() - ground.min()), 1.0)
        p_rain = cfg.rain_rate * (1.0 + (cfg.rain_mountains - 1.0) * rel)
        water = state.get("water_f")
        water = (water_h.astype(xp.float32) if water is None else water) + (rng.random((n, n)) < p_rain)

        total = ground.astype(xp.float32) + water
        big = float(total.max()) + 1000.0
        best_drop = xp.zeros((n, n), dtype=xp.float32)
        best_dir = xp.full((n, n), -1, dtype=xp.int32)
        for k, (dx, dy) in enumerate(self.DIRS4):
            nb = self._shift2(total, dx, dy, xp, fill=big)
            drop = total - nb
            better = drop > best_drop
            best_drop = xp.where(better, drop, best_drop)
            best_dir = xp.where(better, k, best_dir)
        moving = (best_dir >= 0) & (water > 0)
        out = xp.where(moving, xp.minimum(water, xp.maximum(best_drop * cfg.flow, 1.0)), 0.0)
        water = water - out

        # Почву смывает со СКЛОНОВ — не только текущей водой, но и самим дождём.
        # Иначе почва, однажды появившись, остаётся навсегда, и через несколько
        # сотен поколений камня на поверхности не остаётся вовсе: лишайнику
        # негде жить и эволюции не от чего отталкиваться (наступали).
        gslope = xp.zeros((n, n), dtype=xp.float32)
        gdir = xp.full((n, n), -1, dtype=xp.int32)
        gbig = float(ground.max()) + 1000.0
        for k, (dx, dy) in enumerate(self.DIRS4):
            nb = self._shift2(ground.astype(xp.float32), dx, dy, xp, fill=gbig)
            drop = ground.astype(xp.float32) - nb
            better = drop > gslope
            gslope = xp.where(better, drop, gslope)
            gdir = xp.where(better, k, gdir)
        speed = xp.where(moving, best_drop, 0.0)
        wet_here = (water + out) > 0
        wash = ((soil_h > 0) & (gdir >= 0) & wet_here
                & (rng.random((n, n)) < cfg.erode_by_water * xp.minimum(gslope / 2.0, 1.0)))
        soil_h = soil_h - wash.astype(soil_h.dtype)
        for k, (dx, dy) in enumerate(self.DIRS4):
            water = water + self._shift2(xp.where(best_dir == k, out, 0.0), -dx, -dy, xp)
            soil_h = soil_h + self._shift2(
                xp.where((gdir == k) & wash, 1, 0).astype(soil_h.dtype), -dx, -dy, xp)
        # осаждение: где вода стоит (не течёт) и глубока — почва оседает
        calm = (~moving) & (water > 1)
        # (вещество уже перенесено выше; здесь только «успокоение» рельефа)
        state["flow"] = speed
        water = xp.maximum(water * (1.0 - cfg.evaporate), 0.0)
        state["water_f"] = water
        # наружу — целые клетки воды: лужа мельче клетки водоёмом не считается
        state["water_h"] = xp.floor(water).astype(water_h.dtype)
        state["soil_h"] = xp.maximum(soil_h, 0)
        if "fert" in state:
            state["fert"] = xp.where(wash, state["fert"] * 0.4, state["fert"])
        state["stone_h"] = stone_h
        return calm

    # ------------------------------------------------------------------ шаг
    def step(self, state, cfg, xp, correlate, gen):
        n = cfg.n
        rng = state["rng"]
        G = state["genomes"]
        self._hydrology(state, cfg, xp, rng)
        stone_h, soil_h, water_h = state["stone_h"], state["soil_h"], state["water_h"]
        species, energy, age = state["species"], state["energy"], state["age"]

        zz = xp.arange(n)[None, None, :]
        ground = stone_h + soil_h
        ground3 = ground[:, :, None]
        alive = species > 0
        idx = xp.clip(species.astype(xp.int32) - 1, 0, MAX_SPECIES - 1)

        def gene(name):
            return xp.where(alive, G[idx, IDX[name]], 0.0).astype(xp.float32)

        on_ground = alive & (zz == ground3)
        above = alive & (zz > ground3)
        # глубина считается для ЛЮБОЙ клетки, а не только для донной: стебель,
        # идущий вверх сквозь толщу воды, тоже сидит в тени, а вынырнув —
        # получает полный свет. Отсюда смысл расти вверх из воды.
        surface_h = (ground + water_h)[:, :, None]
        depth = xp.maximum(surface_h - zz, 0).astype(xp.float32)
        under = alive & (zz < surface_h)
        submerged = on_ground & (water_h[:, :, None] > 0)

        # --- свет (вода гасит) ------------------------------------------------
        L = light_field(alive, gene("absorb"), xp)
        L = L * xp.where(under, cfg.water_light ** xp.minimum(depth, 6.0), 1.0)

        # --- ниша: камень / почва / вода -------------------------------------
        has_soil = (soil_h > 0)[:, :, None]
        niche = xp.where(submerged, gene("water"),
                         xp.where(has_soil & on_ground, gene("soil"),
                                  xp.where(on_ground, gene("stone"), 0.0)))
        # клетка над землёй (стебель) живёт светом, но только если под ней своя
        same_below = xp.zeros_like(alive)
        same_below[:, :, 1:] = (species[:, :, 1:] == species[:, :, :-1]) & alive[:, :, :-1]
        # стебель: над водой живёт светом (ген up), под водой — ещё и тем,
        # насколько вид приспособлен к воде, иначе из озера не выбраться
        stem = above & same_below
        stem_niche = xp.where(zz < surface_h,
                              xp.maximum(gene("up") * 0.5, gene("water")),
                              gene("up"))
        niche = xp.where(stem, stem_niche, niche)
        factor = cfg.niche_floor + (1.0 - cfg.niche_floor) * xp.clip(niche, 0.0, 1.0)

        # --- бактерии удобряют почву, симбионты живут только на удобренной ----
        fert = state["fert"]
        enrich2d = (gene("enrich") * on_ground).max(axis=2)
        # бактерия удобряет свой столбец и, слабее, соседние: «заражение» почвы
        # расползается, иначе растению-симбионту негде было бы сесть
        spread = enrich2d
        for dx, dy in self.DIRS4:
            spread = xp.maximum(spread, self._shift2(enrich2d, dx, dy, xp) * 0.6)
        enrich2d = spread
        fert = xp.clip(fert * (1.0 - cfg.fert_decay) + cfg.fert_gain * enrich2d, 0.0, 1.0)
        fert = xp.where(soil_h > 0, fert, 0.0)          # нет почвы — нечего удобрять
        state["fert"] = fert
        sym = gene("symbiont")
        factor = factor * (1.0 - sym * (1.0 - cfg.symbiont_floor) * (1.0 - fert[:, :, None]))
        # бактерия и сама неплохо себя чувствует на удобренной почве
        factor = factor * (1.0 + cfg.fert_bonus * fert[:, :, None] * gene("soil"))

        gain = cfg.light_gain * gene("light") * L * factor

        # --- эрозия камня: лишайник на голом мокром камне ----------------------
        bare_stone = on_ground & (~has_soil)
        wet3 = xp.minimum(water_h[:, :, None].astype(xp.float32) + state["flow"][:, :, None], 1.0)
        # на голом камне «мокро» и от дождя, и от потока рядом
        p_er = cfg.erode_rate * gene("erode") * xp.maximum(wet3, 0.15) * bare_stone
        eroding = rng.random(species.shape) < p_er
        er2d = eroding.any(axis=2) & (stone_h > 1)
        gain = gain + xp.where(eroding & er2d[:, :, None], cfg.erode_gain, 0.0)
        # ОБМЕН НЕРАВНОЦЕННЫЙ: клетка камня даёт несколько клеток почвы
        stone_h = stone_h - er2d.astype(stone_h.dtype)
        soil_h = soil_h + (er2d.astype(soil_h.dtype) * cfg.soil_per_stone)

        energy = xp.where(alive, energy + gain - gene("metabolism"), 0.0)
        energy = xp.minimum(energy, xp.where(alive, G[idx, IDX["repro"]] * cfg.energy_cap, 0.0))
        age = xp.where(alive, age + 1, 0)

        # --- смерть ------------------------------------------------------------
        nb6 = self._count6(alive, xp)
        dead = alive & ((energy <= 0) | (age > cfg.lifespan) | (nb6 > cfg.crowd_max)
                        | (rng.random(species.shape) < cfg.p_shock))
        species = xp.where(dead, xp.uint8(0), species)
        energy = xp.where(dead, 0.0, energy)
        age = xp.where(dead, 0, age)
        alive = species > 0

        # --- подложка поднялась/просела: клетки едут вместе с ней --------------
        new_ground = stone_h + soil_h
        species, energy, age = self._settle(species, energy, age, ground, new_ground, cfg, xp)
        ground = new_ground
        alive = species > 0

        state.update(species=species, energy=energy, age=age,
                     stone_h=stone_h, soil_h=soil_h, gen=gen)

        # --- деление и мутации --------------------------------------------------
        idx = xp.clip(species.astype(xp.int32) - 1, 0, MAX_SPECIES - 1)
        repro = xp.where(alive, G[idx, IDX["repro"]], 1e9).astype(xp.float32)
        parents = alive & (energy > repro)
        if bool(parents.any()):
            stress = xp.clip(1.0 - xp.clip(niche, 0, 1), 0.0, 1.0)
            p_mut = xp.clip(xp.where(alive, G[idx, IDX["mut"]], 0.0)
                            + cfg.stress_mut * stress, 0.0, 0.9)
            self._reproduce(state, cfg, xp, rng, parents, p_mut, ground, gen)

        self._sync_volumes(state, cfg, xp)
        return self._pops(state, cfg, gen)

    # ------------------------------------------------------------- помощники
    @staticmethod
    def _shift3(a, d, xp, fill=0):
        from ..motion import shift
        return shift(a, d, xp, fill)

    @staticmethod
    def _count6(alive, xp):
        a = alive.astype(xp.float32)
        c = xp.zeros_like(a)
        c[1:] += a[:-1]; c[:-1] += a[1:]
        c[:, 1:] += a[:, :-1]; c[:, :-1] += a[:, 1:]
        c[:, :, 1:] += a[:, :, :-1]; c[:, :, :-1] += a[:, :, 1:]
        return c

    def _settle(self, species, energy, age, old_ground, new_ground, cfg, xp):
        """Почва поднялась — клетка едет вверх; камень съеден — падает вниз."""
        n = cfg.n
        zz = xp.arange(n)[None, None, :]
        rise = new_ground - old_ground
        for k in range(1, cfg.soil_per_stone + 1):
            move = (species > 0) & (zz == (new_ground - k)[:, :, None]) & (rise >= k)[:, :, None]
            if not bool(move.any()):
                continue
            tgt = xp.zeros_like(move); tgt[:, :, k:] = move[:, :, :-k]
            tgt = tgt & (species == 0)
            src = xp.zeros_like(move); src[:, :, :-k] = tgt[:, :, k:]
            species = xp.where(tgt, xp.roll(species, k, axis=2), species)
            energy = xp.where(tgt, xp.roll(energy, k, axis=2), energy)
            age = xp.where(tgt, xp.roll(age, k, axis=2), age)
            species = xp.where(src, xp.uint8(0), species)
            energy = xp.where(src, 0.0, energy)
        # висящие над новой землёй падают на неё
        for _ in range(2):
            alive = species > 0
            support = xp.zeros_like(alive); support[:, :, 0] = True
            support[:, :, 1:] = alive[:, :, :-1] | (zz[:, :, 1:] <= new_ground[:, :, None])
            fall = alive & ~support
            if not bool(fall.any()):
                break
            tgt = xp.zeros_like(fall); tgt[:, :, :-1] = fall[:, :, 1:]
            tgt = tgt & ~alive
            src = xp.zeros_like(fall); src[:, :, 1:] = tgt[:, :, :-1]
            species = xp.where(tgt, xp.roll(species, -1, axis=2), species)
            energy = xp.where(tgt, xp.roll(energy, -1, axis=2), energy)
            age = xp.where(tgt, xp.roll(age, -1, axis=2), age)
            species = xp.where(src, xp.uint8(0), species)
            energy = xp.where(src, 0.0, energy)
        # погребённые окончательно
        buried = (species > 0) & (zz < new_ground[:, :, None])
        species = xp.where(buried, xp.uint8(0), species)
        energy = xp.where(buried, 0.0, energy)
        age = xp.where(buried, 0, age)
        return species, energy, age

    def _reproduce(self, state, cfg, xp, rng, parents, p_mut, ground, gen):
        """По земле — на поверхность соседнего столбца; вверх — по гену up."""
        from ..motion import shift
        n = cfg.n
        species, energy, age = state["species"], state["energy"], state["age"]
        G = state["genomes"]
        zz = xp.arange(n)[None, None, :]
        ground3 = ground[:, :, None]
        alive = species > 0
        idx = xp.clip(species.astype(xp.int32) - 1, 0, MAX_SPECIES - 1)
        up_p = xp.where(alive, G[idx, IDX["up"]], 0.0).astype(xp.float32)
        repro_here = xp.where(alive, G[idx, IDX["repro"]], 0.0).astype(xp.float32)
        weak = alive & (energy < cfg.takeover * repro_here)
        on_ground = alive & (zz == ground3)

        key = rng.random(species.shape).astype(xp.float32) + 1.0
        mutate = rng.random(species.shape) < p_mut
        go_up = parents & (rng.random(species.shape) < up_p)
        lateral = parents & ~go_up & on_ground

        child_mut = xp.zeros_like(alive)
        child_parent = xp.zeros(species.shape, dtype=xp.uint8)
        placed = xp.zeros_like(alive)

        # --- по земле (2D) ----------------------------------------------------
        def col(a):
            return (a & on_ground).any(axis=2) if a.dtype == bool else (a * on_ground).sum(axis=2)
        par2 = col(lateral)
        if bool(par2.any()):
            key2, sp2 = col(key), col(species.astype(xp.float32)).astype(xp.uint8)
            en2, mut2 = col(energy), col(mutate)
            weak2, occupied = col(weak), col(alive)
            dir2 = (rng.random((n, n)) * 4).astype(xp.int32)
            claims, win = [], xp.zeros((n, n), dtype=xp.float32)
            for k, (dx, dy) in enumerate(self.DIRS4):
                src_par = self._shift2(par2 & (dir2 == k), -dx, -dy, xp, fill=False)
                src_sp = self._shift2(sp2, -dx, -dy, xp)
                ok = src_par & (~occupied | (weak2 & (sp2 != src_sp)))
                claims.append(xp.where(ok, self._shift2(key2, -dx, -dy, xp), 0.0))
                win = xp.maximum(win, claims[-1])
            done2 = xp.zeros((n, n), dtype=bool)
            for k, (dx, dy) in enumerate(self.DIRS4):
                took2 = (claims[k] > 0) & (claims[k] == win)
                src2 = self._shift2(took2, dx, dy, xp, fill=False) & par2 & ~done2
                took2 = self._shift2(src2, -dx, -dy, xp, fill=False)
                if not bool(took2.any()):
                    continue
                p_sp = self._shift2(sp2, -dx, -dy, xp)
                p_en = self._shift2(en2, -dx, -dy, xp)
                p_mut2 = self._shift2(mut2, -dx, -dy, xp, fill=False)
                tgt3 = took2[:, :, None] & (zz == ground3)
                par3 = src2[:, :, None] & on_ground
                species = xp.where(tgt3, p_sp[:, :, None], species)
                energy = xp.where(tgt3, (p_en * 0.5)[:, :, None], energy)
                age = xp.where(tgt3, 0, age)
                energy = xp.where(par3, energy * 0.5, energy)
                child_mut = child_mut | (tgt3 & p_mut2[:, :, None])
                child_parent = xp.where(tgt3, p_sp[:, :, None], child_parent)
                placed = placed | par3
                done2 = done2 | src2
            alive = species > 0

        # --- вверх (стебель) ---------------------------------------------------
        if bool(go_up.any()):
            free = ~alive & (zz > ground3)
            d = (0, 0, 1)
            back = (0, 0, -1)
            ok = go_up & ~placed & shift(free, back, xp, fill=False)
            took = shift(ok, d, xp, fill=False)
            if bool(took.any()):
                half = xp.where(ok, energy * 0.5, 0.0)
                species = xp.where(took, shift(species, d, xp), species)
                energy = xp.where(took, shift(half, d, xp), energy)
                age = xp.where(took, 0, age)
                energy = xp.where(ok, energy * 0.5, energy)
                child_mut = child_mut | (took & shift(mutate, d, xp, fill=False))
                child_parent = xp.where(took, shift(species, d, xp), child_parent)

        state["species"], state["energy"], state["age"] = species, energy, age
        self._speciate(state, cfg, xp, child_mut, child_parent, gen)

    def _speciate(self, state, cfg, xp, child_mut, child_parent, gen):
        from ..backend import to_cpu
        if cfg.max_new_species <= 0 or not bool(child_mut.any()) or not state["free_ids"]:
            return
        cand = np.argwhere(to_cpu(child_mut))
        rng_cpu = state["rng_cpu"]
        rng_cpu.shuffle(cand)
        reg, G_cpu, born = state["registry"], cfg.genomes, 0
        species = state["species"]
        for (x, y, z) in cand[: cfg.max_new_species * 3]:
            if born >= cfg.max_new_species or not state["free_ids"]:
                break
            parent = int(to_cpu(child_parent[x, y, z]))
            if parent == 0:
                continue
            g = G_cpu[parent - 1].copy()
            gi = int(rng_cpu.integers(len(GENES)))
            lo, hi, _ = RANGES[GENES[gi]]
            g[gi] = float(np.clip(g[gi] + rng_cpu.normal(0, cfg.mut_scale * (hi - lo)), lo, hi))
            if abs(g[gi] - G_cpu[parent - 1][gi]) < cfg.min_change * (hi - lo):
                continue
            sid = state["free_ids"].pop(0)
            G_cpu[sid - 1] = g
            reg[sid] = {"parent": parent, "born": gen, "died": None,
                        "genome": g.tolist(), "peak": 1, "changed": GENES[gi]}
            species[x, y, z] = sid
            born += 1
        if born:
            state["genomes"] = xp.asarray(G_cpu)

    def _pops(self, state, cfg, gen):
        from ..backend import to_cpu
        counts = np.bincount(to_cpu(state["species"]).ravel(), minlength=MAX_SPECIES + 1)[1:]
        reg = state["registry"]
        for sid in list(reg):
            c = int(counts[sid - 1])
            reg[sid]["peak"] = max(reg[sid]["peak"], c)
            if c == 0 and gen > 0 and reg[sid]["died"] is None:
                reg[sid]["died"] = gen
                state["lineage"].append({"id": sid, **reg.pop(sid)})
                state["free_ids"].append(sid)
                cfg.genomes[sid - 1] = 0
        return [int(c) for c in counts]

    # ------------------------------------------------------------- интерфейс
    def species_names(self, cfg, state=None):
        names = [""] * MAX_SPECIES
        if state is None:
            names[0] = "лишайник"
            return names
        for sid, r in state["registry"].items():
            g = cfg.genomes[sid - 1]
            kind = self._kind(g)
            names[sid - 1] = ("лишайник" if sid == 1 else
                              f"#{sid} {kind} ← #{r['parent']} ({r.get('changed', '?')}, пок. {r['born']})")
        return names

    @staticmethod
    def _kind(g):
        up, sym, enr = float(g[IDX["up"]]), float(g[IDX["symbiont"]]), float(g[IDX["enrich"]])
        st, so, wa = float(g[IDX["stone"]]), float(g[IDX["soil"]]), float(g[IDX["water"]])
        if up > 0.3:
            return "растение" if (sym > 0.25 or so > 0.4) else "башня"
        if enr > 0.3 and so > 0.25:
            return "бактерия"
        if wa > 0.5 and wa >= max(st, so):
            return "водоросль"
        if so > 0.5 and st > 0.5:
            return "универсал"
        if so > 0.45:
            return "почвенный"
        return "накипь"

    def species_colors(self, cfg, state=None):
        cols = []
        for i in range(MAX_SPECIES):
            g = cfg.genomes[i]
            cols.append("#ffffff" if i == 0 else (genome_color(g, i + 1) if g.any() else "#888888"))
        return cols

    def world_params(self):
        return self.WORLD_PARAMS

    def to_json(self, cfg, state=None):
        ids = sorted(state["registry"]) if state else [1]
        names = self.species_names(cfg, state)
        cols = self.species_colors(cfg, state)
        return {
            "engine": self.name, "fields": list(GENES), "labels": LABELS,
            "ranges": {k: list(v) for k, v in RANGES.items()},
            "ids": ids,
            "names": [names[i - 1] for i in ids],
            "colors": [cols[i - 1] for i in ids],
            "genomes": [np.asarray(cfg.genomes[i - 1]).tolist() for i in ids],
            "world": {k: getattr(cfg, k) for k in self.WORLD_PARAMS},
            "world_labels": self.WORLD_LABELS, "world_ranges": self.WORLD_RANGES,
            "starters": self.starters_json(cfg),
            "reseed": seeding_json(cfg),
            "species_total": (len(state["lineage"]) + len(state["registry"])) if state else 1,
        }

    def fork_species(self, cfg, state, sid, genome, xp, gen=0, share=0.3):
        return fork_dynamic(state, cfg, xp, sid, genome, gen, share)

    def apply_genomes(self, cfg, state, genomes, xp, ids=None):
        g = np.asarray(genomes, dtype=np.float32)
        ids = list(ids) if ids else sorted(state["registry"])
        if g.shape != (len(ids), len(GENES)):
            raise ValueError(f"геномы: ожидается {len(ids)} × {len(GENES)}")
        for row, sid in zip(g, ids):
            cfg.genomes[sid - 1] = row
            if sid in state["registry"]:
                state["registry"][sid]["genome"] = row.tolist()
        state["genomes"] = xp.asarray(cfg.genomes)

    def randomize(self, cfg, rng):
        g = np.zeros_like(cfg.genomes)
        g[0] = LICHEN
        for name, i in IDX.items():
            lo, hi, _ = RANGES[name]
            g[0, i] = rng.uniform(lo, hi)
        g[0, IDX["stone"]] = rng.uniform(0.6, 1.0)     # старт всё же на камне
        g[0, IDX["soil"]] = rng.uniform(0.0, 0.2)
        g[0, IDX["water"]] = 0.0
        g[0, IDX["up"]] = 0.0
        g[0, IDX["metabolism"]] = rng.uniform(0.03, 0.1)
        g[0, IDX["repro"]] = rng.uniform(1.2, 2.5)
        return g


RULES = TerraRules()
