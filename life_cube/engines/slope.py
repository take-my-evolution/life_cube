"""Движок «склон» — сукцессия от голого камня к лесу в низинах.

Мир как его описал пользователь:

  * Подложка — две карты высот: камень и почва поверх него. Поверхность
    столбца = stone_h + soil_h.
  * **Мох** растёт ТОЛЬКО на голом камне и ТОЛЬКО на поверхности. Он
    медленно точит камень в почву (ген `erode`) — это единственный источник
    почвы в мире.
  * **Почва скатывается вниз по склону**: с крутых мест уходит к нижнему
    соседу. Поэтому на вершинах почва не накапливается (мох точит, почва
    съезжает) и там остаётся мох, а в низинах почва копится — и туда
    приходят растения. Сукцессия получается сама, её никто не программировал.
  * **Трава** и **дерево** растут ТОЛЬКО на почве, на траве не растёт ничто.
    Трава стелется по поверхности (ген `trunk` = 0). Дерево сначала гонит
    ствол строго вверх на `trunk` клеток, и только выше начинает ветвиться.
  * Дерево сильно поглощает свет (`absorb`), поэтому под кроной темно, а
    трава живёт светом — тень душит траву тем сильнее, чем плотнее крона.
    Отдельного «затенения» не написано: это обычное поле света из fields.py.
  * **Травоядные и хищники ходят ПО ПОВЕРХНОСТИ** — по камню, почве и траве.
    Забраться на ствол или в крону они не могут: животное живёт на высоте
    walk_z(x,y) и шагает только к соседнему столбцу, чья поверхность не выше
    чем на ступеньку и не занята деревом.

Экономика роста — та же, что в «экологии» после v0.11: клетка копит
`ресурс - metabolism`, новая клетка стоит `repro * mass * growth_cost`,
масса задаёт и цену постройки, и пищевую ценность, и вклад в биомассу.
"""

import colorsys
from dataclasses import dataclass, field

import numpy as np

from . import Rules
from ..fields import light_field

GENES = ("light", "absorb", "substrate", "trunk", "branch", "erode",
         "metabolism", "repro", "lifespan", "mass",
         "hunt", "trophic", "speed", "sense", "armor")
IDX = {g: i for i, g in enumerate(GENES)}

LABELS = {
    "light": "усвоение света", "absorb": "тень, которую даёт",
    "substrate": "подложка: 0 камень, 1 почва", "trunk": "ствол: клеток вверх до ветвей",
    "branch": "ветвление", "erode": "точит камень в почву",
    "metabolism": "обмен веществ", "repro": "порог деления", "lifespan": "предел возраста",
    "mass": "масса клетки", "hunt": "сила атаки", "trophic": "уровень: 0 раст / 1 трав / 2 хищ",
    "speed": "скорость (0 = сидячий)", "sense": "радиус чутья", "armor": "броня",
}
RANGES = {
    "light": (0.0, 1.0, 0.01), "absorb": (0.0, 1.0, 0.01), "substrate": (0.0, 1.0, 1.0),
    "trunk": (0, 12, 1), "branch": (0.0, 2.0, 0.05), "erode": (0.0, 1.0, 0.01),
    "metabolism": (0.0, 0.2, 0.002), "repro": (0.5, 12.0, 0.1), "lifespan": (0, 4000, 50),
    "mass": (0.1, 8.0, 0.1), "hunt": (0.0, 1.0, 0.01), "trophic": (0, 2, 1),
    "speed": (0, 3, 1), "sense": (0, 8, 1), "armor": (0.0, 0.95, 0.05),
}

NAMES = ("мох", "трава", "дерево", "травоядное", "хищник")
COLORS = ("#9fb3a8", "#7bd94a", "#2e8b3d", "#4a9ef2", "#f24a9e")

#                 light absorb subst trunk branch erode metab repro  life mass hunt tro spd sns armor
GENOMES = np.array([
    [0.55, 0.10, 0.0,  0,  0.40, 0.30, 0.006,  3.0,    0, 0.5, 0.0, 0, 0, 0, 0.90],  # мох
    [0.85, 0.22, 1.0,  0,  0.90, 0.00, 0.200,  0.8,  300, 1.0, 0.0, 0, 0, 0, 0.35],  # трава
    [0.95, 0.75, 1.0,  5,  1.20, 0.00, 0.050,  1.6, 1200, 5.0, 0.0, 0, 0, 0, 0.95],  # дерево
    [0.00, 0.00, 0.0,  0,  0.00, 0.00, 0.015,  6.0,  400, 4.0, 0.15, 1, 1, 5, 0.45],  # травоядное
    [0.00, 0.00, 0.0,  0,  0.00, 0.00, 0.120,  8.0,  900, 6.0, 0.30, 2, 2, 7, 0.05],  # хищник
], dtype=np.float32)

N_SPECIES = len(NAMES)

# Кто в кого может ошибиться при делении. Это и есть «засев»: новая жизнь
# приходит не кубиками, брошенными на карту, а сбоем размножения у того, кто
# уже живёт. Пары — только по соседним ярусам: растение может дать травоядное,
# травоядное — хищника, трава — дерево. Через ярус не прыгают.
# Баланс намеренно поставлен НА КРАЙ: хищник достаточно удачлив, чтобы выесть
# травоядных и издохнуть следом. Замер на 2500 поколений (n=48, сиды 7 и 11):
# травоядные срываются в ноль 7–11 раз, хищник 4–5, и каждый раз мутация
# возвращает их — к этому времени корма снова хватает, чтобы вид разошёлся.
# Растения при этом не падают: они основание пирамиды, а не её вершина.

# Кто кого ест. Раньше добычей считалось всё, что ярусом ниже, — а на нулевом
# ярусе вместе с травой и деревом стоит МОХ, и травоядные паслись на камнях.
# Мох в этом мире — не корм, а порода: он точит камень и готовит почву, и
# съесть его некому (в природе лишайник и правда почти никем не поедается).
DIET = {
    4: (2, 3),      # травоядное — трава и дерево
    5: (4,),        # хищник — травоядное
}

MUTATIONS = {
    1: (2, 4),      # мох        -> трава, травоядное
    2: (1, 3, 4),   # трава      -> мох, дерево, травоядное
    3: (2, 4),      # дерево     -> трава, травоядное
    4: (5,),        # травоядное -> хищник
}


@dataclass
class SlopeConfig:
    n: int = 96
    gens: int = 400
    seed_world: int = 20260902
    seed_mut: int = 20260902

    # рельеф
    floor: int = 3                  # плоское каменное дно
    stone_fraction: float = 0.42    # высота горы (доля куба)
    hill_radius: float = 0.75
    hill_roughness: float = 0.30
    soil_start: float = 1.0         # немного почвы в низинах на старте

    # дождь и вода
    rain_rate: float = 0.05
    rain_amount: float = 0.5
    rain_decay: float = 0.97
    water_flow: float = 0.45        # доля воды, стекающей к нижнему соседу

    # почва
    soil_slide: float = 0.28        # вероятность, что единица почвы съедет вниз
    slide_drop: int = 2             # с какого перепада поверхности почва едет
    erode_rate: float = 0.02        # базовая скорость превращения камня в почву

    # жизнь
    growth_cost: float = 5.0
    energy_cap: float = 1.5
    light_gain: float = 0.85
    water_gain: float = 0.5         # прибавка к ресурсу от воды
    crown_light: float = 0.35   # ниже этого света крона не разрастается
    trunk_spacing: int = 3      # ближе этого стволы друг к другу не встают
    log_stone: bool = True      # упавший ствол превращает почву в камень
    log_max: int = 8            # длиннее этого полоса не ложится
    seed_range: int = 6         # как далеко дерево роняет семя
    seed_fall: float = 0.004    # шанс всхода на подходящем месте
    seed_maturity: float = 1.2  # во сколько цен клетки кошелёк даёт семя
    mutate_rate: float = 0.0015  # базовый шанс ошибки при делении
    mutate_rescue: float = 10.0  # во сколько раз чаще, если ниша пуста
    crowd_max: int = 5
    p_shock: float = 0.0006
    start_energy: float = 4.0
    eat_efficiency: float = 0.8
    move_noise: float = 0.15
    seed_density: float = 0.05      # доля столбцов под стартовый мох
    seed_tree: float = 0.010        # доля почвенных столбцов под деревья
    seed_animals: float = 0.020     # доля столбцов под травоядных

    start_species: tuple = ()
    reseed: bool = False        # засев теперь идёт мутацией, не кубиками на карте
    reseed_on_extinction: bool = True
    reseed_species: bool = True     # вернуть вид, выпавший из цепи
    reseed_every: int = 150
    reseed_count: int = 120

    genomes: np.ndarray = field(default_factory=lambda: GENOMES.copy())

    @property
    def n_species(self):
        return N_SPECIES


def hill_relief(cfg, rng):
    """Гора: широкий холм со складками. Высоты — целые (карта высот камня)."""
    n = cfg.n
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cx, cy = n / 2, n / 2
    r = np.hypot(xx - cx, yy - cy) / (n / 2 * cfg.hill_radius)
    bump = np.clip(1.0 - r ** 2, 0, 1) ** 1.1
    H = max(4.0, cfg.stone_fraction * n) - cfg.floor
    rough = (np.sin(xx / (n / 8.0) + 1.3) * np.cos(yy / (n / 6.0))
             + 0.5 * np.sin((xx + yy) / (n / 12.0)))
    h = cfg.floor + H * bump * (1 + cfg.hill_roughness * rough * bump)
    h = h + rng.normal(0, 0.4, (n, n)) * bump
    return np.clip(np.round(h), cfg.floor, n - 8).astype(np.int32)


class SlopeRules(Rules):
    name = "slope"
    title = "Склон: мох точит камень, почва съезжает, лес в низинах"
    summary = ("Мох живёт только на голом камне и точит его в почву. Почва "
               "скатывается по склону, поэтому на вершинах остаётся мох, а в "
               "низинах копится земля и приходят трава и деревья. Деревья "
               "гонят ствол вверх, потом ветвятся и затеняют траву. "
               "Травоядные и хищники ходят только по поверхности.")
    doc = "docs/engines/slope.md"
    Config = SlopeConfig
    heightmaps = True
    terrain_changes = True
    can_seed = True

    WORLD_PARAMS = ("n", "seed_world", "seed_mut", "stone_fraction", "hill_radius",
                    "hill_roughness", "rain_rate", "rain_amount", "rain_decay",
                    "water_flow", "soil_slide", "erode_rate", "growth_cost",
                    "soil_start", "crown_light", "trunk_spacing", "seed_range", "seed_fall", "seed_maturity",
                    "mutate_rate", "mutate_rescue",
                    "seed_density", "seed_tree",
                    "seed_animals", "p_shock")
    WORLD_RANGES = {"n": [32, 256, 32], "stone_fraction": [0.15, 0.7, 0.01],
                    "hill_radius": [0.3, 1.2, 0.05], "hill_roughness": [0.0, 0.8, 0.02],
                    "rain_rate": [0.0, 0.3, 0.005], "rain_amount": [0.0, 2.0, 0.05],
                    "rain_decay": [0.8, 0.999, 0.001], "soil_slide": [0.0, 1.0, 0.02],
                    "erode_rate": [0.0, 0.2, 0.002], "growth_cost": [1.0, 20.0, 0.5],
                    "seed_density": [0.005, 0.4, 0.005], "seed_tree": [0.0, 0.1, 0.002],
                    "seed_animals": [0.0, 0.1, 0.002], "soil_start": [0.0, 3.0, 0.25], "crown_light": [0.05, 0.8, 0.05], "trunk_spacing": [0, 8, 1],
                    "seed_range": [0, 16, 1], "seed_fall": [0.0, 0.05, 0.001],
                    "seed_maturity": [1.0, 6.0, 0.1],
                    "mutate_rate": [0.0, 0.02, 0.0005],
                    "mutate_rescue": [1.0, 200.0, 5.0],
                    "water_flow": [0.0, 0.9, 0.05], "p_shock": [0.0, 0.01, 0.0002]}
    WORLD_LABELS = {"n": "размер куба", "seed_world": "сид мира", "seed_mut": "сид жизни",
                    "stone_fraction": "высота горы", "hill_radius": "радиус горы",
                    "hill_roughness": "складки склона", "rain_rate": "дождь: доля столбцов",
                    "rain_amount": "дождь: сколько льёт", "rain_decay": "высыхание",
                    "soil_slide": "скатывание почвы", "erode_rate": "мох точит камень",
                    "growth_cost": "цена роста (общая)", "water_flow": "вода стекает вниз",
                    "soil_start": "почвы в низинах на старте",
                    "crown_light": "света хватает кроне",
                    "trunk_spacing": "просвет между стволами",
                    "seed_range": "далеко ли летят семена",
                    "seed_fall": "шанс всхода семени",
                    "seed_maturity": "зрелость для семян",
                    "mutate_rate": "шанс мутации при делении",
                    "mutate_rescue": "во сколько раз чаще в пустой нише",
                    "seed_density": "плотность засева мха",
                    "seed_tree": "плотность засева деревьев",
                    "seed_animals": "плотность засева зверей",
                    "p_shock": "выветривание"}

    # ------------------------------------------------------------- мир
    def init_state(self, cfg, xp):
        rng = np.random.default_rng(cfg.seed_world)
        n = cfg.n
        stone_h = hill_relief(cfg, rng)
        soil_h = np.zeros((n, n), np.int32)
        if cfg.soil_start > 0:
            # Тонкий слой почвы в низинах, дальше её делает мох. Слой именно
            # ТОНКИЙ: при трёх клетках подножие горы тонуло в равнине, и гора
            # выглядела наполовину закопанной.
            rel = (stone_h - stone_h.min()) / max(stone_h.max() - stone_h.min(), 1)
            soil_h = np.round(cfg.soil_start * (1 - rel) ** 2).astype(np.int32)
        species = np.zeros((n, n, n), np.uint8)
        energy = np.zeros((n, n, n), np.float32)
        surf = stone_h + soil_h
        self._seed_start(cfg, rng, species, energy, stone_h, soil_h, surf)
        state = {
            "species": xp.asarray(species),
            "energy": xp.asarray(energy),
            "age": xp.zeros((n, n, n), dtype=xp.int32),
            "stone_h": xp.asarray(stone_h),
            "soil_h": xp.asarray(soil_h),
            "water_h": xp.zeros((n, n), dtype=xp.int32),
            "wet": xp.zeros((n, n), dtype=xp.float32),
            "genomes": xp.asarray(cfg.genomes),
            "rng": xp.random.default_rng(cfg.seed_mut),
            "rng_rain": xp.random.default_rng(cfg.seed_mut ^ 0x7a17a170),
            "gen": 0,
        }
        self._sync_volumes(state, cfg, xp)
        return state, stone_h.copy()

    def _seed_start(self, cfg, rng, species, energy, stone_h, soil_h, surf):
        """Стартовый засев: мох на голом камне, трава и деревья на почве,
        травоядные и хищники — там, где можно ходить.

        Мир начинается уже населённым всеми четырьмя ярусами: голая гора со
        мхом наверху, почва с травой и лесом в низинах, звери на них. Иначе
        первые несколько тысяч поколений в кадре только мох: почвы в начале
        нет, а точит он камень медленно.
        """
        n = cfg.n
        bare = soil_h == 0
        ground = soil_h > 0

        def put(mask, s, dens, z=None):
            xs, ys = np.nonzero(mask & (rng.random((n, n)) < dens))
            if not len(xs):
                return
            zz = surf[xs, ys] if z is None else z[xs, ys]
            zz = np.clip(zz, 0, n - 1)
            free = species[xs, ys, zz] == 0
            xs, ys, zz = xs[free], ys[free], zz[free]
            species[xs, ys, zz] = s
            g = cfg.genomes[s - 1]
            if float(g[IDX["speed"]]) > 0:
                # у зверя repro — это порог энергии, а не множитель цены клетки:
                # сажаем сытым, но не готовым тут же делиться
                e = float(g[IDX["repro"]]) * 0.7
            else:
                e = float(g[IDX["repro"]]) * max(float(g[IDX["mass"]]), 0.1) * cfg.growth_cost * 0.6
            energy[xs, ys, zz] = max(e, 1.0)

        put(bare, 1, cfg.seed_density)                    # мох
        put(ground, 2, cfg.seed_density * 3)              # трава
        # деревья и на старте расставляем с просветом, иначе первое поколение
        # леса стоит стеной, а правило просвета видно только у их потомков
        tree_spots = ground & (rng.random((n, n)) < cfg.seed_tree)
        sp2 = int(cfg.trunk_spacing)
        taken = np.zeros((n, n), bool)
        for x, y in np.argwhere(tree_spots):
            if taken[max(x - sp2, 0):x + sp2 + 1, max(y - sp2, 0):y + sp2 + 1].any():
                continue
            z = int(np.clip(surf[x, y], 0, n - 1))
            if species[x, y, z]:
                continue
            g3 = cfg.genomes[2]
            species[x, y, z] = 3
            energy[x, y, z] = max(float(g3[IDX["repro"]]) * max(float(g3[IDX["mass"]]), 0.1)
                                  * cfg.growth_cost * 0.6, 1.0)
            taken[x, y] = True
        walk = surf + (species[np.arange(n)[:, None], np.arange(n)[None, :],
                               np.clip(surf, 0, n - 1)] > 0).astype(surf.dtype)
        put(~np.zeros((n, n), bool), 4, cfg.seed_animals, z=walk)        # травоядное
        put(~np.zeros((n, n), bool), 5, cfg.seed_animals * 0.5, z=walk)  # хищник

    def _sync_volumes(self, state, cfg, xp):
        n = cfg.n
        zz = xp.arange(n)[None, None, :]
        sh, so = state["stone_h"][:, :, None], state["soil_h"][:, :, None]
        state["stone"] = zz < sh
        state["soil"] = (zz >= sh) & (zz < sh + so)

    # ------------------------------------------------------------- утилиты
    @staticmethod
    def _shift2(a, dx, dy, xp, fill=0):
        out = xp.full_like(a, fill)
        sx = slice(max(dx, 0), a.shape[0] + min(dx, 0))
        sy = slice(max(dy, 0), a.shape[1] + min(dy, 0))
        tx = slice(max(-dx, 0), a.shape[0] + min(-dx, 0))
        ty = slice(max(-dy, 0), a.shape[1] + min(-dy, 0))
        out[sx, sy] = a[tx, ty]
        return out

    DIRS2 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    # перестановки ходов зверя (четыре стороны и «стоять»): перебирать их
    # всегда в одном порядке — значит систематически предпочитать первый
    MOVES = tuple(tuple(p) for p in __import__("itertools").permutations(
        ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))))

    def _downhill(self, surface, xp):
        """Куда «вниз» с каждого столбца: индекс направления и перепад."""
        big = surface.max() + 1000
        best_drop = xp.zeros_like(surface)
        best_dir = xp.full(surface.shape, -1, dtype=xp.int32)
        for k, (dx, dy) in enumerate(self.DIRS2):
            nsurf = self._shift2(surface, dx, dy, xp, fill=big)
            drop = surface - nsurf
            better = drop > best_drop
            best_drop = xp.where(better, drop, best_drop)
            best_dir = xp.where(better, k, best_dir)
        return best_dir, best_drop

    # ------------------------------------------------------------- шаг
    def step(self, state, cfg, xp, correlate, gen):  # noqa: C901
        n = cfg.n
        rng, rng_rain = state["rng"], state["rng_rain"]
        G = state["genomes"]
        species, energy, age = state["species"], state["energy"], state["age"]
        stone_h, soil_h, wet = state["stone_h"], state["soil_h"], state["wet"]
        zz = xp.arange(n)[None, None, :]
        surface = stone_h + soil_h
        alive = species > 0
        idx = xp.clip(species.astype(xp.int32) - 1, 0, N_SPECIES - 1)

        def gene(name):
            return xp.where(alive, G[idx, IDX[name]], 0.0).astype(xp.float32)

        mobile = np.asarray(cfg.genomes)[:, IDX["speed"]] > 0
        plant_ids = [s for s in range(1, N_SPECIES + 1) if not mobile[s - 1]]
        anim_ids = [s for s in range(1, N_SPECIES + 1) if mobile[s - 1]]
        is_anim = xp.zeros(species.shape, dtype=bool)
        for s in anim_ids:
            is_anim = is_anim | (species == s)
        plants = alive & ~is_anim

        # --- 1. дождь --------------------------------------------------------
        drop = rng_rain.random((n, n)) < cfg.rain_rate
        wet = wet * cfg.rain_decay + xp.where(drop, xp.float32(cfg.rain_amount), xp.float32(0.0))
        # вода стекает к нижнему соседу
        bdir, bdrop = self._downhill(surface, xp)
        can = bdrop >= 1
        out = xp.where(can, wet * cfg.water_flow, 0.0)
        wet = wet - out
        for k, (dx, dy) in enumerate(self.DIRS2):
            wet = wet + self._shift2(xp.where(bdir == k, out, 0.0), -dx, -dy, xp)

        # --- 2. мох точит камень в почву -------------------------------------
        # Это единственный источник почвы в мире. Мох работает только там, где
        # он стоит на ГОЛОМ камне (soil_h == 0) и только на поверхности.
        on_surface = plants & (zz == surface[:, :, None])
        bare = (soil_h == 0)[:, :, None]
        er = gene("erode") * on_surface * bare
        p_er = cfg.erode_rate * er * xp.minimum(wet[:, :, None], 1.0)
        eroding = (rng.random(species.shape) < p_er).any(axis=2) & (stone_h > cfg.floor)
        stone_h = stone_h - eroding.astype(stone_h.dtype)
        soil_h = soil_h + eroding.astype(soil_h.dtype)

        # --- 3. почва скатывается вниз по склону ------------------------------
        surface = stone_h + soil_h
        bdir, bdrop = self._downhill(surface, xp)
        # Единица почвы, уехавшая при перепаде 1, делает перепад −1: сосед
        # становится выше, и на следующем шаге та же единица едет обратно. На
        # ровном дне (где склона нет вовсе) так «кипело» 462 столбца из 2440 за
        # шаг, а из 15 384 перемещений почвы за 30 шагов чистыми были 1233 —
        # 92 % движения было дрожью на месте. Порог 2 — это угол естественного
        # откоса: перенос ровно выравнивает столбцы и обратного хода не создаёт.
        go = (bdrop >= max(2, int(cfg.slide_drop))) & (soil_h > 0) \
            & (rng.random((n, n)) < cfg.soil_slide)
        soil_h = soil_h - go.astype(soil_h.dtype)
        for k, (dx, dy) in enumerate(self.DIRS2):
            soil_h = soil_h + self._shift2(
                xp.where(go & (bdir == k), 1, 0).astype(soil_h.dtype), -dx, -dy, xp)
        new_surface = stone_h + soil_h

        # --- 4. подложка поехала: поднимаем/роняем то, что на ней стояло ------
        rise = new_surface - surface
        for k in (1, 2):
            lift = plants & (zz == (new_surface - k)[:, :, None]) & (rise >= k)[:, :, None]
            if not bool(lift.any()):
                continue
            tgt = xp.zeros_like(alive); tgt[:, :, k:] = lift[:, :, :-k]
            tgt = tgt & ~(species > 0)
            src = xp.zeros_like(alive); src[:, :, :-k] = tgt[:, :, k:]
            species = xp.where(tgt, xp.roll(species, k, axis=2), species)
            energy = xp.where(tgt, xp.roll(energy, k, axis=2), energy)
            age = xp.where(tgt, xp.roll(age, k, axis=2), age)
            species = xp.where(src, xp.uint8(0), species)
            energy = xp.where(src, 0.0, energy)
        surface = new_surface
        alive = species > 0
        is_anim = xp.zeros(species.shape, dtype=bool)
        for s in anim_ids:
            is_anim = is_anim | (species == s)
        plants = alive & ~is_anim
        idx = xp.clip(species.astype(xp.int32) - 1, 0, N_SPECIES - 1)

        # Подложка под стелющимся растением движется: почва съезжает вниз по
        # склону, а мох точит камень прямо под собой. Два следствия:
        #   * растение, оставшееся ВИСЕТЬ над новой поверхностью, опускается на
        #     неё (а не остаётся в воздухе);
        #   * мох, под которым появилась почва, гибнет: голый камень — его
        #     единственная подложка, на почве его вытесняет трава.
        # Траву и деревья второе не касается: почва под ними никуда не девается,
        # а массово убивать их за сползший грунт значит выкосить весь ярус.
        gtrunk = xp.asarray(np.asarray(cfg.genomes)[:, IDX["trunk"]] > 0)
        gsoil = xp.asarray(np.asarray(cfg.genomes)[:, IDX["substrate"]] >= 0.5)
        flat = plants & ~gtrunk[idx]
        surf3 = surface[:, :, None]
        hang = flat & (zz > surf3)
        if bool(hang.any()):
            drop = zz == surf3
            free = drop & ~(species > 0)
            for k in (1, 2):
                src = xp.zeros_like(alive); src[:, :, k:] = free[:, :, :-k] & hang[:, :, k:]
                if not bool(src.any()):
                    continue
                tgt = xp.zeros_like(alive); tgt[:, :, :-k] = src[:, :, k:]
                species = xp.where(tgt, xp.roll(species, -k, axis=2), species)
                energy = xp.where(tgt, xp.roll(energy, -k, axis=2), energy)
                age = xp.where(tgt, xp.roll(age, -k, axis=2), age)
                species = xp.where(src, xp.uint8(0), species)
                energy = xp.where(src, 0.0, energy)
                free = free & ~tgt
            alive = species > 0
            idx = xp.clip(species.astype(xp.int32) - 1, 0, N_SPECIES - 1)
            plants = alive & ~is_anim
            flat = plants & ~gtrunk[idx]
        # всё, что после падения всё ещё висит, — гибнет.
        # Колода на столбце делает его «голым»: почва под ней накрыта, и для
        # мха это законная подложка — он её и перерабатывает.
        bare_col = (soil_h == 0)[:, :, None]
        wrong = flat & ((zz != surf3) | (~gsoil[idx] & ~bare_col))
        species = xp.where(wrong, xp.uint8(0), species)
        energy = xp.where(wrong, 0.0, energy)
        alive = species > 0
        plants = alive & ~is_anim
        idx = xp.clip(species.astype(xp.int32) - 1, 0, N_SPECIES - 1)

        # Дерево — ОДИН организм, а не стопка независимых клеток. Помечаем
        # каждую его клетку столбцом-корнем: крона держится на стволе, и всё,
        # что потеряло связь с землёй, обваливается тем же шагом. Эта же метка
        # даёт дереву общий кошелёк (см. ниже) — крона зарабатывает свет на всё
        # дерево, а объеденная клетка бьёт по всему дереву, а не по себе одной.
        trunky = plants & gtrunk[idx]
        tree_id = xp.zeros(species.shape, dtype=xp.int32)
        if bool(trunky.any()):
            cols = (xp.arange(n)[:, None] * n + xp.arange(n)[None, :] + 1).astype(xp.int32)
            tree_id = xp.where(trunky & (zz == surface[:, :, None]), cols[:, :, None], 0)
            reach = (int(np.asarray(cfg.genomes)[:, IDX["trunk"]].max())
                     + int(round(float(np.asarray(cfg.genomes)[:, IDX["branch"]].max()) * 3)) + 2)
            for _ in range(reach):
                grown = tree_id.copy()
                grown[:, :, 1:] = xp.maximum(grown[:, :, 1:], tree_id[:, :, :-1])   # вверх
                for dx, dy in self.DIRS2:                                           # вбок
                    grown = xp.maximum(grown, self._shift2(tree_id, dx, dy, xp))
                grown = xp.where(trunky, grown, 0)
                if bool((grown == tree_id).all()):
                    break
                tree_id = grown
            fell = trunky & (tree_id == 0)
            species = xp.where(fell, xp.uint8(0), species)
            energy = xp.where(fell, 0.0, energy)
            trunky = trunky & ~fell
            alive = species > 0
            plants = alive & ~is_anim
            idx = xp.clip(species.astype(xp.int32) - 1, 0, N_SPECIES - 1)
        state["tree_id"] = tree_id
        had_root = (trunky & (zz == surface[:, :, None])).any(axis=2)
        # высота дерева пригодится, когда оно упадёт: колода ложится длиной со
        # ствол, а после гибели считать уже нечего
        had_height = xp.where(trunky, zz - surface[:, :, None] + 1, 0).max(axis=2)

        # похороненное подложкой гибнет
        buried = alive & (zz < surface[:, :, None])
        species = xp.where(buried, xp.uint8(0), species)
        energy = xp.where(buried, 0.0, energy)
        alive = species > 0
        plants = alive & ~is_anim
        idx = xp.clip(species.astype(xp.int32) - 1, 0, N_SPECIES - 1)

        # --- 5. свет и ресурс -------------------------------------------------
        # тень — обычное поле света: крона дерева (absorb 0.8) гасит его, и под
        # деревом трава (которая живёт светом) недобирает ресурс. Отдельного
        # «затенения» писать не пришлось, оно и есть плотность кроны над точкой
        absorb = xp.where(plants, G[idx, IDX["absorb"]], 0.0).astype(xp.float32)
        L = light_field(plants, absorb, xp)
        wet3 = xp.minimum(wet, 1.0)[:, :, None]
        lightg = xp.where(plants, G[idx, IDX["light"]], 0.0).astype(xp.float32)
        R = cfg.light_gain * lightg * L * (1.0 + cfg.water_gain * wet3)
        # чужая подложка не кормит: трава, из-под которой ушла почва, не растёт,
        # а доедает запас и вымирает сама — мягче, чем убивать её на месте
        gsoil5 = xp.asarray(np.asarray(cfg.genomes)[:, IDX["substrate"]] >= 0.5)
        gtrunk5 = xp.asarray(np.asarray(cfg.genomes)[:, IDX["trunk"]] > 0)
        wrong_sub = plants & ~gtrunk5[idx] & (gsoil5[idx] != (soil_h > 0)[:, :, None])
        R = xp.where(wrong_sub, 0.0, R)

        # --- 6. экономика: копим, стареем, голодаем ---------------------------
        metab = xp.where(plants, G[idx, IDX["metabolism"]], 0.0).astype(xp.float32)
        mass = xp.where(alive, xp.maximum(G[idx, IDX["mass"]], 0.1), 0.0).astype(xp.float32)
        cost = xp.where(plants, G[idx, IDX["repro"]] * mass * cfg.growth_cost, 0.0)
        flatp = plants & ~gtrunk[idx]
        energy = xp.where(flatp, xp.minimum(energy + R - metab, cost * cfg.energy_cap), energy)

        # Кошелёк дерева лежит в его корневой клетке: крона зарабатывает свет на
        # всё дерево и на него же тратит. Поэтому объеденная крона бьёт по всему
        # организму, а не по одной клетке, и дерево, у которого не осталось на
        # что жить, гибнет целиком — вместе со стволом.
        if bool(trunky.any()):
            flat_id = tree_id.ravel()
            nbin = int(n * n + 1)
            gain = xp.bincount(flat_id, weights=(R - metab).ravel(), minlength=nbin)
            root = trunky & (zz == surface[:, :, None])
            rid = xp.where(root, tree_id, 0).ravel()
            purse = xp.bincount(rid, weights=energy.ravel(), minlength=nbin)
            cap = xp.bincount(flat_id, weights=(cost * cfg.energy_cap).ravel(), minlength=nbin)
            purse = xp.minimum(purse + gain, xp.maximum(cap, 1.0))
            purse[0] = 0.0
            state["tree_purse"] = purse
            energy = xp.where(root, purse[tree_id], xp.where(trunky, 0.0, energy))

        age = xp.where(alive, age + 1, 0)
        life = xp.where(alive, G[idx, IDX["lifespan"]], 0.0).astype(xp.float32)
        too_old = alive & (life > 0) & (age.astype(xp.float32) > life)
        starved = flatp & (energy <= 0)
        if bool(trunky.any()):
            # дерево без кошелька гибнет целиком, а не осыпается по клетке
            broke = trunky & (state["tree_purse"][tree_id] <= 0)
            starved = starved | broke
        shock = alive & (rng.random(species.shape) < cfg.p_shock)
        dead = too_old | starved | shock
        species = xp.where(dead, xp.uint8(0), species)
        energy = xp.where(dead, 0.0, energy)
        age = xp.where(dead, 0, age)
        alive = species > 0
        plants = alive & ~is_anim

        # --- 6б. колода: погибшее дерево оставляет после себя камень ----------
        # Ствол не исчезает бесследно: он лежит на земле, накрывает почву, и
        # растениям там места нет. За переработку древесины в этом мире отвечает
        # МОХ — единственный, кто живёт на голом камне: он медленно точит колоду
        # обратно в почву. Круг замыкается: лес → колода → камень → мох → почва.
        if cfg.log_stone:
            now_root = (plants & gtrunk[idx] & (zz == surface[:, :, None])).any(axis=2)
            fallen_root = had_root & ~now_root       # где корень был, а теперь нет
            if bool(fallen_root.any()):
                stone_h, soil_h = self._fell_trunks(
                    state, cfg, xp, fallen_root, had_height, stone_h, soil_h)

        # --- 7. рождение растений --------------------------------------------
        species, energy, age = self._grow(
            state, cfg, xp, correlate, rng, species, energy, age,
            plants, surface, soil_h, L, zz)

        state.update(species=species, energy=energy, age=age,
                     stone_h=stone_h, soil_h=soil_h, wet=wet, gen=gen)
        self._sync_volumes(state, cfg, xp)

        # --- 8. животные: ходят по поверхности --------------------------------
        if anim_ids:
            self._animals(state, cfg, xp, rng, surface)

        sp = state["species"]
        from ..backend import to_cpu
        counts = np.bincount(to_cpu(sp).ravel(), minlength=N_SPECIES + 1)
        pops = [int(c) for c in counts[1:N_SPECIES + 1]]
        state["pops"] = pops
        # «ниша» — сколько мест вид вообще может занять: по ним и меряем тесноту
        soil_cols = int(to_cpu(soil_h > 0).sum())
        bare_cols = int(cfg.n * cfg.n - soil_cols)
        walk_cols = int(cfg.n * cfg.n)
        state["niche"] = {1: bare_cols, 2: soil_cols, 3: max(soil_cols // 9, 1),
                          4: walk_cols, 5: walk_cols}
        return pops

    # ------------------------------------------------------------- рост
    def _grow(self, state, cfg, xp, correlate, rng, species, energy, age,
              plants, surface, soil_h, L, zz):
        """Рождение растений. Подложка решает, КТО где может появиться:

          мох    — только на голом камне (soil_h == 0) и только на поверхности;
          трава  — только на почве (soil_h > 0) и только на поверхности;
          дерево — на почве: сначала ствол строго вверх на `trunk` клеток,
                   выше ствола — ветвление вбок.

        На траве не растёт ничто: занятая клетка поверхности — не место для
        рождения, а выше травы подложки нет.
        """
        G = state["genomes"]
        n = cfg.n
        surf3 = surface[:, :, None]
        empty = ~(species > 0) & (zz >= surf3)
        on_soil = (soil_h > 0)[:, :, None]
        K1 = state.get("k1")
        if K1 is None:
            K1 = xp.ones((3, 3, 3), dtype=xp.float32); K1[1, 1, 1] = 0
            state["k1"] = K1

        # столбцы, где новый ствол вставать не должен: рядом уже есть чужой
        gtr = np.asarray(cfg.genomes)[:, IDX["trunk"]] > 0
        trunk_block = None
        r = int(cfg.trunk_spacing)
        if r > 0 and gtr.any():
            trunks = xp.zeros((n, n), dtype=xp.float32)
            for si in np.nonzero(gtr)[0]:
                trunks = trunks + ((species == int(si) + 1) & (zz == surf3)
                                   ).any(axis=2).astype(xp.float32)
            disc = state.get("_disc")
            if disc is None or disc.shape[0] != 2 * r + 1:
                ax = xp.arange(-r, r + 1)
                disc = ((ax[:, None] ** 2 + ax[None, :] ** 2) <= r * r).astype(xp.float32)
                state["_disc"] = disc
            trunk_block = (correlate(trunks, disc, mode="constant", cval=0.0) > 0)[:, :, None]

        for s in [i + 1 for i in range(N_SPECIES)]:
            g = cfg.genomes[s - 1]
            if float(g[IDX["speed"]]) > 0:
                continue
            mine_mask = (species == s)
            if not bool(mine_mask.any()):
                continue
            mine = correlate(mine_mask.astype(xp.float32), K1, mode="constant", cval=0.0)
            emine = correlate(xp.where(mine_mask, energy, 0.0).astype(xp.float32), K1,
                              mode="constant", cval=0.0)
            cost_s = float(g[IDX["repro"]]) * max(float(g[IDX["mass"]]), 0.1) * cfg.growth_cost
            tree_id = state.get("tree_id")
            purse = state.get("tree_purse")
            as_one = (float(g[IDX["trunk"]]) > 0 and tree_id is not None
                      and purse is not None and bool(mine_mask.any()))
            if as_one:
                # У дерева общий кошелёк: новую клетку оплачивает весь организм,
                # а не средняя энергия соседей вокруг точки роста. Пустая клетка
                # своего номера ещё не имеет — берём номер дерева, к которому она
                # прирастает (максимум по соседям): именно оно и заплатит.
                owner = tree_id.copy()
                owner[:, :, :-1] = xp.maximum(owner[:, :, :-1], tree_id[:, :, 1:])
                owner[:, :, 1:] = xp.maximum(owner[:, :, 1:], tree_id[:, :, :-1])
                for dx, dy in self.DIRS2:
                    owner = xp.maximum(owner, self._shift2(tree_id, dx, dy, xp))
                rich = purse[owner] > cost_s
            else:
                rich = (emine / xp.maximum(mine, 1e-6)) > cost_s

            wants_soil = float(g[IDX["substrate"]]) >= 0.5
            trunk = int(g[IDX["trunk"]])
            at_surface = empty & (zz == surf3)
            substrate_ok = at_surface & (on_soil if wants_soil else ~on_soil)
            if int(g[IDX["trunk"]]) > 0 and as_one:
                # Ствол вплотную к чужому не встаёт, а рождение шло только
                # вплотную к своим клеткам — значит новому стволу взяться
                # неоткуда вовсе. Дерево роняет СЕМЯ: взрослое (кошелёк тянет
                # больше двух клеток) сеет в радиусе `seed_range` на свободную
                # почву и платит за всходы из своего кошелька.
                species, energy, age = self._seed_fall(
                    state, cfg, xp, correlate, rng, species, energy, age,
                    s, g, cost_s, tree_id, purse, surface, soil_h, zz)
            if int(g[IDX["trunk"]]) > 0 and trunk_block is not None:
                # Новый ствол не встаёт вплотную к чужому: лес из деревьев,
                # стоящих плечом к плечу, — это сплошная стена, а не лес. Заодно
                # кроны соседей перестают перекрываться, и каждое дерево
                # остаётся отдельным организмом со своим кошельком.
                substrate_ok = substrate_ok & ~trunk_block

            if trunk > 0:
                # ствол: клетка прямо над своей же, пока не выросли `trunk`
                below_mine = xp.zeros_like(mine_mask)
                below_mine[:, :, 1:] = mine_mask[:, :, :-1]
                height = zz - surf3
                up = empty & below_mine & (height <= trunk) & on_soil
                # крона: вбок, но только в пределах нескольких ярусов над
                # стволом — иначе дерево растёт в высоту без предела и лес
                # превращается в сплошной куб
                crown_h = max(1, int(round(float(g[IDX["branch"]]) * 3)))
                side = (empty & (height > trunk) & (height <= trunk + crown_h)
                        & (mine > 0) & (L > cfg.crown_light) & on_soil)
                ok = substrate_ok | up | side
            else:
                ok = substrate_ok
            ok = ok & (mine > 0) & rich
            if not bool(ok.any()):
                continue
            born = ok & (rng.random(species.shape) < 0.5)
            if not bool(born.any()):
                continue
            species = xp.where(born, xp.uint8(s), species)
            age = xp.where(born, xp.int32(0), age)
            if as_one:
                # платит кошелёк дерева; новая клетка сама по себе пустая
                energy = xp.where(born, 0.0, energy)
                # свежая клетка сразу получает номер своего дерева: иначе до
                # следующего шага она числится ничьей, и счётчик организмов
                # (да и всякий, кто смотрит в состояние) видит неправду
                state["tree_id"] = xp.where(born, owner, tree_id)
                tree_id = state["tree_id"]
                nbin = int(cfg.n * cfg.n + 1)
                bill = xp.bincount(xp.where(born, owner, 0).ravel(),
                                   minlength=nbin).astype(xp.float32) * xp.float32(cost_s)
                bill[0] = 0.0
                # кошелёк лежит в корневой клетке — с неё и списываем
                root_here = mine_mask & (tree_id > 0) & (energy > 0)
                energy = xp.where(root_here,
                                  xp.maximum(energy - bill[tree_id], 0.0), energy)
            else:
                energy = xp.where(born, xp.float32(cost_s * 0.5), energy)
                # за постройку платят соседи-родители
                paid = correlate(xp.where(born, xp.float32(cost_s), 0.0).astype(xp.float32),
                                 K1, mode="constant", cval=0.0) / float(K1.sum())
                energy = xp.where(mine_mask, xp.maximum(energy - paid, 0.0), energy)
            # родитель заплатил за клетку — а выросло из неё не всегда своё
            species, energy, age = self._mutate(
                state, cfg, xp, rng, species, energy, age, s, born, surf3, zz,
                crowded=trunk_block)
        return species, energy, age

    def mutation_chance(self, state, cfg, src, dst):
        """Шанс, что клетка вида `src` при делении даст клетку вида `dst`.

        Две поправки к базовому шансу, обе по просьбе «засев должен идти через
        мутацию, а не кубиками на карте»:

        * ТЕСНОТА. Чем полнее вид занял свою нишу, тем чаще он ошибается. Трава,
          выевшая всю доступную землю, чаще даёт травоядное — ровно тогда, когда
          травоядному есть что есть.
        * ПУСТАЯ НИША. Если вида `dst` в мире не осталось вовсе, шанс взлетает в
          `mutate_rescue` раз. Это и есть спасательный круг: цепь
          восстанавливается сама, изнутри живого, а не подсевом снаружи.

        Условием пустая ниша НЕ является: мир, где все пять видов на месте, всё
        равно изредка порождает чужаков.
        """
        pops = state.get("pops")
        if pops is None or cfg.mutate_rate <= 0:
            return 0.0
        p = float(cfg.mutate_rate)
        room = max(int(state.get("niche", {}).get(src, 0)), 1)
        crowd = min(pops[src - 1] / room, 1.0)
        p *= 0.15 + 0.85 * crowd
        if pops[dst - 1] <= 0:
            p *= float(cfg.mutate_rescue)
        else:
            # чем гуще целевой вид, тем реже к нему мутируют
            p *= 1.0 / (1.0 + pops[dst - 1] / max(room * 0.05, 1.0))
        return min(p, 0.5)

    def _mutate(self, state, cfg, xp, rng, species, energy, age, src, born,
                surf3, zz, crowded=None):
        """Часть новорождённых вида `src` выходит чужим видом."""
        out = born
        for dst in MUTATIONS.get(src, ()):
            p = self.mutation_chance(state, cfg, src, dst)
            if p <= 0:
                continue
            g = cfg.genomes[dst - 1]
            pick = out & (zz == surf3) & (rng.random(species.shape) < p)
            if float(g[IDX["trunk"]]) > 0 and crowded is not None:
                pick = pick & ~crowded          # ствол вплотную к чужому не встаёт
            if not bool(pick.any()):
                continue
            if float(g[IDX["speed"]]) > 0:
                e = float(g[IDX["repro"]]) * 0.7
            else:
                e = float(g[IDX["repro"]]) * max(float(g[IDX["mass"]]), 0.1) \
                    * cfg.growth_cost * 0.5
            species = xp.where(pick, xp.uint8(dst), species)
            energy = xp.where(pick, xp.float32(max(e, 1.0)), energy)
            age = xp.where(pick, xp.int32(0), age)
            out = out & ~pick
        return species, energy, age

    def _fell_trunks(self, state, cfg, xp, fallen, had_height, stone_h, soil_h):
        """Погибшее дерево ПАДАЕТ: ствол валится в случайную сторону и по всей
        своей длине подменяет почву камнем.

        Почему именно так, а не отдельным слоем «колода» (как было в v0.14.1):
        колода поверх почвы поднимала поверхность на клетку, и всё, что на этой
        полосе стояло, оказывалось то в воздухе, то похороненным — в кадре
        висели кубы. Подмена «почва → камень» высоту НЕ меняет вовсе: висеть
        нечему. А смысл тот же и даже честнее — под упавшим стволом земля
        мертва, расти там нельзя никому, кроме мха (он один живёт на камне), и
        мох же вернёт полосу в почву, когда её источит.
        """
        from ..backend import to_cpu
        n = cfg.n
        fall = to_cpu(fallen)
        hh = to_cpu(had_height)
        line = np.zeros((n, n), bool)
        rng_cpu = np.random.default_rng(int(state.get("gen", 0)) ^ 0x10ff5eed)
        for x, y in np.argwhere(fall):
            length = int(min(max(hh[x, y], 1), cfg.log_max))
            dx, dy = self.DIRS2[rng_cpu.integers(len(self.DIRS2))]
            for k in range(length):
                px, py = int(x) + dx * k, int(y) + dy * k
                if not (0 <= px < n and 0 <= py < n):
                    break
                line[px, py] = True
        mask = xp.asarray(line)
        # почва уходит в камень: поверхность на месте, подложка мертва
        stone_h = stone_h + xp.where(mask, soil_h, 0)
        soil_h = xp.where(mask, 0, soil_h)
        return stone_h, soil_h

    def _seed_fall(self, state, cfg, xp, correlate, rng, species, energy, age,
                   sid, g, cost_s, tree_id, purse, surface, soil_h, zz):
        """Взрослое дерево роняет семя в радиусе нескольких клеток.

        Всё считается по поверхности (2D): у каждого столбца одна точка роста,
        и объёмный поиск тут не нужен. Семя ложится только на свободную почву,
        не ближе `trunk_spacing` к чужому стволу, а платит за него то дерево,
        чей номер дотянулся до этой клетки, — из общего кошелька.
        """
        n = cfg.n
        rng_seed = int(cfg.seed_range)
        if rng_seed <= 0:
            return species, energy, age
        surf3 = surface[:, :, None]
        root = (species == sid) & (zz == surf3)
        root2d = xp.where(root.any(axis=2), xp.max(xp.where(root, tree_id, 0), axis=2), 0)
        # взрослое = кошелёк тянет больше двух клеток: одну потратит на семя,
        # на другую будет расти само
        adult = xp.where(root2d > 0, purse[root2d] > cost_s * cfg.seed_maturity, False)
        src = xp.where(adult, root2d, 0).astype(xp.int32)
        if not bool((src > 0).any()):
            return species, energy, age
        for _ in range(rng_seed):                     # разлёт: максимум по кругу
            grown = src
            for dx, dy in self.DIRS2:
                grown = xp.maximum(grown, self._shift2(src, dx, dy, xp))
            src = grown
        surf_cell = (zz == surf3)
        free = (~(species > 0) & surf_cell).any(axis=2)
        spacing = int(cfg.trunk_spacing)
        blocked = xp.zeros((n, n), dtype=bool)
        if spacing > 0:
            trunks2d = ((species == sid) & surf_cell).any(axis=2).astype(xp.float32)
            ax = xp.arange(-spacing, spacing + 1)
            disc = ((ax[:, None] ** 2 + ax[None, :] ** 2) <= spacing * spacing).astype(xp.float32)
            blocked = correlate(trunks2d, disc, mode="constant", cval=0.0) > 0
        spot = (src > 0) & free & (soil_h > 0) & ~blocked
        spot = spot & (rng.random((n, n)) < cfg.seed_fall)
        if not bool(spot.any()):
            return species, energy, age
        nbin = int(n * n + 1)
        bill = xp.bincount(xp.where(spot, src, 0).ravel(),
                           minlength=nbin).astype(xp.float32) * xp.float32(cost_s)
        bill[0] = 0.0
        born = spot[:, :, None] & surf_cell & ~(species > 0)
        species = xp.where(born, xp.uint8(sid), species)
        age = xp.where(born, xp.int32(0), age)
        # всход получает половину цены клетки, вторая половина — расход родителя
        energy = xp.where(born, xp.float32(cost_s * 0.5), energy)
        # всход — уже отдельный организм: номер по его собственному столбцу
        cols = (xp.arange(n)[:, None] * n + xp.arange(n)[None, :] + 1).astype(xp.int32)
        state["tree_id"] = xp.where(born, cols[:, :, None], tree_id)
        pay = root & (tree_id > 0)
        energy = xp.where(pay, xp.maximum(energy - bill[tree_id], 0.0), energy)
        return species, energy, age

    # ------------------------------------------------------------- животные
    def _animals(self, state, cfg, xp, rng, surface):
        """Животные живут НА ПОВЕРХНОСТИ и ходят только по камню, почве и
        траве. Ствол и крона для них — препятствие: в столбец, где на уровне
        шага стоит дерево, зайти нельзя, забраться на него тоже.

        Модель двумерная по построению: у каждого столбца есть высота шага
        walk_z, животное всегда на ней, а шаг возможен к соседнему столбцу с
        перепадом не больше одной ступеньки. Это и есть «ходят по земле», и
        заодно на порядок дешевле трёхмерного поиска пути.
        """
        from ..backend import to_cpu
        n = cfg.n
        sp = to_cpu(state["species"]).copy()
        en = to_cpu(state["energy"]).copy()
        ag = to_cpu(state["age"]).copy()
        surf = to_cpu(surface).astype(np.int64)
        G = np.asarray(cfg.genomes)
        trunk_sp = [i + 1 for i in range(N_SPECIES) if G[i][IDX["trunk"]] > 0]
        anim_sp = [i + 1 for i in range(N_SPECIES) if G[i][IDX["speed"]] > 0]

        xs = np.arange(n)[:, None].repeat(n, 1)
        ys = np.arange(n)[None, :].repeat(n, 0)
        # высота шага: поверхность, а если на ней лежит плоское растение
        # (мох/трава) — на клетку выше; по ним ходить можно
        top = sp[xs, ys, np.clip(surf, 0, n - 1)]
        flat_here = np.isin(top, [s for s in range(1, N_SPECIES + 1)
                                  if s not in trunk_sp and G[s - 1][IDX["speed"]] == 0])
        walk = np.clip(surf + flat_here.astype(np.int64), 0, n - 1)
        # столбец занят деревом на уровне шага — туда не пройти
        blocked = np.isin(sp[xs, ys, walk], trunk_sp)

        # почва под зверем могла съехать за этот шаг: роняем/поднимаем всех на
        # свой уровень шага, иначе животное остаётся висеть над обрывом
        anim3 = np.isin(sp, anim_sp)
        for x, y, z in np.argwhere(anim3):
            if not blocked[x, y] and int(walk[x, y]) == z:
                continue                                  # уже стоит как надо
            spots = [(x, y)] if not blocked[x, y] else []
            spots += [(x + dx, y + dy) for dx, dy in self.DIRS2
                      if 0 <= x + dx < n and 0 <= y + dy < n and not blocked[x + dx, y + dy]]
            for nx, ny in spots:
                nz = int(walk[nx, ny])
                if sp[nx, ny, nz] != 0:
                    continue
                sp[nx, ny, nz], en[nx, ny, nz], ag[nx, ny, nz] = sp[x, y, z], en[x, y, z], ag[x, y, z]
                sp[x, y, z], en[x, y, z], ag[x, y, z] = 0, 0.0, 0
                break
            else:
                sp[x, y, z], en[x, y, z], ag[x, y, z] = 0, 0.0, 0   # некуда встать

        for s in anim_sp:
            g = G[s - 1]
            speed, sense = int(g[IDX["speed"]]), int(g[IDX["sense"]])
            level = int(g[IDX["trophic"]])
            here = np.argwhere(sp == s)
            if not len(here):
                continue
            prey_sp = [i for i in DIET.get(s, ()) if 1 <= i <= N_SPECIES]
            prey2d = np.zeros((n, n), np.float32)
            for ps in prey_sp:
                prey2d += (sp == ps).any(axis=2).astype(np.float32)
            if sense > 0:
                from scipy.ndimage import gaussian_filter
                attract = gaussian_filter(prey2d, sigma=max(sense / 2.0, 0.5), mode="constant")
            else:
                attract = prey2d
            # Где в столбце лежит ближайшая добыча. Раньше это искали заново
            # (np.isin по всему столбцу) на каждое из пяти направлений каждого
            # зверя: 8800 вызовов isin за шаг и 90 % времени всего движка.
            # Одна карта на вид — и поиск становится чтением двух чисел.
            # шум в единицах размаха поля: и на пустом поле, и на насыщенном
            # он остаётся сопоставим с перепадом, по которому зверь и решает
            span = float(attract.max() - attract.min())
            noise_amp = cfg.move_noise * (span if span > 0 else 1.0)
            prey_mask = np.isin(sp, prey_sp) if prey_sp else np.zeros(sp.shape, bool)
            # ВЕРХНЯЯ клетка добычи, а не нижняя: травоядное объедает крону
            # сверху и спускается по мере объедания. Раньше оно ело самую
            # нижнюю — то есть основание ствола, — и целое дерево валилось с
            # одного укуса.
            prey_z = np.where(prey_mask.any(axis=2),
                              sp.shape[2] - 1 - prey_mask[:, :, ::-1].argmax(axis=2), -1)
            rng_cpu = np.random.default_rng((cfg.seed_mut ^ (s * 7919)) + int(state["gen"]))
            for (x, y, z) in here:
                if sp[x, y, z] != s:
                    continue
                for _ in range(speed):
                    best, bx, by = -1e9, x, y
                    # Порядок направлений тасуем, а шум делаем АДДИТИВНЫМ.
                    # Мультипликативный шум (attract·(1±0.15)) исчезает ровно
                    # там, где поле добычи пустое, — а это единственное место,
                    # где случайный ход и нужен: без него все ничьи разрешались
                    # первым направлением списка, и каждый шестой хищник каждый
                    # шаг маршировал строго в +x. В кадре это выглядело как
                    # река из существ, текущая в одну сторону.
                    order = self.MOVES[rng_cpu.integers(len(self.MOVES))]
                    for dx, dy in order:
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < n and 0 <= ny < n):
                            continue
                        if (dx or dy):
                            if blocked[nx, ny]:
                                continue                    # дерево на пути
                            if abs(int(walk[nx, ny]) - int(walk[x, y])) > 1:
                                continue                    # обрыв: не вскарабкаться
                            if sp[nx, ny, walk[nx, ny]] in anim_sp:
                                continue                    # занято другим зверем
                        v = attract[nx, ny] + noise_amp * (rng_cpu.random() * 2 - 1)
                        if v > best:
                            best, bx, by = v, nx, ny
                    if (bx, by) == (x, y):
                        break
                    nz = int(walk[bx, by])
                    sp[bx, by, nz], en[bx, by, nz], ag[bx, by, nz] = s, en[x, y, z], ag[x, y, z]
                    sp[x, y, z], en[x, y, z], ag[x, y, z] = 0, 0.0, 0
                    x, y, z = bx, by, nz
                # охота: съесть добычу в своём столбце или рядом
                hunt = float(g[IDX["hunt"]])
                if hunt > 0:
                    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < n and 0 <= ny < n):
                            continue
                        pz = int(prey_z[nx, ny])
                        if pz < 0:
                            continue
                        victim = int(sp[nx, ny, pz])
                        if victim == 0:            # карта отстала: добычу уже съели
                            prey_mask[nx, ny, pz] = False
                            col = prey_mask[nx, ny]
                            prey_z[nx, ny] = (len(col) - 1 - col[::-1].argmax()
                                              if col.any() else -1)
                            continue
                        armor = float(G[victim - 1][IDX["armor"]])
                        if rng_cpu.random() > hunt * (1 - armor):
                            continue
                        en[x, y, z] += cfg.eat_efficiency * float(G[victim - 1][IDX["mass"]])
                        sp[nx, ny, pz], en[nx, ny, pz], ag[nx, ny, pz] = 0, 0.0, 0
                        prey_mask[nx, ny, pz] = False
                        col = prey_mask[nx, ny]
                        prey_z[nx, ny] = (len(col) - 1 - col[::-1].argmax()
                                          if col.any() else -1)
                        break
                # обмен, смерть, деление
                en[x, y, z] -= float(g[IDX["metabolism"]])
                if en[x, y, z] <= 0:
                    sp[x, y, z], en[x, y, z], ag[x, y, z] = 0, 0.0, 0
                    continue
                if en[x, y, z] > float(g[IDX["repro"]]):
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < n and 0 <= ny < n) or blocked[nx, ny]:
                            continue
                        nz = int(walk[nx, ny])
                        if sp[nx, ny, nz] != 0:
                            continue
                        en[x, y, z] *= 0.5
                        # потомок не всегда выходит своим видом: так в мире
                        # заводится хищник, если хищников не осталось
                        child = s
                        for dst in MUTATIONS.get(s, ()):
                            if rng_cpu.random() < self.mutation_chance(state, cfg, s, dst):
                                child = dst
                                break
                        e0 = en[x, y, z]
                        if child != s:
                            e0 = max(float(G[child - 1][IDX["repro"]]) * 0.7, 1.0)
                        sp[nx, ny, nz], en[nx, ny, nz], ag[nx, ny, nz] = child, e0, 0
                        break
        state["species"] = xp.asarray(sp)
        state["energy"] = xp.asarray(en)
        state["age"] = xp.asarray(ag)

    # ------------------------------------------------------------- засев
    def seed(self, state, cfg, xp, rng, count=None, gen=0):
        from ..backend import to_cpu
        n = cfg.n
        sp = to_cpu(state["species"]).copy()
        en = to_cpu(state["energy"]).copy()
        surf = to_cpu(state["stone_h"]) + to_cpu(state["soil_h"])
        soil = to_cpu(state["soil_h"])
        rng_cpu = np.random.default_rng((cfg.seed_mut ^ 0x5bf03635) + gen)
        cnt = int(count if count is not None else cfg.reseed_count)
        G = np.asarray(cfg.genomes)

        # Подсаживаем в первую очередь ВЫМЕРШИХ: без этого спасательный засев
        # льёт траву в мир, где вымер хищник, и цепь так и не восстанавливается.
        gone = [i + 1 for i in range(N_SPECIES) if not (sp == i + 1).any()]
        xs = rng_cpu.integers(0, n, cnt); ys = rng_cpu.integers(0, n, cnt)
        put = 0
        for k, (x, y) in enumerate(zip(xs, ys)):
            z = int(np.clip(surf[x, y], 0, n - 1))
            if sp[x, y, z]:
                z += 1                        # на мох/траву можно встать зверю
                if z >= n or sp[x, y, z]:
                    continue
            if gone:
                s = gone[k % len(gone)]
            else:
                s = 2 if soil[x, y] > 0 else 1     # на почве трава, на камне мох
            g = G[s - 1]
            if float(g[IDX["speed"]]) == 0:
                # растение уважает подложку: мох на камень, остальные на почву
                wants_soil = float(g[IDX["substrate"]]) >= 0.5
                if wants_soil != (soil[x, y] > 0) or z != int(surf[x, y]):
                    continue
                e = float(g[IDX["repro"]]) * max(float(g[IDX["mass"]]), 0.1) * cfg.growth_cost * 0.6
            else:
                e = float(g[IDX["repro"]]) * 0.7
            sp[x, y, z] = s
            en[x, y, z] = max(e, 1.0)
            put += 1
        state["species"] = xp.asarray(sp)
        state["energy"] = xp.asarray(en)
        state["last_reseed"] = gen
        return put

    # ------------------------------------------------------------- интерфейс
    def species_names(self, cfg):
        return list(NAMES)

    def species_colors(self, cfg):
        return list(COLORS)

    def species_organisms(self, cfg, state=None):
        """Клетка = организм для всех, кроме деревьев: у дерева ствол и крона —
        один организм, и одно дерево не должно весить как двадцать травинок."""
        pops = (state or {}).get("pops")
        if pops is None:
            return None
        out = list(pops)
        tid = (state or {}).get("tree_id")
        if tid is not None:
            from ..backend import to_cpu
            t = to_cpu(tid)
            ntree = int(np.unique(t[t > 0]).size)
            for i, g in enumerate(np.asarray(cfg.genomes)):
                if g[IDX["trunk"]] > 0:
                    out[i] = ntree
                    break
        return [int(v) for v in out]

    def species_mass(self, cfg):
        g = np.asarray(cfg.genomes)
        return [float(v) if v > 0 else 1.0 for v in g[:, IDX["mass"]]]

    def world_params(self):
        return self.WORLD_PARAMS

    def starters_json(self, cfg):
        out = []
        habitats = ("камень", "почва", "почва", "поверхность", "поверхность")
        for i, nm in enumerate(NAMES):
            out.append({"i": i + 1, "name": nm, "habitat": habitats[i], "on": True})
        return out

    def to_json(self, cfg, state=None):
        from . import seeding_json
        return {
            "engine": self.name,
            "fields": list(GENES), "labels": LABELS,
            "ranges": {k: list(v) for k, v in RANGES.items()},
            "names": list(NAMES), "colors": list(COLORS),
            "genomes": np.asarray(cfg.genomes).tolist(),
            "world": {k: getattr(cfg, k) for k in self.WORLD_PARAMS},
            "world_labels": self.WORLD_LABELS, "world_ranges": self.WORLD_RANGES,
            "fixed_genes": list(self.fixed_genes(cfg)),
            "starters": self.starters_json(cfg),
            "reseed": seeding_json(cfg),
        }

    def apply_genomes(self, cfg, state, genomes, xp):
        g = np.asarray(genomes, dtype=np.float32)
        if g.shape != (N_SPECIES, len(GENES)):
            raise ValueError(f"геномы: ожидается {N_SPECIES} × {len(GENES)}")
        cfg.genomes = g
        state["genomes"] = xp.asarray(g)

    # Роль, подложка и форма роста случайными не бывают: мох со стволом 9 — это
    # не «другой мох», это сломанный мир. Список едет и клиенту, чтобы кнопка
    # «случайные гены» в панели не крутила то, что движок держит.
    FIXED_GENES = ("trophic", "speed", "substrate", "trunk")

    def randomize(self, cfg, rng):
        g = np.asarray(cfg.genomes, dtype=np.float32).copy()
        for s in range(N_SPECIES):
            for name, i in IDX.items():
                if name in self.FIXED_GENES:
                    continue
                lo, hi, _ = RANGES[name]
                g[s, i] = rng.uniform(lo, hi)
        return g

    GENE_DOCS = {
        "light": "Насколько хорошо вид усваивает свет.",
        "absorb": "Сколько света клетка съедает по дороге вниз — то есть какую "
                  "тень она даёт. У дерева высокий: под кроной трава недобирает "
                  "ресурс и редеет.",
        "substrate": "На чём вид может расти: 0 — голый камень (мох), 1 — почва "
                     "(трава, дерево). Клетка поверхности, уже кем-то занятая, "
                     "местом для рождения не является — на траве не растёт ничто.",
        "trunk": "Сколько клеток вид гонит строго ВВЕРХ, прежде чем начать "
                 "ветвиться. 0 — стелется по земле (трава, мох), 5 — дерево.",
        "branch": "Вес бокового роста выше ствола — из него получается крона.",
        "erode": "Как быстро вид точит камень под собой в почву. Работает только "
                 "на голом камне и во влажных местах. У мха это единственная "
                 "профессия: он готовит почву тем, кто придёт после него.",
        "metabolism": "Трата энергии за поколение.",
        "repro": "Порог накопления для деления; новая клетка стоит "
                 "repro × mass × growth_cost.",
        "lifespan": "Предельный возраст, 0 — не стареет.",
        "mass": "Масса клетки: цена постройки, пищевая ценность и вклад в биомассу.",
        "hunt": "Сила атаки. Работает только у подвижных.",
        "trophic": "0 растение, 1 травоядное, 2 хищник.",
        "speed": "Шагов по поверхности за поколение.",
        "sense": "Радиус чутья.",
        "armor": "Шанс отбиться от нападения.",
    }

    def gene_docs(self):
        return dict(self.GENE_DOCS)


RULES = SlopeRules()
