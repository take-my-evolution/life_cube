"""Движок «железо» — клеточный автомат, спроектированный под ПЛИС.

Всё, что здесь есть, обязано ложиться на конвейер «один воксель за такт»:

  * только целые числа: энергия 0..15, возраст 0..255, влага 0..15, свет 0..15;
    ни одного float ни в состоянии, ни в правилах;
  * только локальные правила: клетка видит себя, 26 соседей, свой столбец
    (высоты камня и почвы, влага, свет сверху) и пять глобальных счётчиков
    населения за прошлое поколение;
  * случайность — хеш от (x, y, z, поколение, сид), а не общий генератор:
    тот же сид даёт тот же мир бит-в-бит на CPU, на GPU и в железе;
  * никаких «толчков»: всё, что двигается (почва, звери, семена), двигается
    правилом «пустая клетка сама выбирает, кого впустить» — так конфликтов
    нет по построению, и порядок обхода не важен.

Два варианта в одном движке (параметр мира `variant`):

  0 — «жизнь 3D»: простейший B/S-автомат на кубе, живые клетки наследуют
      цвет большинства соседей (иммиграция), рельеф непроходим; редкие
      «споры» из хеша не дают миру замереть. Сотни событий за поколение.
  1 — «экология»: мох на голом камне, трава и дерево на почве, травоядное
      и хищник на поверхности. Свет сверху (регистр столбца), вода снизу,
      мох точит камень, почва съезжает, дерево тянет ствол и крону, звери
      ходят двумерным автоматом с правилом тяги.

Описание: docs/engines/iron.md
"""

from dataclasses import dataclass, field

import numpy as np

from . import Rules, seeding_json

U32 = np.uint32

# ------------------------------------------------------------------ хеш
# Целочисленный хеш координат. В железе это ~десять операций на такт;
# в numpy/cupy — те же операции над массивами uint32 с естественным
# переполнением. Константы — из xxhash/murmur, ничего магического.
H1, H2, H3, H4 = U32(0x9E3779B1), U32(0x85EBCA77), U32(0xC2B2AE3D), U32(0x27D4EB2F)
M1, M2 = U32(0x2C1B3C6D), U32(0x297A2D39)


def hash32(xp, x, y, z, gen, salt):
    """Хеш от (x, y, z, поколение, соль) -> uint32 (массив по broadcast)."""
    x = xp.asarray(x, dtype=U32); y = xp.asarray(y, dtype=U32); z = xp.asarray(z, dtype=U32)
    h = (x * H1) ^ (y * H2) ^ (z * H3) ^ U32((int(gen) * 0x27D4EB2F) & 0xFFFFFFFF) ^ U32(int(salt) & 0xFFFFFFFF)
    h = h ^ (h >> U32(15)); h = h * M1
    h = h ^ (h >> U32(12)); h = h * M2
    h = h ^ (h >> U32(15))
    return h


def hash32_py(x, y, z, gen, salt):
    """Эталон на чистом Python — тот же хеш поразрядно (для тестов и HDL)."""
    m = 0xFFFFFFFF
    h = ((x * 0x9E3779B1) & m) ^ ((y * 0x85EBCA77) & m) ^ ((z * 0xC2B2AE3D) & m) \
        ^ ((gen * 0x27D4EB2F) & m) ^ (salt & m)
    h ^= h >> 15; h = (h * 0x2C1B3C6D) & m
    h ^= h >> 12; h = (h * 0x297A2D39) & m
    h ^= h >> 15
    return h


def thr(p):
    """Вероятность 0..1 -> порог для сравнения с uint32-хешем."""
    return U32(int(min(max(p, 0.0), 1.0) * 4294967295.0))


# соли: у каждого решения свой независимый поток случайности
S_SPORE, S_SHOCK, S_ERODE, S_RAIN, S_DRY, S_TIE, S_MOVE, S_EAT, S_REPRO, S_MUT, S_LIFE = (
    0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000A, 0x000B)

# ------------------------------------------------------------------ виды
MOSS, GRASS, TREE, HERB, PRED = 1, 2, 3, 4, 5
NAMES = ("мох", "трава", "дерево", "травоядное", "хищник")
COLORS = ("#8fbf6a", "#4caf50", "#8d6e3f", "#e0a84a", "#d84a4a")
LIFE_NAMES = ("красные", "синие", "жёлтые", "зелёные")
LIFE_COLORS = ("#e05050", "#4f7fe0", "#e0c84f", "#5fbf5f")

# Геном — ЦЕЛЫЕ гены. Именно в таком виде таблица ляжет в регистры ПЛИС.
GENES = ("light", "water", "absorb", "birth", "cost", "metab", "lifespan",
         "value", "hunt", "armor", "repro", "speed")
LABELS = {"light": "свет (доля 0..16)", "water": "вода (доля 0..16)",
          "absorb": "тень (гасит света)", "birth": "порог рождения (взвеш. соседи)",
          "cost": "цена клетки", "metab": "обмен за поколение",
          "lifespan": "предельный возраст (0 — вечно)", "value": "пищевая ценность",
          "hunt": "сила атаки (0..16)", "armor": "броня (0..16)",
          "repro": "энергия деления", "speed": "подвижность (0 — растение)"}
RANGES = {"light": (0, 16, 1), "water": (0, 16, 1), "absorb": (0, 8, 1),
          "birth": (1, 12, 1), "cost": (1, 8, 1), "metab": (0, 4, 1),
          "lifespan": (0, 255, 1), "value": (0, 15, 1), "hunt": (0, 16, 1),
          "armor": (0, 16, 1), "repro": (2, 15, 1), "speed": (0, 1, 1)}
IDX = {g: i for i, g in enumerate(GENES)}

#                     light water absorb birth cost metab life value hunt armor repro speed
GENOMES = np.array([[10,    6,    1,     2,    1,   1,    80,   2,    0,   0,    8,    0],   # мох
                    [12,    4,    2,     2,    1,   1,    40,   3,    0,   0,    8,    0],   # трава
                    [14,    2,    4,     4,    2,   1,    0,    4,    0,   0,    8,    0],   # дерево
                    [0,     0,    0,     0,    2,   1,    120,  6,    8,   6,    9,    1],   # травоядное
                    [0,     0,    0,     0,    3,   1,    160,  0,    10,  0,    11,   1]],  # хищник
                   dtype=np.int32)


@dataclass
class IronConfig:
    n: int = 32
    gens: int = 400
    seed_world: int = 20260903
    seed_mut: int = 20260903
    variant: int = 1                # 0 — жизнь 3D, 1 — экология, 2 — волны (циклический автомат)
    cyc_threshold: int = 6          # волны: сколько соседей следующего цвета нужно, чтобы перекраситься
    cyc_states: int = 4             # волны: сколько цветов по кругу

    # рельеф (карта высот камня; целые)
    floor: int = 2
    stone_fraction: float = 0.4
    hill_radius: float = 0.8
    hill_roughness: float = 0.3
    soil_start: float = 1.0

    # --- вариант 0: жизнь 3D ---
    birth_lo: int = 6
    birth_hi: int = 7
    survive_lo: int = 4
    survive_hi: int = 6
    spore: float = 0.0004           # шанс споры в пустой клетке над поверхностью
    life_density: float = 0.25      # стартовая плотность засева над рельефом

    # --- вариант 1: экология ---
    rain_rate: float = 0.02         # шанс дождя в столбце за поколение
    rain_amount: int = 6            # сколько влаги приносит дождь (0..15)
    dry_rate: float = 0.15          # шанс потерять единицу влаги
    erode_rate: float = 0.005       # мох точит камень (при влаге 15 — вчетверо чаще)
    slide_drop: int = 2             # перепад, с которого почва едет
    slide_rate: float = 0.3
    trunk: int = 4                  # клеток ствола до кроны
    branch: int = 3                 # ярусов кроны
    spacing: int = 3                # просвет между стволами
    crown_light: int = 6            # ниже этого света крона не растёт
    tree_seed: float = 0.003        # шанс, что трава в тесноте даст дерево
    spore_rate: float = 0.003       # спора мха на голом камне
    p_shock: float = 0.0005         # выветривание
    mutate: float = 0.004           # шанс мутации при делении/рождении
    rescue: int = 64                # во сколько раз чаще, если ниша пуста
    seed_density: float = 0.08
    seed_animals: float = 0.02

    start_species: tuple = ()
    reseed: bool = False
    reseed_on_extinction: bool = True
    reseed_species: bool = False
    reseed_every: int = 200
    reseed_count: int = 60

    genomes: np.ndarray = field(default_factory=lambda: GENOMES.copy())

    @property
    def n_species(self):
        if self.variant == 2:
            return int(self.cyc_states)
        return 4 if self.variant == 0 else 5


# ------------------------------------------------------------------ рельеф
def relief(cfg, rng):
    """Гора из целых высот. Генерируется один раз при создании мира — в железе
    это загружаемая карта, а не правило, поэтому float здесь допустим."""
    n = cfg.n
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2 * cfg.hill_radius)
    bump = np.clip(1.0 - r ** 2, 0, 1) ** 1.1
    H = max(3.0, cfg.stone_fraction * n) - cfg.floor
    rough = np.sin(xx / (n / 8.0) + 1.3) * np.cos(yy / (n / 6.0)) + 0.5 * np.sin((xx + yy) / (n / 12.0))
    h = cfg.floor + H * bump * (1 + cfg.hill_roughness * rough * bump) + rng.normal(0, 0.4, (n, n)) * bump
    return np.clip(np.round(h), cfg.floor, n - 6).astype(np.int32)


# ------------------------------------------------------------------ утилиты
def shift2(xp, a, dx, dy, fill=0):
    out = xp.full_like(a, fill)
    n0, n1 = a.shape[0], a.shape[1]
    out[max(dx, 0):n0 + min(dx, 0), max(dy, 0):n1 + min(dy, 0)] = \
        a[max(-dx, 0):n0 + min(-dx, 0), max(-dy, 0):n1 + min(-dy, 0)]
    return out


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))     # индекс направления 0..3


def neighbours2(xp, a, fill=0):
    """Значение соседа в направлении d: nb[d][c] = a[c + DIRS[d]] (за краем — fill)."""
    return [shift2(xp, a, -dx, -dy, fill) for dx, dy in DIRS]


def downhill(xp, surf):
    """Куда стекает столбец: (индекс направления самого низкого соседа, его высота)."""
    nbs = xp.stack(neighbours2(xp, surf, fill=10 ** 6))
    return nbs.argmin(axis=0), nbs.min(axis=0)


def kernel3(face=2, edge=1, corner=1, below=2, level=1, above=0):
    """Анизотропное ядро 3×3×3 с целыми весами: снизу опора, сверху не считается."""
    k = np.zeros((3, 3, 3), np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                m = sum(map(abs, (dx, dy, dz)))
                w = {1: face, 2: edge, 3: corner}[m]
                k[dx + 1, dy + 1, dz + 1] = w * {-1: below, 0: level, 1: above}[dz]
    return k


K_ANISO = kernel3()
K_ALL = np.ones((3, 3, 3), np.int32); K_ALL[1, 1, 1] = 0


class IronRules(Rules):
    name = "iron"
    title = "Железо: целочисленный автомат под ПЛИС"
    summary = ("Всё в целых числах и только по соседям: случайность из хеша "
               "координат, свет — регистр столбца, звери и почва двигаются "
               "правилом тяги. Вариант 0 — жизнь 3D с наследованием цвета, "
               "вариант 1 — мох, трава, дерево, травоядное, хищник. "
               "Тот же сид — тот же мир бит-в-бит: так же посчитает ПЛИС.")
    doc = "docs/engines/iron.md"
    Config = IronConfig
    heightmaps = True
    terrain_changes = True
    can_seed = True

    WORLD_PARAMS = ("n", "seed_world", "seed_mut", "variant", "stone_fraction", "hill_radius",
                    "hill_roughness", "soil_start", "cyc_threshold", "cyc_states",
                    "birth_lo", "birth_hi", "survive_lo", "survive_hi", "spore", "life_density",
                    "rain_rate", "rain_amount", "dry_rate", "erode_rate", "slide_drop", "slide_rate",
                    "trunk", "branch", "spacing", "crown_light", "tree_seed", "spore_rate",
                    "p_shock", "mutate", "rescue", "seed_density", "seed_animals")
    WORLD_RANGES = {"n": [16, 128, 16], "variant": [0, 2, 1], "cyc_threshold": [1, 26, 1],
                    "cyc_states": [3, 8, 1], "stone_fraction": [0.1, 0.7, 0.05],
                    "hill_radius": [0.3, 1.2, 0.05], "hill_roughness": [0.0, 0.8, 0.05],
                    "soil_start": [0.0, 3.0, 0.5],
                    "birth_lo": [1, 26, 1], "birth_hi": [1, 26, 1], "survive_lo": [0, 26, 1],
                    "survive_hi": [0, 26, 1], "spore": [0.0, 0.01, 0.0001], "life_density": [0.0, 0.6, 0.05],
                    "rain_rate": [0.0, 0.2, 0.005], "rain_amount": [0, 15, 1], "dry_rate": [0.0, 1.0, 0.05],
                    "erode_rate": [0.0, 0.1, 0.002], "slide_drop": [1, 6, 1], "slide_rate": [0.0, 1.0, 0.05],
                    "trunk": [1, 10, 1], "branch": [0, 6, 1], "spacing": [0, 5, 1],
                    "crown_light": [0, 15, 1], "tree_seed": [0.0, 0.2, 0.005],
                    "spore_rate": [0.0, 0.01, 0.0001], "p_shock": [0.0, 0.01, 0.0001],
                    "mutate": [0.0, 0.05, 0.001], "rescue": [1, 256, 1],
                    "seed_density": [0.0, 0.5, 0.01], "seed_animals": [0.0, 0.2, 0.005]}
    WORLD_LABELS = {"n": "размер куба", "seed_world": "сид мира", "seed_mut": "сид жизни",
                    "variant": "вариант: 0 жизнь 3D / 1 экология / 2 волны",
                    "cyc_threshold": "волны: порог соседей", "cyc_states": "волны: цветов по кругу",
                    "stone_fraction": "высота горы", "hill_radius": "радиус горы",
                    "hill_roughness": "складки", "soil_start": "почвы в низинах на старте",
                    "birth_lo": "жизнь: рождение от", "birth_hi": "жизнь: рождение до",
                    "survive_lo": "жизнь: выживание от", "survive_hi": "жизнь: выживание до",
                    "spore": "жизнь: споры", "life_density": "жизнь: стартовая плотность",
                    "rain_rate": "дождь: шанс в столбце", "rain_amount": "дождь: влага",
                    "dry_rate": "высыхание", "erode_rate": "мох точит камень",
                    "slide_drop": "почва едет с перепада", "slide_rate": "шанс схода почвы",
                    "trunk": "ствол (клеток)", "branch": "крона (ярусов)", "spacing": "просвет стволов",
                    "crown_light": "света хватает кроне", "tree_seed": "трава → дерево",
                    "spore_rate": "споры мха", "p_shock": "выветривание",
                    "mutate": "шанс мутации", "rescue": "чаще в пустой нише",
                    "seed_density": "стартовый засев растений", "seed_animals": "стартовый засев зверей"}
    FIXED_GENES = ("speed",)

    # ------------------------------------------------------------- мир
    def n_species(self, cfg):
        return cfg.n_species

    WAVE_NAMES = ("алые", "янтарные", "лаймовые", "бирюзовые", "синие", "лиловые", "розовые", "серые")
    WAVE_COLORS = ("#e53935", "#fb8c00", "#c0ca33", "#26a69a", "#1e88e5", "#8e24aa", "#ec407a", "#9e9e9e")

    def species_names(self, cfg):
        if cfg.variant == 2:
            return list(self.WAVE_NAMES[:cfg.n_species])
        return list(LIFE_NAMES) if cfg.variant == 0 else list(NAMES)

    def species_colors(self, cfg):
        if cfg.variant == 2:
            return list(self.WAVE_COLORS[:cfg.n_species])
        return list(LIFE_COLORS) if cfg.variant == 0 else list(COLORS)

    def mobile_species(self, cfg):
        if cfg.variant != 1:
            return []
        g = np.asarray(cfg.genomes)
        return [i + 1 for i in range(len(g)) if g[i, IDX["speed"]] > 0]

    def init_state(self, cfg, xp):
        rng = np.random.default_rng(cfg.seed_world)
        n = cfg.n
        stone_h = relief(cfg, rng)
        soil_h = np.zeros((n, n), np.int32)
        if cfg.soil_start > 0 and cfg.variant == 1:
            rel = (stone_h - stone_h.min()) / max(stone_h.max() - stone_h.min(), 1)
            soil_h = np.round(cfg.soil_start * (1 - rel) ** 2).astype(np.int32)
        surf = stone_h + soil_h
        species = np.zeros((n, n, n), np.uint8)
        energy = np.zeros((n, n, n), np.uint8)
        anim = np.zeros((n, n), np.uint8)
        aen = np.zeros((n, n), np.uint8)
        if cfg.variant == 0:
            self._seed_life(cfg, rng, species, surf)
        elif cfg.variant == 2:
            zz = np.arange(n)[None, None, :]
            room = zz >= surf[:, :, None]
            species[room] = rng.integers(1, cfg.n_species + 1, int(room.sum()))
        else:
            self._seed_eco(cfg, rng, species, energy, anim, aen, soil_h, surf)
        state = {
            "species": xp.asarray(species),
            "energy": xp.asarray(energy),
            "age": xp.zeros((n, n, n), np.uint8),
            "stone_h": xp.asarray(stone_h),
            "soil_h": xp.asarray(soil_h),
            "water_h": xp.zeros((n, n), np.int32),
            "wet": xp.zeros((n, n), np.uint8),
            "anim": xp.asarray(anim), "aen": xp.asarray(aen),
            "aage": xp.zeros((n, n), np.uint8),
            "genomes": xp.asarray(np.asarray(cfg.genomes, np.int32)),
            "pops": [0] * cfg.n_species,
            "gen": 0,
            # координаты для хеша — один раз, без пересчёта на каждый шаг
            "cx": xp.arange(n, dtype=U32)[:, None, None],
            "cy": xp.arange(n, dtype=U32)[None, :, None],
            "cz": xp.arange(n, dtype=U32)[None, None, :],
        }
        if cfg.variant == 1:
            self._place_animals(state, xp)
        return state, stone_h.copy()

    def _seed_life(self, cfg, rng, species, surf):
        n = cfg.n
        zz = np.arange(n)[None, None, :]
        above = zz >= surf[:, :, None]
        near = zz < (surf[:, :, None] + max(3, n // 6))
        pick = rng.random((n, n, n)) < cfg.life_density
        cells = above & near & pick
        species[cells] = rng.integers(1, cfg.n_species + 1, int(cells.sum()))

    def _seed_eco(self, cfg, rng, species, energy, anim, aen, soil_h, surf):
        n = cfg.n
        g = np.asarray(cfg.genomes)
        bare, ground = soil_h == 0, soil_h > 0
        xs, ys = np.nonzero(bare & (rng.random((n, n)) < cfg.seed_density))
        species[xs, ys, surf[xs, ys]] = MOSS; energy[xs, ys, surf[xs, ys]] = 6
        xs, ys = np.nonzero(ground & (rng.random((n, n)) < cfg.seed_density * 3))
        species[xs, ys, surf[xs, ys]] = GRASS; energy[xs, ys, surf[xs, ys]] = 6
        # деревья — редкие саженцы на почве с просветом
        xs, ys = np.nonzero(ground & (rng.random((n, n)) < cfg.seed_density * 0.4))
        taken = np.zeros((n, n), bool)
        for x, y in zip(xs, ys):
            if taken[max(0, x - cfg.spacing):x + cfg.spacing + 1, max(0, y - cfg.spacing):y + cfg.spacing + 1].any():
                continue
            taken[x, y] = True
            h = surf[x, y]
            for dz in range(min(3, n - 1 - h)):
                species[x, y, h + dz] = TREE; energy[x, y, h + dz] = 8
        # звери — на проходимых столбцах
        xs, ys = np.nonzero(rng.random((n, n)) < cfg.seed_animals)
        for x, y in zip(xs, ys):
            if species[x, y, surf[x, y]] == TREE:
                continue
            anim[x, y] = HERB if rng.random() < 0.8 else PRED
            aen[x, y] = int(g[anim[x, y] - 1, IDX["repro"]]) * 2 // 3

    # ------------------------------------------------------------- шаг
    def step(self, state, cfg, xp, correlate, gen):
        if cfg.variant == 0:
            pops = self._step_life(state, cfg, xp, correlate, gen)
        elif cfg.variant == 2:
            pops = self._step_waves(state, cfg, xp, correlate, gen)
        else:
            pops = self._step_eco(state, cfg, xp, correlate, gen)
        state["gen"] = gen
        state["pops"] = pops
        return pops

    # ---- вариант 0: жизнь 3D ------------------------------------------
    def _step_life(self, state, cfg, xp, correlate, gen):
        sp = state["species"]
        n = cfg.n
        S = cfg.n_species
        surf = (state["stone_h"] + state["soil_h"])[:, :, None]
        zz = state["cz"].astype(np.int32)
        room = zz >= surf
        alive = sp > 0
        cnt = correlate(alive.astype(np.int32), xp.asarray(K_ALL), mode="constant")
        survive = alive & (cnt >= cfg.survive_lo) & (cnt <= cfg.survive_hi)
        born = (~alive) & room & (cnt >= cfg.birth_lo) & (cnt <= cfg.birth_hi)
        # цвет новорождённого — большинство среди соседей, ничья — по хешу
        best = xp.zeros(sp.shape, np.int32)
        bestc = xp.full(sp.shape, -1, np.int32)
        tie = hash32(xp, state["cx"], state["cy"], state["cz"], gen, cfg.seed_mut ^ S_TIE)
        for s in range(1, S + 1):
            c = correlate((sp == s).astype(np.int32), xp.asarray(K_ALL), mode="constant")
            # чуть-чуть хеша к счётчику: у равных побеждает случайный, но
            # детерминированный
            c4 = (c << 3) + ((tie >> U32(s * 3)) & U32(7)).astype(np.int32)
            better = c4 > bestc
            bestc = xp.where(better, c4, bestc)
            best = xp.where(better, s, best)
        # споры: редкая жизнь из ничего, чтобы автомат не замирал
        h = hash32(xp, state["cx"], state["cy"], state["cz"], gen, cfg.seed_mut ^ S_SPORE)
        spore = (~alive) & room & (zz < surf + 3) & (h < thr(cfg.spore))
        spore_sp = ((h >> U32(8)) % U32(S)).astype(np.int32) + 1
        new = xp.where(survive, sp.astype(np.int32), 0)
        new = xp.where(born, best, new)
        new = xp.where(spore & (new == 0), spore_sp, new)
        state["species"] = new.astype(np.uint8)
        age = state["age"].astype(np.int32)
        state["age"] = xp.where((new == sp.astype(np.int32)) & (new > 0), xp.minimum(age + 1, 255), 0).astype(np.uint8)
        pops = xp.bincount(state["species"].ravel().astype(np.int64), minlength=S + 1)[1:S + 1]
        return [int(v) for v in np.asarray(pops.get() if hasattr(pops, "get") else pops)]

    # ---- вариант 2: волны (циклический автомат) -----------------------
    # Каждая клетка над рельефом окрашена в один из S цветов по кругу. Клетка
    # цвета s перекрашивается в s+1, если хотя бы T соседей уже этого цвета.
    # Из случайного шума за десятки поколений вырастают спиральные волны,
    # которые потом бегут по кубу вечно: картинка структурная, а не кипящая,
    # а событий (перекрасок) — как раз столько, сколько фронтов волн.
    def _step_waves(self, state, cfg, xp, correlate, gen):
        sp = state["species"].astype(np.int32)
        S = cfg.n_species
        surf = (state["stone_h"] + state["soil_h"])[:, :, None]
        room = state["cz"].astype(np.int32) >= surf
        nxt = xp.where(sp >= S, 1, sp + 1)
        flip = xp.zeros(sp.shape, bool)
        for s in range(1, S + 1):
            c = correlate((sp == s).astype(np.int32), xp.asarray(K_ALL), mode="constant")
            flip = flip | ((nxt == s) & (c >= cfg.cyc_threshold))
        new = xp.where(flip & room & (sp > 0), nxt, sp)
        age = state["age"].astype(np.int32)
        state["age"] = xp.where(flip, 0, xp.minimum(age + 1, 255)).astype(np.uint8)
        state["species"] = new.astype(np.uint8)
        pops = xp.bincount(state["species"].ravel().astype(np.int64), minlength=S + 1)[1:S + 1]
        return [int(v) for v in np.asarray(pops.get() if hasattr(pops, "get") else pops)]

    # ---- вариант 1: экология ------------------------------------------
    def _step_eco(self, state, cfg, xp, correlate, gen):  # noqa: C901
        n = cfg.n
        G = state["genomes"]
        g = np.asarray(cfg.genomes, np.int32)
        cx, cy, cz = state["cx"], state["cy"], state["cz"]
        cx2, cy2 = cx[:, :, 0], cy[:, :, 0]
        zz = cz.astype(np.int32)
        stone_h, soil_h, wet = state["stone_h"], state["soil_h"], state["wet"].astype(np.int32)
        sp = state["species"].astype(np.int32)
        en = state["energy"].astype(np.int32)
        age = state["age"].astype(np.int32)
        anim, aen, aage = state["anim"].astype(np.int32), state["aen"].astype(np.int32), state["aage"].astype(np.int32)
        pops_prev = state.get("pops") or [0] * 5
        seed = cfg.seed_mut

        def h2(salt):
            return hash32(xp, cx2, cy2, 0, gen, seed ^ salt)

        def h3(salt):
            return hash32(xp, cx, cy, cz, gen, seed ^ salt)

        # животные в объёме — только маркеры; на время шага убираем
        is_anim3 = (sp == HERB) | (sp == PRED)
        sp = xp.where(is_anim3, 0, sp); en = xp.where(is_anim3, 0, en)

        # 1. дождь и влага (двумерно, правило тяги: столбец принимает сток от
        #    тех соседей, для которых он — самый низкий)
        surf = stone_h + soil_h
        rain = h2(S_RAIN) < thr(cfg.rain_rate)
        wet = xp.minimum(15, wet + xp.where(rain, cfg.rain_amount, 0))
        down, low = downhill(xp, surf)                     # куда стекает столбец
        has_down = low < surf
        out = xp.where(has_down, wet >> 2, 0)
        inflow = xp.zeros_like(wet)
        for d, (dx, dy) in enumerate(DIRS):
            # сосед s = c - (dx,dy) стекает в направлении d, значит его «вниз» = c
            src_out = shift2(xp, out, dx, dy)
            src_dir = shift2(xp, down, dx, dy, fill=-1)
            inflow = inflow + xp.where(src_dir == d, src_out, 0)
        wet = xp.clip(wet - out + inflow, 0, 15)
        wet = xp.where(h2(S_DRY) < thr(cfg.dry_rate), xp.maximum(wet - 1, 0), wet)

        # 2. мох точит камень: единица камня становится единицей почвы
        surf_sp = xp.take_along_axis(sp, xp.clip(surf, 0, n - 1)[:, :, None], axis=2)[:, :, 0]
        erode_thr = thr(cfg.erode_rate) + (thr(cfg.erode_rate * 3) // U32(16)) * wet.astype(U32)
        erode = (surf_sp == MOSS) & (stone_h > cfg.floor) & (h2(S_ERODE) < erode_thr)
        stone_h = xp.where(erode, stone_h - 1, stone_h)
        soil_h = xp.where(erode, soil_h + 1, soil_h)

        # 3. почва съезжает к самому низкому соседу (тяга: низкий столбец
        #    принимает одну единицу от одного из высоких)
        surf = stone_h + soil_h
        surf_before = surf
        down, _ = downhill(xp, surf)
        drop_ok = h2(S_TIE) < thr(cfg.slide_rate)
        give = xp.zeros_like(surf, dtype=bool)
        take = xp.zeros_like(surf, dtype=bool)
        order = (h2(S_TIE) >> U32(4)) % U32(4)
        for k in range(4):
            d_arr = ((order + U32(k)) % U32(4)).astype(np.int32)
            for d, (dx, dy) in enumerate(DIRS):
                cand = (d_arr == d) & ~take
                src_surf = shift2(xp, surf, dx, dy, fill=-10 ** 6)
                src_soil = shift2(xp, soil_h, dx, dy)
                src_ok = shift2(xp, drop_ok, dx, dy)
                src_dir = shift2(xp, down, dx, dy, fill=-1)     # «вниз» у источника — это мы
                pull = cand & (src_soil > 0) & (src_surf - surf >= cfg.slide_drop) & src_ok & (src_dir == d)
                take = take | pull
                give = give | shift2(xp, pull, -dx, -dy)
        soil_h = soil_h + take.astype(np.int32) - give.astype(np.int32)
        old_surf, surf = surf_before, stone_h + soil_h
        # столбец переехал на ±1 — всё живое в нём едет вместе с подложкой
        # (в железе это смещение регистра столбца)
        dz = xp.clip(old_surf - surf, -1, 1)
        if bool((dz != 0).any()):
            idx = xp.clip(zz + dz[:, :, None], 0, n - 1)
            sp = xp.take_along_axis(sp, idx, axis=2)
            en = xp.take_along_axis(en, idx, axis=2)
            age = xp.take_along_axis(age, idx, axis=2)
            sp = xp.where(zz < surf[:, :, None], 0, sp)

        # 4. свет сверху (регистр столбца) и вода снизу
        absorb_lut = xp.asarray(np.concatenate([[0], g[:, IDX["absorb"]]]).astype(np.int32))
        absorb = absorb_lut[sp]
        cum = xp.cumsum(absorb[:, :, ::-1], axis=2)[:, :, ::-1] - absorb
        light = xp.clip(15 - cum, 0, 15)
        hgt = zz - surf[:, :, None]
        water = xp.clip(wet[:, :, None] - xp.maximum(hgt, 0), 0, 15)
        # сок дерева: клетка отдаёт четверть энергии вниз, если ниже — дерево
        below_tree = xp.zeros_like(sp, dtype=bool); below_tree[:, :, 1:] = sp[:, :, :-1] == TREE
        above_tree = xp.zeros_like(sp, dtype=bool); above_tree[:, :, :-1] = sp[:, :, 1:] == TREE
        sap_out = xp.where((sp == TREE) & below_tree, en >> 2, 0)
        sap_in = xp.zeros_like(en); sap_in[:, :, :-1] = sap_out[:, :, 1:]

        # 5. энергия растений
        lut = lambda name: xp.asarray(np.concatenate([[0], g[:, IDX[name]]]).astype(np.int32))  # noqa: E731
        res = (light * lut("light")[sp] + water * lut("water")[sp]) >> 4
        plant = (sp >= MOSS) & (sp <= TREE)
        en = xp.where(plant, xp.clip(en + res - lut("metab")[sp] - sap_out + sap_in, 0, 15), 0)
        age = xp.where(plant, xp.minimum(age + 1, 255), 0)

        # 6. гибель растений
        life = lut("lifespan")[sp]
        die = plant & ((en == 0) & (age > 1) | ((life > 0) & (age >= life)))
        die = die | (plant & (h3(S_SHOCK) < thr(cfg.p_shock)))
        die = die | ((sp == MOSS) & (soil_h[:, :, None] > 0))        # под мхом появилась почва
        die = die | ((sp == MOSS) | (sp == GRASS)) & (hgt != 0)      # висит или закопано
        # дерево держится на дереве под собой (или на земле): сломался ствол —
        # всё выше осыпается по клетке за поколение
        tree_ok = (sp == TREE) & ((hgt == 0) | below_tree | (
            correlate(((sp == TREE) & (hgt < (cfg.trunk + cfg.branch))).astype(np.int32),
                      xp.asarray(kernel3(face=1, edge=1, corner=0, below=1, level=1, above=0)),
                      mode="constant") > 0) & (hgt >= cfg.trunk))
        die = die | ((sp == TREE) & ~tree_ok)
        sp = xp.where(die, 0, sp); en = xp.where(die, 0, en); age = xp.where(die, 0, age)

        # 7. рождение растений (конкуренция по избытку над порогом)
        K = xp.asarray(K_ANISO)
        empty = (sp == 0) & (hgt >= 0)
        ground = hgt == 0
        bare, soil = (soil_h == 0)[:, :, None], (soil_h > 0)[:, :, None]
        # столбцы стволов — по КОРНЮ (клетка дерева на поверхности): крона в
        # соседнем столбце стволом не считается, иначе кроны расползались бы
        # по всему миру, а зверям было бы негде ходить
        tree_col = ((sp == TREE) & ground).any(axis=2)
        # просвет между стволами: окно (2*spacing+1)²
        near_tree = tree_col.copy()
        for _ in range(cfg.spacing):
            near_tree = near_tree | xp.stack(neighbours2(xp, near_tree)).any(axis=0)
        best = xp.zeros_like(sp); best_score = xp.full(sp.shape, -1, np.int32)
        parent_e = {}
        tie = h3(S_TIE)
        for s in (MOSS, GRASS):
            mine = sp == s
            cnt = correlate(mine.astype(np.int32), K, mode="constant")
            rich = correlate((mine & (en >= int(g[s - 1, IDX["cost"]]) + 1)).astype(np.int32), K, mode="constant") > 0
            ok = empty & ground & rich & (cnt >= int(g[s - 1, IDX["birth"]]))
            ok = ok & (bare if s == MOSS else soil)
            score = (cnt - int(g[s - 1, IDX["birth"]])) * 4 + ((tie >> U32(s)) & U32(3)).astype(np.int32)
            better = ok & (score > best_score)
            best = xp.where(better, s, best); best_score = xp.where(better, score, best_score)
            parent_e[s] = cnt
        # дерево: ствол растёт над деревом, крона — рядом с деревом выше ствола
        trunk_up = empty & below_tree & (hgt >= 1) & (hgt < cfg.trunk)
        en_below = xp.zeros_like(en); en_below[:, :, 1:] = en[:, :, :-1]
        trunk_up = trunk_up & (en_below >= int(g[TREE - 1, IDX["cost"]]) + 2)
        crown_nb = correlate((sp == TREE).astype(np.int32),
                             xp.asarray(kernel3(face=1, edge=1, corner=0, below=1, level=1, above=0)),
                             mode="constant")
        crown = empty & (hgt >= cfg.trunk) & (hgt < cfg.trunk + cfg.branch) & (crown_nb > 0) \
            & (light >= cfg.crown_light) & (~ground)
        # крона не дальше одного столбца от ствола
        trunk_near = tree_col | xp.stack(neighbours2(xp, tree_col)).any(axis=0)
        crown = crown & trunk_near[:, :, None] & ((tie >> U32(7)) & U32(3) == 0)
        best = xp.where(trunk_up | crown, TREE, best)
        # трава в тесноте даёт дерево (мутация), если просвет позволяет
        grass_cnt = parent_e[GRASS]
        empty_tree = pops_prev[TREE - 1] == 0
        tree_thr = thr(min(1.0, cfg.tree_seed * (cfg.rescue if empty_tree else 1)))
        sprout = empty & ground & soil & (grass_cnt >= 4) & (~near_tree[:, :, None]) & (h3(S_MUT) < tree_thr)
        best = xp.where(sprout, TREE, best)
        # спора мха на голом камне: единственная жизнь из ничего
        spore = empty & ground & bare & (h3(S_SPORE) < thr(cfg.spore_rate)) & (best == 0)
        best = xp.where(spore, MOSS, best)
        born = best > 0
        start_e = lut("cost")[best] + 2
        sp = xp.where(born, best, sp); en = xp.where(born, start_e, en); age = xp.where(born, 0, age)
        # родители платят за новорождённых по соседству (локально, в следующем такте)
        for s in (MOSS, GRASS):
            newborn = correlate((born & (best == s)).astype(np.int32), xp.asarray(K_ALL), mode="constant")
            pay = xp.minimum(newborn * int(g[s - 1, IDX["cost"]]) // 2, en)
            en = xp.where((sp == s) & (age > 0), en - pay, en)

        # 8. звери — двумерный автомат на поверхности
        surf_sp = xp.take_along_axis(sp, xp.clip(surf, 0, n - 1)[:, :, None], axis=2)[:, :, 0]
        food = ((surf_sp == GRASS) | (surf_sp == MOSS)).astype(np.int32)
        walk = ~tree_col
        food3 = food.copy()
        for dx, dy in DIRS:
            food3 = food3 + shift2(xp, food, dx, dy)
        herb = anim == HERB
        pred = anim == PRED
        prey3 = herb.astype(np.int32)
        for dx, dy in DIRS:
            prey3 = prey3 + shift2(xp, herb.astype(np.int32), dx, dy)
        smell = xp.where(pred, prey3, food3)
        # желание: направление с самым сытным соседом, при ничьей — хеш
        hm = h2(S_MOVE)
        want = xp.full((n, n), -1, np.int32); want_v = xp.full((n, n), -1, np.int32)
        for d, (dx, dy) in enumerate(DIRS):
            v = shift2(xp, smell, -dx, -dy, fill=-1) * 4 + ((hm >> U32(d * 2)) & U32(3)).astype(np.int32)
            pass_ok = (xp.abs(shift2(xp, surf, -dx, -dy, fill=10 ** 6) - surf) <= 1) & shift2(xp, walk, -dx, -dy)
            v = xp.where(pass_ok, v, -1)
            better = v > want_v
            want = xp.where(better, d, want); want_v = xp.where(better, v, want_v)
        # сытый на еде — стоит и ест; голодный идёт
        eating = (anim > 0) & (smell > 0) & (xp.where(pred, prey3, food) > 0)
        want = xp.where(eating & (h2(S_EAT) < thr(0.7)), -1, want)
        # 8a. еда: травоядное съедает растение под собой, хищник — соседа
        bite = herb & (food > 0) & (h2(S_EAT) < thr(0.5))
        surf_idx = xp.clip(surf, 0, n - 1)[:, :, None]
        eaten_val = lut("value")[surf_sp]
        aen = aen + xp.where(bite, eaten_val, 0)
        sp = xp.where(bite[:, :, None] & (zz == surf[:, :, None]), 0, sp)
        # хищник: пустая клетка не нужна — жертва рядом; жертву выбирает
        # ХИЩНИК (у жертвы может быть несколько охотников — все кусают)
        hunt_thr = thr(g[PRED - 1, IDX["hunt"]] / 16.0 * (1 - g[HERB - 1, IDX["armor"]] / 16.0))
        attack = pred & (prey3 - herb.astype(np.int32) > 0) & (h2(S_EAT) < hunt_thr)
        killed = xp.zeros((n, n), bool)
        for dx, dy in DIRS:
            killed = killed | (shift2(xp, attack, dx, dy) & herb)
        anim = xp.where(killed, 0, anim)
        fed = xp.zeros((n, n), bool)
        for dx, dy in DIRS:
            fed = fed | shift2(xp, killed, -dx, -dy)
        aen = xp.where(attack & fed, aen + int(g[HERB - 1, IDX["value"]]), aen)
        herb = anim == HERB
        # 8b. движение тягой: пустой проходимый столбец впускает одного соседа,
        #     который хочет именно сюда
        moved_in = xp.zeros((n, n), np.int32); moved_e = xp.zeros((n, n), np.int32); moved_a = xp.zeros((n, n), np.int32)
        gone = xp.zeros((n, n), bool)
        free = (anim == 0) & walk
        order = (h2(S_MOVE) >> U32(8)) % U32(4)
        for k in range(4):
            d_arr = ((order + U32(k)) % U32(4)).astype(np.int32)
            for d, (dx, dy) in enumerate(DIRS):
                src_a = shift2(xp, anim, dx, dy); src_w = shift2(xp, want, dx, dy, fill=-1)
                pull = free & (moved_in == 0) & (d_arr == d) & (src_a > 0) & (src_w == d)
                moved_in = xp.where(pull, src_a, moved_in)
                moved_e = xp.where(pull, shift2(xp, aen, dx, dy), moved_e)
                moved_a = xp.where(pull, shift2(xp, aage, dx, dy), moved_a)
                gone = gone | shift2(xp, pull, -dx, -dy)
        anim = xp.where(gone, 0, anim); aen = xp.where(gone, 0, aen); aage = xp.where(gone, 0, aage)
        anim = xp.where(moved_in > 0, moved_in, anim)
        aen = xp.where(moved_in > 0, moved_e, aen); aage = xp.where(moved_in > 0, moved_a, aage)
        # 8c. обмен, старение, смерть
        alive = anim > 0
        aen = xp.where(alive, xp.clip(aen - lut("metab")[anim], 0, 15), 0)
        aage = xp.where(alive, xp.minimum(aage + 1, 255), 0)
        alife = lut("lifespan")[anim]
        adie = alive & ((aen == 0) | ((alife > 0) & (aage >= alife)))
        anim = xp.where(adie, 0, anim); aen = xp.where(adie, 0, aen); aage = xp.where(adie, 0, aage)
        # 8d. деление тягой: пустой столбец берёт детёныша у сытого соседа;
        #     родитель платит, когда предлагает (есть свободный сосед)
        repro = lut("repro")[anim]
        offer = (anim > 0) & (aen >= repro)
        free = (anim == 0) & walk
        child = xp.zeros((n, n), np.int32); child_e = xp.zeros((n, n), np.int32)
        order = (h2(S_REPRO) >> U32(8)) % U32(4)
        for k in range(4):
            d_arr = ((order + U32(k)) % U32(4)).astype(np.int32)
            for d, (dx, dy) in enumerate(DIRS):
                src = shift2(xp, offer, dx, dy)
                pull = free & (child == 0) & (d_arr == d) & src
                child = xp.where(pull, shift2(xp, anim, dx, dy), child)
                child_e = xp.where(pull, shift2(xp, aen, dx, dy) >> 1, child_e)
        has_free = xp.zeros((n, n), bool)
        for dx, dy in DIRS:
            has_free = has_free | shift2(xp, free, -dx, -dy)
        aen = xp.where(offer & has_free, aen >> 1, aen)
        # мутация детёныша: травоядное → хищник (чаще, если хищников нет)
        empty_pred = pops_prev[PRED - 1] == 0
        mut_thr = thr(min(1.0, cfg.mutate * (cfg.rescue if empty_pred else 1)))
        child = xp.where((child == HERB) & (h2(S_MUT) < mut_thr), PRED, child)
        anim = xp.where(child > 0, child, anim); aen = xp.where(child > 0, xp.maximum(child_e, 2), aen)
        aage = xp.where(child > 0, 0, aage)
        # 8e. трава в тесноте рождает травоядное (растение → зверь), чаще,
        #     если травоядных нет вовсе
        empty_herb = pops_prev[HERB - 1] == 0
        herb_thr = thr(min(1.0, cfg.mutate * (cfg.rescue if empty_herb else 1)))
        gsurf = xp.take_along_axis(parent_e[GRASS], surf_idx, axis=2)[:, :, 0]
        spawn = (anim == 0) & walk & (gsurf >= 6) & (h2(S_LIFE) < herb_thr)
        anim = xp.where(spawn, HERB, anim); aen = xp.where(spawn, 6, aen); aage = xp.where(spawn, 0, aage)

        # 9. записать зверей в объём как маркеры (на клетку выше растения)
        state["anim"], state["aen"], state["aage"] = anim.astype(np.uint8), aen.astype(np.uint8), aage.astype(np.uint8)
        surf_sp = xp.take_along_axis(sp, surf_idx, axis=2)[:, :, 0]
        az = xp.clip(surf + (surf_sp > 0).astype(np.int32), 0, n - 1)
        amask = (zz == az[:, :, None]) & (anim > 0)[:, :, None]
        sp = xp.where(amask, anim[:, :, None], sp)

        state["species"] = sp.astype(np.uint8)
        state["energy"] = en.astype(np.uint8)
        state["age"] = age.astype(np.uint8)
        state["stone_h"], state["soil_h"] = stone_h, soil_h
        state["wet"] = wet.astype(np.uint8)
        state["water_h"] = (wet // 6).astype(np.int32)
        pops = xp.bincount(state["species"].ravel().astype(np.int64), minlength=6)[1:6]
        pops = np.asarray(pops.get() if hasattr(pops, "get") else pops)
        return [int(v) for v in pops]

    def _place_animals(self, state, xp):
        """Маркеры зверей в объёме для стартового кадра."""
        n = state["species"].shape[0]
        surf = state["stone_h"] + state["soil_h"]
        sp = state["species"].astype(np.int32)
        zz = state["cz"].astype(np.int32)
        surf_sp = xp.take_along_axis(sp, xp.clip(surf, 0, n - 1)[:, :, None], axis=2)[:, :, 0]
        az = xp.clip(surf + (surf_sp > 0).astype(np.int32), 0, n - 1)
        anim = state["anim"].astype(np.int32)
        amask = (zz == az[:, :, None]) & (anim > 0)[:, :, None]
        state["species"] = xp.where(amask, anim[:, :, None], sp).astype(np.uint8)

    # ------------------------------------------------------------- засев
    def seed(self, state, cfg, xp, rng, count=None, gen=0):
        from ..backend import to_cpu
        n = cfg.n
        count = int(count if count is not None else cfg.reseed_count)
        rng_cpu = np.random.default_rng((cfg.seed_mut ^ 0x1234) + gen)
        species = to_cpu(state["species"]).copy()
        energy = to_cpu(state["energy"]).copy()
        surf = to_cpu(state["stone_h"] + state["soil_h"])
        xs = rng_cpu.integers(0, n, count); ys = rng_cpu.integers(0, n, count)
        zs = np.clip(surf[xs, ys], 0, n - 1)
        free = species[xs, ys, zs] == 0
        xs, ys, zs = xs[free], ys[free], zs[free]
        if cfg.variant == 0:
            species[xs, ys, zs] = rng_cpu.integers(1, cfg.n_species + 1, len(xs))
        else:
            soil = to_cpu(state["soil_h"])[xs, ys]
            species[xs, ys, zs] = np.where(soil > 0, GRASS, MOSS)
            energy[xs, ys, zs] = 6
        state["species"] = xp.asarray(species); state["energy"] = xp.asarray(energy)
        state["last_reseed"] = gen
        return int(len(xs))

    def starters_json(self, cfg):
        names = self.species_names(cfg)
        return [{"i": i + 1, "name": nm, "habitat": "куб", "on": True} for i, nm in enumerate(names)]

    # ------------------------------------------------------------- панель
    def species_mass(self, cfg):
        return [1.0] * cfg.n_species

    def world_params(self):
        return self.WORLD_PARAMS

    def to_json(self, cfg, state=None):
        genomes = np.asarray(cfg.genomes).tolist()
        if cfg.variant == 0:
            genomes = genomes[:4]
        return {
            "engine": self.name,
            "fields": list(GENES), "labels": LABELS,
            "ranges": {k: list(v) for k, v in RANGES.items()},
            "names": self.species_names(cfg), "colors": self.species_colors(cfg),
            "genomes": genomes,
            "world": {k: getattr(cfg, k) for k in self.WORLD_PARAMS},
            "world_labels": self.WORLD_LABELS, "world_ranges": self.WORLD_RANGES,
            "fixed_genes": list(self.FIXED_GENES),
            "starters": self.starters_json(cfg),
            "reseed": seeding_json(cfg),
            "mobile": self.mobile_species(cfg),
            "iron": True,
        }

    def apply_genomes(self, cfg, state, genomes, xp):
        g = np.rint(np.asarray(genomes, dtype=np.float64)).astype(np.int32)
        if g.ndim != 2 or g.shape[1] != len(GENES):
            raise ValueError(f"геномы: ожидается таблица вид × {len(GENES)}")
        full = np.asarray(cfg.genomes, np.int32).copy()
        full[:len(g)] = g[:len(full)]
        for name, (lo, hi, _) in RANGES.items():
            full[:, IDX[name]] = np.clip(full[:, IDX[name]], lo, hi)
        cfg.genomes = full
        state["genomes"] = xp.asarray(full)

    def randomize(self, cfg, rng):
        g = np.asarray(cfg.genomes, np.int32).copy()
        for s in range(len(g)):
            for name, i in IDX.items():
                if name in self.FIXED_GENES:
                    continue
                lo, hi, _ = RANGES[name]
                if g[s, IDX["speed"]] == 0 and name in ("hunt", "armor", "repro"):
                    continue
                if g[s, IDX["speed"]] > 0 and name in ("light", "water", "absorb", "birth"):
                    continue
                g[s, i] = int(rng.integers(lo, hi + 1))
            if g[s, IDX["speed"]] == 0:
                g[s, IDX["light"]] = int(rng.integers(6, 15))
                g[s, IDX["water"]] = int(rng.integers(1, 8))
                g[s, IDX["metab"]] = int(rng.integers(0, 2))
        return g

    GENE_DOCS = {
        "light": "Доля света, которую клетка превращает в энергию (0..16 из 16).",
        "water": "Доля воды столбца, которую клетка берёт (0..16 из 16). Вода "
                 "убывает на единицу с каждой клеткой высоты над поверхностью.",
        "absorb": "Сколько единиц света (из 15) клетка гасит для всех, кто ниже.",
        "birth": "Сколько взвешенных соседей нужно пустой клетке, чтобы родиться "
                 "(снизу — 2, сбоку — 1, сверху — 0; грань ×2, ребро/угол ×1).",
        "cost": "Цена клетки: с ней рождаются, её половину платит каждый сосед-родитель.",
        "metab": "Единиц энергии в поколение на жизнь.",
        "lifespan": "Возраст, после которого клетка гибнет; 0 — не стареет.",
        "value": "Сколько энергии получает тот, кто это съест.",
        "hunt": "Шанс укуса хищника (из 16), помноженный на (16 − броня жертвы).",
        "armor": "Броня травоядного (из 16).",
        "repro": "Энергия, с которой зверь предлагает детёныша свободному соседу.",
        "speed": "1 — ходит по поверхности, 0 — растение. Роль, не сила.",
    }

    def gene_docs(self):
        return dict(self.GENE_DOCS)


RULES = IronRules()
