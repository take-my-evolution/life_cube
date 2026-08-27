"""Движок «лишайник» — эволюция под давлением среды.

Концепт (docs/engines/lichen.md, кратко):
  * Мир: плоское дно и каменная возвышенность (~stone_fraction высоты куба).
    Вся поверхность камня в начале покрыта ОДНИМ видом лишайника.
  * Энергия приходит сверху (свет). Дождь капает в случайные точки, вода
    стекает вниз по склону и уносит с собой почву.
  * Лишайник с геном `erode` на мокром камне проедает его: камень → почва
    (и получает за это энергию). Почва смывается водой в низины.
  * На почве лишайник (substrate ≈ 0) живёт плохо — стресс. Стресс повышает
    вероятность мутации при делении. Мутация меняет один ген; потомок с новым
    геномом — новый вид. Когда мутация сдвигает `substrate` к 1, потомок
    получает энергию на почве — так «растение» возникает из лишайника само.
  * Вымершие виды освобождают id; вся генеалогия пишется в state["lineage"].

Реализация: подложка — две карты высот (камень, почва), жизнь — 3D массив
видов над ними. Всё векторно на xp (numpy/cupy).
"""

import colorsys
from dataclasses import dataclass, field

import numpy as np

from . import Rules, fork_dynamic
from ..fields import light_field

GENES = ("light", "substrate", "erode", "rain", "up", "absorb",
         "metabolism", "repro", "mut")
LABELS = {
    "light": "усвоение света", "substrate": "субстрат: 0 камень … 1 почва",
    "erode": "разъедание камня (при воде)", "rain": "польза от дождя",
    "up": "рост вверх (0 = плоский)", "absorb": "тень (поглощение света)",
    "metabolism": "обмен веществ", "repro": "порог деления",
    "mut": "базовая частота мутаций",
}
RANGES = {
    "light": (0.0, 1.0, 0.01), "substrate": (0.0, 1.0, 0.01), "erode": (0.0, 1.0, 0.01),
    "rain": (0.0, 1.0, 0.01), "up": (0.0, 1.0, 0.01), "absorb": (0.0, 1.0, 0.01),
    "metabolism": (0.01, 0.4, 0.005), "repro": (0.5, 6.0, 0.05), "mut": (0.0, 0.3, 0.005),
}
IDX = {g: i for i, g in enumerate(GENES)}
MAX_SPECIES = 64

#              light subst erode rain  up   absorb metab repro mut
LICHEN = np.array([0.80, 0.00, 0.60, 0.60, 0.00, 0.50, 0.06, 2.0, 0.003], np.float32)


@dataclass
class LichenConfig:
    n: int = 96
    gens: int = 400
    seed_world: int = 20260826
    seed_mut: int = 20260826

    # рельеф
    floor: int = 2                  # плоское каменное дно
    stone_fraction: float = 0.30    # высота возвышенности (доля куба)
    hill_radius: float = 0.55       # радиус холма (доля половины стороны)
    hill_roughness: float = 0.25    # шероховатость склонов

    # дождь и вода
    rain_rate: float = 0.03         # доля столбцов, куда за поколение падает капля
    rain_amount: float = 1.0
    wet_decay: float = 0.70         # сколько воды остаётся к следующему поколению
    flow: float = 0.5               # доля воды, стекающей к нижнему соседу
    wash: float = 0.35              # вероятность смыва единицы почвы водой за поколение
    min_drop: int = 1               # перепад высот, при котором почва/вода стекают

    # жизнь
    start_energy: float = 1.0
    light_gain: float = 0.9         # энергия за полный свет при полном усвоении
    mismatch_floor: float = 0.02    # доля энергии на «чужом» субстрате (на нём лишайник голодает)
    rain_gain: float = 0.8          # прибавка к росту при полной воде (множитель)
    erode_gain: float = 0.8         # энергия за проеденную единицу камня
    erode_rate: float = 0.25        # вероятность проесть при erode=1 и воде=1
    lifespan: int = 300
    takeover: float = 0.35          # чужую клетку можно вытеснить, если её энергия < этой доли её порога деления (она голодает)
    energy_cap: float = 2.5         # запас энергии не больше repro × этого: без запаса чужой субстрат убивает быстро
    crowd_max: int = 5              # максимум живых соседей по 6-связности
    stress_mut: float = 0.5         # добавка к вероятности мутации при полном стрессе
    min_change: float = 0.06        # мутация меньше этой доли диапазона — не новый вид
    mut_scale: float = 0.2          # шаг мутации в долях диапазона гена
    max_new_species: int = 4        # сколько новых видов может появиться за поколение
    p_shock: float = 0.0005

    genomes: np.ndarray = field(default_factory=lambda: np.zeros((MAX_SPECIES, len(GENES)), np.float32))

    def __post_init__(self):
        if not self.genomes.any():
            self.genomes[0] = LICHEN

    @property
    def n_species(self):
        return MAX_SPECIES


def hill_relief(cfg, rng):
    n = cfg.n
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cx, cy = n / 2 + rng.uniform(-n / 8, n / 8), n / 2 + rng.uniform(-n / 8, n / 8)
    r = np.hypot(xx - cx, yy - cy) / (n / 2 * cfg.hill_radius)
    bump = np.clip(1.0 - r ** 2, 0, 1) ** 1.2
    H = max(3.0, cfg.stone_fraction * n) - cfg.floor
    rough = (np.sin(xx / (n / 9.0) + rng.uniform(0, 6)) * np.cos(yy / (n / 7.0) + rng.uniform(0, 6))
             + 0.5 * np.sin((xx + yy) / (n / 13.0)))
    relief = cfg.floor + H * bump * (1 + cfg.hill_roughness * rough * bump)
    relief = relief + rng.normal(0, 0.3, (n, n)) * bump
    return np.clip(np.round(relief), cfg.floor, n - 6).astype(np.int32)


def _hsl(i):
    h = (i * 0.618033988) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.75)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


class LichenRules(Rules):
    name = "lichen"
    title = "Лишайник: дождь, эрозия, стресс-мутации"
    summary = ("Поверхность камня покрыта одним лишайником. Дождь, эрозия камня "
               "в почву, смыв почвы в низины; на чужом субстрате — стресс, стресс — "
               "мутации, мутации — новые виды. Растение должно возникнуть само.")
    doc = "docs/engines/lichen.md"
    Config = LichenConfig
    dynamic_species = True
    can_fork = True
    terrain_changes = True
    WORLD_PARAMS = ("n", "seed_world", "seed_mut", "stone_fraction", "hill_radius",
                    "rain_rate", "wet_decay", "wash", "erode_rate", "stress_mut",
                    "mut_scale", "max_new_species", "lifespan", "p_shock")
    WORLD_RANGES = {"n": [32, 256, 32], "stone_fraction": [0.1, 0.6, 0.01],
                    "hill_radius": [0.2, 1.0, 0.05], "rain_rate": [0.0, 0.3, 0.005],
                    "wet_decay": [0.0, 0.95, 0.05], "wash": [0.0, 1.0, 0.05],
                    "erode_rate": [0.0, 1.0, 0.05], "stress_mut": [0.0, 1.0, 0.02],
                    "mut_scale": [0.02, 0.6, 0.02], "max_new_species": [0, 16, 1],
                    "lifespan": [20, 2000, 10], "p_shock": [0, 0.01, 0.0001]}
    WORLD_LABELS = {"n": "размер куба", "seed_world": "сид мира", "seed_mut": "сид мутаций",
                    "stone_fraction": "высота горы (доля куба)", "hill_radius": "радиус горы",
                    "rain_rate": "дождь: доля столбцов за поколение", "wet_decay": "вода: сколько остаётся",
                    "wash": "смыв почвы водой", "erode_rate": "скорость проедания камня",
                    "stress_mut": "мутации от стресса", "mut_scale": "размер мутации",
                    "max_new_species": "новых видов за поколение (макс.)",
                    "lifespan": "предел возраста", "p_shock": "выветривание"}

    # ------------------------------------------------------------- мир
    def init_state(self, cfg, xp):
        rng = np.random.default_rng(cfg.seed_world)
        n = cfg.n
        stone_h = hill_relief(cfg, rng)                  # (n,n) высота камня
        soil_h = np.zeros((n, n), np.int32)
        species = np.zeros((n, n, n), np.uint8)
        xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        species[xx, yy, stone_h] = 1                     # лишайник на всей поверхности
        energy = np.where(species > 0, cfg.start_energy, 0).astype(np.float32)
        wet = np.zeros((n, n), np.float32)
        pops = np.zeros(MAX_SPECIES, np.int64); pops[0] = n * n
        state = {
            "species": xp.asarray(species),
            "energy": xp.asarray(energy),
            "age": xp.zeros((n, n, n), dtype=xp.int32),
            "stone_h": xp.asarray(stone_h),
            "soil_h": xp.asarray(soil_h),
            "wet": xp.asarray(wet),
            "genomes": xp.asarray(cfg.genomes),
            "rng": xp.random.default_rng(cfg.seed_mut),
            "rng_cpu": np.random.default_rng(cfg.seed_mut ^ 0x5bd1e995),
            # реестр видов: id -> dict(parent, born, died, genome)
            "registry": {1: {"parent": 0, "born": 0, "died": None,
                             "genome": cfg.genomes[0].tolist(), "peak": n * n}},
            "lineage": [],                               # закрытые записи
            "free_ids": list(range(2, MAX_SPECIES + 1)),
            "gen": 0,
        }
        self._sync_volumes(state, cfg, xp)
        return state, stone_h.copy()

    def _sync_volumes(self, state, cfg, xp):
        """3D-маски камня и почвы из карт высот (для снимков и света)."""
        n = cfg.n
        zz = xp.arange(n)[None, None, :]
        sh, so = state["stone_h"][:, :, None], state["soil_h"][:, :, None]
        state["stone"] = zz < sh
        state["soil"] = (zz >= sh) & (zz < sh + so)

    # ------------------------------------------------------------- шаг
    def step(self, state, cfg, xp, correlate, gen):
        n = cfg.n
        rng = state["rng"]
        G = state["genomes"]
        species, energy, age = state["species"], state["energy"], state["age"]
        stone_h, soil_h, wet = state["stone_h"], state["soil_h"], state["wet"]
        alive = species > 0
        idx = xp.clip(species.astype(xp.int32) - 1, 0, MAX_SPECIES - 1)

        def gene(name):
            return xp.where(alive, G[idx, IDX[name]], 0.0).astype(xp.float32)

        surface = stone_h + soil_h                          # высота подложки (n,n)
        zz = xp.arange(n)[None, None, :]
        on_surface = alive & (zz == surface[:, :, None])   # клетки, стоящие на подложке
        on_soil2d = soil_h > 0

        # --- 1. дождь и вода --------------------------------------------------
        drops = rng.random((n, n)) < cfg.rain_rate
        wet = wet * cfg.wet_decay + drops * cfg.rain_amount
        # стекание к самому низкому из 4 соседей, если перепад >= min_drop
        wet, soil_h, stone_moved = self._flow(wet, soil_h, surface, cfg, xp, rng)

        # --- 2. свет ---------------------------------------------------------
        absorb = gene("absorb")
        L = light_field(alive, absorb, xp)

        # --- 3. энергия ------------------------------------------------------
        soil_below = on_soil2d[:, :, None] & on_surface     # стоит на почве
        stone_below = (~on_soil2d[:, :, None]) & on_surface
        sub = gene("substrate")
        # клетка не на подложке (выросла вверх) — корка без грунта, кормится
        # только на mismatch_floor: ген `up` в этом движке почти не выгоден.
        # Это осознанно: лишайник здесь — двумерная корка, объёмный рост —
        # тема движка «экология» (и будущего гена стебля)
        match = xp.where(soil_below, sub, xp.where(stone_below, 1.0 - sub, 0.0))
        factor = cfg.mismatch_floor + (1.0 - cfg.mismatch_floor) * match
        wet3 = wet[:, :, None] * on_surface
        # дождь — множитель к свету, а не отдельный корм: иначе мокрая почва
        # кормит лишайник и стресса на ней нет (наступали)
        gain = (cfg.light_gain * gene("light") * L * factor
                * (1.0 + cfg.rain_gain * gene("rain") * xp.minimum(wet3, 1.0)))

        # --- 4. эрозия: лишайник на мокром камне проедает его ----------------
        er = gene("erode")
        p_er = cfg.erode_rate * er * xp.minimum(wet3, 1.0) * stone_below
        eroding = (rng.random(species.shape) < p_er)
        eroded2d = eroding.any(axis=2) & (stone_h > cfg.floor)
        gain = gain + xp.where(eroding & eroded2d[:, :, None], cfg.erode_gain, 0.0)
        stone_h = stone_h - eroded2d.astype(stone_h.dtype)
        soil_h = soil_h + eroded2d.astype(soil_h.dtype)
        # клетка стояла на камне, камень стал почвой ТОЙ ЖЕ высоты — стоит дальше

        energy = xp.where(alive, energy + gain - gene("metabolism"), 0.0)
        cap = xp.where(alive, G[idx, IDX["repro"]] * cfg.energy_cap, 0.0)
        energy = xp.minimum(energy, cap)
        age = xp.where(alive, age + 1, 0)

        # --- 5. смерть -------------------------------------------------------
        nb6 = self._count6(alive, xp)
        dead = alive & ((energy <= 0) | (age > cfg.lifespan) | (nb6 > cfg.crowd_max)
                        | (rng.random(species.shape) < cfg.p_shock))
        # почва поднялась под клеткой: клетка ПОДНИМАЕТСЯ вместе с ней (сидит
        # сверху), а не хоронится — иначе низины, куда всё смывается, были бы
        # мёртвой зоной и эволюции там негде было бы случиться (наступали)
        new_surface = stone_h + soil_h
        rise = (new_surface - surface)
        for k in (1, 2, 3, 4):
            lift = alive & (zz == (new_surface - k)[:, :, None]) & (rise >= k)[:, :, None]
            if not bool(lift.any()):
                continue
            tgt = xp.zeros_like(alive); tgt[:, :, k:] = lift[:, :, :-k]
            tgt = tgt & ~alive
            src = xp.zeros_like(alive); src[:, :, :-k] = tgt[:, :, k:]
            species = xp.where(tgt, xp.roll(species, k, axis=2), species)
            energy = xp.where(tgt, xp.roll(energy, k, axis=2), energy)
            age = xp.where(tgt, xp.roll(age, k, axis=2), age)
            species = xp.where(src, xp.uint8(0), species)
            energy = xp.where(src, 0.0, energy)
            alive = species > 0
        surface = new_surface
        buried = alive & (zz < surface[:, :, None])      # кого не удалось поднять
        dead = dead | buried
        species = xp.where(dead, xp.uint8(0), species)
        energy = xp.where(dead, 0.0, energy)
        age = xp.where(dead, 0, age)
        alive = species > 0

        # --- 6. тяжесть: без опоры снизу клетка падает на одну ---------------
        for _ in range(2):
            support = xp.zeros_like(alive); support[:, :, 0] = True
            support[:, :, 1:] = alive[:, :, :-1] | (zz[:, :, 1:] <= surface[:, :, None])
            falling = alive & ~support
            if not bool(falling.any()):
                break
            tgt = xp.zeros_like(alive); tgt[:, :, :-1] = falling[:, :, 1:]
            tgt = tgt & ~alive
            src_ok = xp.zeros_like(alive); src_ok[:, :, 1:] = tgt[:, :, :-1]
            species = xp.where(tgt, xp.roll(species, -1, axis=2), species)
            energy = xp.where(tgt, xp.roll(energy, -1, axis=2), energy)
            age = xp.where(tgt, xp.roll(age, -1, axis=2), age)
            species = xp.where(src_ok, xp.uint8(0), species)
            energy = xp.where(src_ok, 0.0, energy)
            alive = species > 0

        # --- 7. деление и мутации --------------------------------------------
        idx = xp.clip(species.astype(xp.int32) - 1, 0, MAX_SPECIES - 1)
        repro = xp.where(alive, G[idx, IDX["repro"]], 1e9).astype(xp.float32)
        parents = alive & (energy > repro)
        new_species_born = 0
        if bool(parents.any()):
            up_p = xp.where(alive, G[idx, IDX["up"]], 0.0).astype(xp.float32)
            # стресс: не тот субстрат + тень
            on_surface = alive & (zz == surface[:, :, None])
            sub = xp.where(alive, G[idx, IDX["substrate"]], 0.0)
            match = xp.where(on_soil2d[:, :, None] & on_surface, sub,
                             xp.where(on_surface, 1.0 - sub, 0.0))
            stress = xp.clip(0.7 * (1.0 - match) + 0.3 * (1.0 - L), 0.0, 1.0)
            mut_base = xp.where(alive, G[idx, IDX["mut"]], 0.0)
            p_mut = xp.clip(mut_base + cfg.stress_mut * stress, 0.0, 0.9)

            species, energy, age, new_species_born = self._reproduce(
                state, cfg, xp, rng, species, energy, age, parents, up_p, p_mut,
                surface, gen, stress, sub, on_soil2d)

        state.update(species=species, energy=energy, age=age,
                     stone_h=stone_h, soil_h=soil_h, wet=wet, gen=gen)
        self._sync_volumes(state, cfg, xp)
        return self._pops(state, cfg, xp, gen)

    # ------------------------------------------------------------- вода/почва
    @staticmethod
    def _shift2(a, dx, dy, xp, fill=0):
        out = xp.full_like(a, fill)
        sx = slice(max(dx, 0), a.shape[0] + min(dx, 0))
        sy = slice(max(dy, 0), a.shape[1] + min(dy, 0))
        tx = slice(max(-dx, 0), a.shape[0] + min(-dx, 0))
        ty = slice(max(-dy, 0), a.shape[1] + min(-dy, 0))
        out[sx, sy] = a[tx, ty]
        return out

    def _flow(self, wet, soil_h, surface, cfg, xp, rng):
        """Вода и почва стекают к самому низкому соседу при перепаде >= min_drop."""
        big = surface.max() + 1000
        best_drop = xp.zeros_like(surface)
        best_dir = xp.full(surface.shape, -1, dtype=xp.int32)
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for k, (dx, dy) in enumerate(dirs):
            nsurf = self._shift2(surface, dx, dy, xp, fill=big)   # высота соседа
            drop = surface - nsurf
            better = drop > best_drop
            best_drop = xp.where(better, drop, best_drop)
            best_dir = xp.where(better, k, best_dir)
        can = best_drop >= cfg.min_drop
        # вода
        out = xp.where(can, wet * cfg.flow, 0.0)
        wet = wet - out
        # почва: единица уходит с вероятностью wash × вода
        soil_go = can & (soil_h > 0) & (rng.random(surface.shape) < cfg.wash * xp.minimum(wet + out, 1.0))
        soil_h = soil_h - soil_go
        for k, (dx, dy) in enumerate(dirs):
            m = best_dir == k
            wet = wet + self._shift2(xp.where(m, out, 0.0), -dx, -dy, xp)
            soil_h = soil_h + self._shift2(xp.where(m & soil_go, 1, 0).astype(soil_h.dtype), -dx, -dy, xp)
        return wet, soil_h, None

    @staticmethod
    def _count6(alive, xp):
        a = alive.astype(xp.float32)
        c = xp.zeros_like(a)
        c[1:] += a[:-1]; c[:-1] += a[1:]
        c[:, 1:] += a[:, :-1]; c[:, :-1] += a[:, 1:]
        c[:, :, 1:] += a[:, :, :-1]; c[:, :, :-1] += a[:, :, 1:]
        return c

    # ------------------------------------------------------------- деление
    DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1))

    def _reproduce(self, state, cfg, xp, rng, species, energy, age, parents, up_p,
                   p_mut, surface, gen, stress, sub, on_soil2d):
        """Деление. Клетка НА ПОДЛОЖКЕ ползёт по земле: боковой потомок
        садится на поверхность соседнего столбца (какой бы высоты она ни была) —
        так лишайник остаётся двумерной коркой и не громоздится на склонах
        (наступали: при боковом шаге «на той же высоте» он вырастал башнями).
        Вверх — только с вероятностью `up`. Клетки над подложкой (результат
        роста вверх) делятся по 3D-правилу: в соседнюю пустую клетку с опорой."""
        from ..motion import shift
        n = cfg.n
        zz = xp.arange(n)[None, None, :]
        alive = species > 0
        idx = xp.clip(species.astype(xp.int32) - 1, 0, MAX_SPECIES - 1)
        G = state["genomes"]
        repro_here = xp.where(alive, G[idx, IDX["repro"]], 0.0).astype(xp.float32)
        soil_col = on_soil2d[:, :, None] & xp.ones_like(alive)
        surf3 = surface[:, :, None]
        on_surface = alive & (zz == surf3)
        # клетка-хозяин слаба (голодает на чужом субстрате) — её можно вытеснить
        weak_cell = alive & (energy < cfg.takeover * repro_here)

        mutate = rng.random(species.shape) < p_mut
        r = rng.random(species.shape)
        go_up = parents & (r < up_p)
        lateral = parents & ~go_up
        dir4 = (rng.random(species.shape) * 4).astype(xp.int32)
        key = rng.random(species.shape).astype(xp.float32) + 1.0

        child_mut = xp.zeros_like(alive)
        child_stress = xp.zeros(species.shape, dtype=xp.float32)
        child_parent_id = xp.zeros(species.shape, dtype=xp.uint8)
        placed = xp.zeros_like(alive)          # родители, уже поделившиеся

        # ---------- A. по земле: родители на подложке, цель — поверхность соседа
        # сводим к 2D: в столбце родитель на поверхности ровно один
        def col(a):        # значение с поверхности столбца -> 2D
            return (a * on_surface).sum(axis=2) if a.dtype != bool else (a & on_surface).any(axis=2)
        par2 = col(lateral)
        dir2 = col(dir4.astype(xp.float32)).astype(xp.int32)
        key2 = col(key)
        en2 = col(energy)
        sp2 = col(species.astype(xp.float32)).astype(xp.uint8)
        mut2 = col(mutate)
        st2 = col(stress)
        sub2 = col(sub)
        # что лежит на поверхности целевого столбца
        top_sp = sp2                                  # вид на поверхности (0 = пусто)
        top_weak = col(weak_cell)
        dirs2 = ((1, 0), (-1, 0), (0, 1), (0, -1))
        claims2, win2 = [], xp.zeros(par2.shape, dtype=xp.float32)
        for k, (dx, dy) in enumerate(dirs2):
            # родитель в (x,y) хочет в (x+dx,y+dy): смотрим с точки зрения цели
            src_par = self._shift2(par2 & (dir2 == k), -dx, -dy, xp, fill=False)
            src_sp = self._shift2(sp2, -dx, -dy, xp)
            ok_t = src_par & ((top_sp == 0) | (top_weak & (top_sp != src_sp)))
            claims2.append(xp.where(ok_t, self._shift2(key2, -dx, -dy, xp), 0.0))
            win2 = xp.maximum(win2, claims2[-1])
        for k, (dx, dy) in enumerate(dirs2):
            took2 = (claims2[k] > 0) & (claims2[k] == win2)
            if not bool(took2.any()):
                continue
            src2 = self._shift2(took2, dx, dy, xp, fill=False)   # столбцы-родители
            # снимаем с уже поделившихся
            src2 = src2 & ~col(placed)
            took2 = self._shift2(src2, -dx, -dy, xp, fill=False)
            if not bool(took2.any()):
                continue
            p_sp = self._shift2(sp2, -dx, -dy, xp)
            p_en = self._shift2(en2, -dx, -dy, xp)
            p_mut2 = self._shift2(mut2, -dx, -dy, xp, fill=False)
            p_st = self._shift2(st2, -dx, -dy, xp)
            p_sub = self._shift2(sub2, -dx, -dy, xp)
            # 3D-маски: цель — поверхность столбца; родитель — поверхность своего
            tgt3 = took2[:, :, None] & (zz == surf3)
            par3 = src2[:, :, None] & on_surface
            mism2 = xp.where(on_soil2d, 1.0 - p_sub, p_sub)      # чужой ли субстрат потомку
            extra2 = took2 & (rng.random(took2.shape) < cfg.stress_mut * mism2)
            species = xp.where(tgt3, p_sp[:, :, None], species)
            energy = xp.where(tgt3, (p_en * 0.5)[:, :, None], energy)
            age = xp.where(tgt3, 0, age)
            energy = xp.where(par3, energy * 0.5, energy)
            child_mut = child_mut | (tgt3 & (p_mut2 | extra2)[:, :, None])
            child_stress = xp.where(tgt3, xp.maximum(p_st, mism2)[:, :, None], child_stress)
            child_parent_id = xp.where(tgt3, p_sp[:, :, None], child_parent_id)
            placed = placed | par3
        alive = species > 0

        # ---------- B. 3D: вверх для всех желающих, вбок — для клеток над подложкой
        empty = ~alive & (zz >= surf3)
        support = xp.zeros_like(alive); support[:, :, 0] = True
        support[:, :, 1:] = alive[:, :, :-1] | (zz[:, :, 1:] <= surf3)
        free = empty & support
        choice = xp.where(go_up, 4, dir4)
        par3d = parents & ~placed & (go_up | ~on_surface)
        claims, winner = [], xp.zeros(species.shape, dtype=xp.float32)
        for di, d in enumerate(self.DIRS):
            back = tuple(-k for k in d)
            ok = par3d & (choice == di) & shift(free, back, xp, fill=False)
            at = shift(xp.where(ok, key, 0.0), d, xp)
            claims.append(at); winner = xp.maximum(winner, at)
        for di, d in enumerate(self.DIRS):
            took = (claims[di] > 0) & (claims[di] == winner)
            if not bool(took.any()):
                continue
            back = tuple(-k for k in d)
            par = shift(took, back, xp, fill=False) & par3d & ~placed
            took = shift(par, d, xp, fill=False)
            half = xp.where(par, energy * 0.5, 0.0)
            species = xp.where(took, shift(species, d, xp), species)
            energy = xp.where(took, shift(half, d, xp), energy)
            age = xp.where(took, 0, age)
            energy = xp.where(par, energy * 0.5, energy)
            child_mut = child_mut | (took & shift(mutate, d, xp, fill=False))
            child_stress = xp.where(took, shift(stress, d, xp), child_stress)
            child_parent_id = xp.where(took, shift(species, d, xp), child_parent_id)
            placed = placed | par

        # --- новые виды: не больше max_new_species за поколение ---------------
        born = 0
        if cfg.max_new_species > 0 and bool(child_mut.any()) and state["free_ids"]:
            from ..backend import to_cpu
            cand = np.argwhere(to_cpu(child_mut))
            rng_cpu = state["rng_cpu"]
            rng_cpu.shuffle(cand)
            # сперва самые стрессованные потомки: слоты видов ограничены
            cs = to_cpu(child_stress)
            cand = cand[np.argsort(-cs[cand[:, 0], cand[:, 1], cand[:, 2]], kind="stable")]
            reg, G_cpu = state["registry"], cfg.genomes
            for (x, y, z) in cand[: cfg.max_new_species]:
                if not state["free_ids"]:
                    break
                parent = int(to_cpu(child_parent_id[x, y, z]))
                if parent == 0:
                    continue
                g = G_cpu[parent - 1].copy()
                gi = int(rng_cpu.integers(len(GENES)))
                lo, hi, _ = RANGES[GENES[gi]]
                g[gi] = float(np.clip(g[gi] + rng_cpu.normal(0, cfg.mut_scale * (hi - lo)), lo, hi))
                if abs(g[gi] - G_cpu[parent - 1][gi]) < cfg.min_change * (hi - lo):
                    continue                     # слишком мелко — тот же вид
                new_id = state["free_ids"].pop(0)
                G_cpu[new_id - 1] = g
                reg[new_id] = {"parent": parent, "born": gen, "died": None,
                               "genome": g.tolist(), "peak": 1,
                               "changed": GENES[gi]}
                species[x, y, z] = new_id
                born += 1
            if born:
                state["genomes"] = xp.asarray(G_cpu)
        return species, energy, age, born

    # ------------------------------------------------------------- население
    def _pops(self, state, cfg, xp, gen):
        from ..backend import to_cpu
        counts = np.bincount(to_cpu(state["species"]).ravel(), minlength=MAX_SPECIES + 1)[1:]
        reg = state["registry"]
        for sid in list(reg):
            c = int(counts[sid - 1])
            if c > reg[sid]["peak"]:
                reg[sid]["peak"] = c
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
            if sid == 1:
                names[0] = "лишайник"
            else:
                names[sid - 1] = f"#{sid} ← #{r['parent']} ({r.get('changed', '?')}, пок. {r['born']})"
        return names

    def species_colors(self, cfg, state=None):
        return [_hsl(i) for i in range(MAX_SPECIES)]

    def world_params(self):
        return self.WORLD_PARAMS

    def to_json(self, cfg, state=None):
        ids = sorted(state["registry"]) if state else [1]
        return {
            "engine": self.name,
            "fields": list(GENES), "labels": LABELS,
            "ranges": {k: list(v) for k, v in RANGES.items()},
            "ids": ids,
            "names": [self.species_names(cfg, state)[i - 1] for i in ids],
            "colors": [_hsl(i - 1) for i in ids],
            "genomes": [np.asarray(cfg.genomes[i - 1]).tolist() for i in ids],
            "world": {k: getattr(cfg, k) for k in self.WORLD_PARAMS},
            "world_labels": self.WORLD_LABELS, "world_ranges": self.WORLD_RANGES,
            "species_total": len(state["lineage"]) + len(state["registry"]) if state else 1,
            "lineage_tail": (state["lineage"][-30:] if state else []),
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
        """Случайный стартовый лишайник (остальные слоты пусты)."""
        g = np.zeros_like(cfg.genomes)
        for name, i in IDX.items():
            lo, hi, _ = RANGES[name]
            g[0, i] = rng.uniform(lo, hi)
        g[0, IDX["substrate"]] = rng.uniform(0.0, 0.25)   # старт всё же на камне
        g[0, IDX["light"]] = rng.uniform(0.5, 1.0)
        g[0, IDX["metabolism"]] = rng.uniform(0.03, 0.12)
        g[0, IDX["repro"]] = rng.uniform(1.2, 3.0)
        return g


RULES = LichenRules()
