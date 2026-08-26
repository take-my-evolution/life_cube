"""Геномы видов и параметры мира.

Геном — 14 чисел. Первые шесть работают у растений (сидячих), остальные —
у всех, но большая часть имеет смысл только у подвижных существ.

    0  absorb     сколько света клетка поглощает (и, значит, отбрасывает тень)
    1  up         тяга вверх: множитель в оценке привлекательности места
    2  birth      порог взвешенной суммы соседей для рождения (растения)
    3  need       минимальный ресурс, ниже которого растение не живёт
    4  water      жадность к воде: 0 = чистый фотосинтетик, 1 = чистый водолюб
    5  branch     ветвление: вес бокового роста, умножается на свет в точке
    6  hunt       сила атаки и доля энергии жертвы, которая усваивается
    7  trophic    уровень: 0 растение, 1 травоядное, 2 хищник. Жертва — уровень
                  на единицу ниже. Хищник в мире без травоядных вымирает, даже
                  если травы полно: это и есть механизм баланса.
    8  speed      шагов в поколение; > 0 делает вид ПОДВИЖНЫМ (не растёт, ходит)
    9  sense      радиус чутья: на сколько клеток чувствует еду
   10  metabolism трата энергии за поколение
   11  repro      порог энергии для деления (дочерней достаётся половина)
   12  lifespan   предельный возраст в поколениях, 0 = не стареет
   13  armor      шанс отбиться от нападения (0..1)
"""

from dataclasses import dataclass, field

import numpy as np

GENOME_FIELDS = ("absorb", "up", "birth", "need", "water", "branch",
                 "hunt", "trophic", "speed", "sense", "metabolism", "repro",
                 "lifespan", "armor")

GENOME_LABELS = {
    "absorb": "поглощение света", "up": "тяга вверх", "birth": "порог рождения",
    "need": "минимум ресурса", "water": "жадность к воде", "branch": "ветвление",
    "hunt": "сила атаки", "trophic": "уровень: 0 раст / 1 трав / 2 хищ",
    "speed": "скорость (0 = сидячий)", "sense": "радиус чутья",
    "metabolism": "обмен веществ", "repro": "порог деления",
    "lifespan": "предел возраста", "armor": "броня",
}

GENOME_RANGES = {
    "absorb": (0.0, 1.0, 0.01), "up": (0.0, 3.0, 0.05), "birth": (0.5, 4.0, 0.05),
    "need": (0.0, 2.0, 0.02), "water": (0.0, 1.0, 0.01), "branch": (0.0, 2.0, 0.05),
    "hunt": (0.0, 1.0, 0.01), "trophic": (0, 2, 1), "speed": (0, 4, 1),
    "sense": (0, 8, 1), "metabolism": (0.0, 0.5, 0.005), "repro": (0.5, 12.0, 0.1),
    "lifespan": (0, 400, 10), "armor": (0.0, 0.95, 0.05),
}

#             absorb  up  birth  need water branch hunt tro spd sns metab repro life armor
DEFAULT_GENOMES = np.array([
    [0.08, 0.60, 1.40, 0.18, 0.70, 0.50, 0.0, 0, 0, 0, 0.00, 0.0, 0, 0.00],  # 1 мох
    [0.25, 0.90, 1.60, 0.46, 0.30, 0.15, 0.0, 0, 0, 0, 0.00, 0.0, 0, 0.00],  # 2 корка
    [0.60, 1.70, 1.85, 0.42, 0.25, 0.00, 0.0, 0, 0, 0, 0.00, 0.0, 0, 0.00],  # 3 башня
    [0.14, 0.80, 1.50, 0.24, 0.85, 0.30, 0.0, 0, 0, 0, 0.00, 0.0, 0, 0.00],  # 4 теневой
    [0.40, 1.20, 1.80, 0.38, 0.45, 0.25, 0.0, 0, 0, 0, 0.00, 0.0, 0, 0.00],  # 5 универсал
    [0.70, 1.50, 1.75, 0.40, 0.35, 1.10, 0.0, 0, 0, 0, 0.00, 0.0, 0, 0.00],  # 6 дерево
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.60, 1, 1, 5, 0.07, 3.5, 250, 0.50],  # 7 травоядное
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 2, 2, 7, 0.080, 5.0, 350, 0.05],  # 8 хищник
], dtype=np.float32)

SPECIES_NAMES = ("мох", "корка", "башня", "теневой", "универсал", "дерево",
                 "травоядное", "хищник")

SPECIES_COLORS = ("#d9d9d9", "#2ec7b8", "#f2c14e", "#c05ce0", "#f2683c",
                  "#7bd94a", "#4a9ef2", "#f24a9e")

IDX = {name: i for i, name in enumerate(GENOME_FIELDS)}


@dataclass
class Config:
    n: int = 128                  # размер куба по каждой оси
    gens: int = 200               # число поколений
    seed_world: int = 20260825    # сид мира: рельеф, влажность, засев
    seed_mut: int = 20260825      # сид случайности во время жизни

    # физика мира
    water_decay: float = 0.90     # доля воды, доходящая до следующей клетки вверх
    water_min: float = 0.075      # ниже этого вода не поддерживает рождение
    crowd_max: float = 7.8        # верхний предел соседей: смерть от тесноты
    surv_factor: float = 0.62     # порог ресурса для выживания = need * этого
    birth_window: float = 3.40    # ширина окна соседей для рождения
    birth_own: float = 0.55       # доля порога, которую должны дать СВОИ соседи
    lateral_steps: int = 6        # предел растекания воды вбок по телу за поколение (с ранним выходом)
    lateral_decay: float = 0.80   # потеря воды на каждой боковой клетке (0 = не течёт)
    require_substrate: bool = True  # ткань без связи с землёй (вода = 0) гибнет

    # рельеф
    stone_fraction: float = 0.33  # какую долю высоты куба занимает камень
    relief_amp: float = 0.45      # размах холмов относительно толщины камня

    # животные
    plant_energy: float = 0.9     # энергия, запасённая одной клеткой растения
    eat_efficiency: float = 0.8   # доля энергии жертвы, которая достаётся едоку
    move_noise: float = 0.15      # доля случайности в выборе направления
    start_energy: float = 4.0     # энергия существа при засеве и после деления

    # события
    p_mutate: float = 0.0006      # смена вида при делении (заглушка, см. доку)
    p_dissolve: float = 0.0016    # растворение камня клеткой, стоящей на нём
    p_shock: float = 0.0009       # случайная гибель (выветривание)

    seed_density: float = 0.006   # доля точек поверхности под споры
    animal_share: float = 0.30    # какая доля засева — животные
    genomes: np.ndarray = field(default_factory=lambda: DEFAULT_GENOMES.copy())

    @property
    def n_species(self) -> int:
        return len(self.genomes)

    def gene(self, s, name):
        """Значение гена по имени для вида s (1-based)."""
        i = IDX[name]
        g = self.genomes[s - 1]
        return float(g[i]) if i < len(g) else 0.0

    def mobile_mask(self):
        """Булев вектор по видам: подвижен ли вид (speed > 0)."""
        i = IDX["speed"]
        if self.genomes.shape[1] <= i:
            return np.zeros(self.n_species, bool)
        return np.asarray(self.genomes)[:, i] > 0

    def to_json(self):
        """Всё, что нужно панели конструктора в браузере."""
        return {
            "fields": list(GENOME_FIELDS),
            "labels": GENOME_LABELS,
            "ranges": {k: list(v) for k, v in GENOME_RANGES.items()},
            "names": list(SPECIES_NAMES)[: self.n_species],
            "colors": list(SPECIES_COLORS)[: self.n_species],
            "genomes": np.asarray(self.genomes).tolist(),
            "world": {"n": self.n, "seed_world": self.seed_world,
                      "seed_mut": self.seed_mut, "seed_density": self.seed_density,
                      "animal_share": self.animal_share,
                      "stone_fraction": self.stone_fraction,
                      "relief_amp": self.relief_amp,
                      "p_shock": self.p_shock, "p_dissolve": self.p_dissolve,
                      "plant_energy": self.plant_energy,
                      "eat_efficiency": self.eat_efficiency,
                      "move_noise": self.move_noise},
        }
