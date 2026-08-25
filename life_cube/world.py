"""Мир: анизотропное ядро соседства, рельеф камня, влажность, засев."""

import numpy as np

from .config import Config


def build_kernel(xp=np):
    """Анизотропное ядро соседства 3x3x3.

    Веса: снизу 1.6 (опора), сбоку 1.0, сверху 0.45; граневые x1.25,
    угловые x0.6. Центр нулевой — клетка себя не считает.
    Именно это превращает бесформенное расползание в рост вверх.
    """
    K = np.zeros((3, 3, 3), dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                w = 1.0
                if dz == -1:
                    w = 1.6
                elif dz == 1:
                    w = 0.45
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if manhattan == 1:
                    w *= 1.25
                elif manhattan == 3:
                    w *= 0.6
                K[dx + 1, dy + 1, dz + 1] = w
    return xp.asarray(K)


def build_world(cfg: Config, xp=np):
    """Рельеф камня, карта влажности подложки и стартовый засев спорами.

    Всё детерминировано от cfg.seed_world: один и тот же сид — один и тот же
    ландшафт, независимо от того, что потом произойдёт с мутациями.

    Возвращает (stone, wet, species, relief); relief всегда numpy на CPU.
    """
    rng = np.random.default_rng(cfg.seed_world)
    n = cfg.n

    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

    # холмы и складки; последнее слагаемое — мелкая шероховатость
    relief = (6 + 5 * np.sin(xx / 17.0) * np.cos(yy / 23.0)
              + 3 * np.sin((xx + yy) / 11.0)
              + rng.normal(0, 0.8, (n, n)))
    relief = np.clip(relief, 3, max(4, n // 7)).astype(int)

    zz = np.arange(n)[None, None, :]
    stone = zz < relief[:, :, None]

    # влажность камня неоднородна по площади — отсюда берутся ниши
    wet = np.clip(0.55 + 0.45 * np.sin(xx / 29.0 + 1.2) * np.cos(yy / 19.0),
                  0.15, 1.0).astype(np.float32)

    # споры садятся ровно на первый свободный слой над камнем
    surface = np.zeros((n, n, n), dtype=bool)
    surface[xx, yy, relief] = True
    seed_mask = surface & (rng.random((n, n)) < cfg.seed_density)[:, :, None]

    species = np.zeros((n, n, n), dtype=np.int8)
    k = int(seed_mask.sum())
    if k == 0:
        raise RuntimeError("засев пуст — подними seed_density")
    # виды раздаются поровну: иначе исход решает случайность стартовых чисел
    cycle = (np.arange(k) % len(cfg.genomes) + 1).astype(np.int8)
    rng.shuffle(cycle)
    species[seed_mask] = cycle

    return xp.asarray(stone), xp.asarray(wet), xp.asarray(species), relief
