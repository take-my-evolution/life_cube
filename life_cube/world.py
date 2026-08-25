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

    # камень занимает stone_fraction высоты куба; холмы и складки вокруг этого
    base = max(3.0, cfg.stone_fraction * n)
    amp = cfg.relief_amp * base
    relief = (base
              + amp * 0.55 * np.sin(xx / (n / 7.5)) * np.cos(yy / (n / 5.5))
              + amp * 0.30 * np.sin((xx + yy) / (n / 11.0))
              + amp * 0.15 * np.sin(xx / (n / 23.0) + 2.0) * np.sin(yy / (n / 19.0))
              + rng.normal(0, max(0.5, amp * 0.06), (n, n)))
    relief = np.clip(relief, 2, n - 4).astype(int)

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

    # растения и животные засеваются отдельно: животных обычно меньше, и они
    # должны стартовать вперемешку с едой, а не сплошным пятном
    mobile = cfg.mobile_mask()
    plants = np.flatnonzero(~mobile) + 1
    animals = np.flatnonzero(mobile) + 1
    if len(animals) == 0 or cfg.animal_share <= 0:
        cycle = (plants[np.arange(k) % len(plants)]).astype(np.int8)
    else:
        n_anim = int(round(k * cfg.animal_share))
        cycle = np.concatenate([
            plants[np.arange(k - n_anim) % max(len(plants), 1)],
            animals[np.arange(n_anim) % len(animals)],
        ]).astype(np.int8)
    rng.shuffle(cycle)
    species[seed_mask] = cycle

    energy = np.zeros((n, n, n), dtype=np.float32)
    energy[species > 0] = cfg.plant_energy
    anim_cells = np.isin(species, animals) if len(animals) else np.zeros_like(species, bool)
    energy[anim_cells] = cfg.start_energy

    return (xp.asarray(stone), xp.asarray(wet), xp.asarray(species),
            relief, xp.asarray(energy))
