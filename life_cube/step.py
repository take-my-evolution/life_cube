"""Одно поколение мира.

Ключевые механики (важно не сломать при правках):
  * конкуренция по ИЗБЫТКУ (R - need), а не по абсолюту — иначе один вид
    выигрывает везде и ниши не возникают;
  * нет нижнего порога выживания по соседям — одиночка не умирает от
    одиночества, только от нехватки ресурса или тесноты;
  * выветривание (p_shock) не даёт системе застыть в неподвижной точке.
"""

import numpy as np

from .config import IDX, Config
from .fields import light_field, water_field, water_supply, resource
from .motion import animals_step


def step(state: dict, cfg: Config, xp, correlate, gen: int = 0):
    """Одно поколение. state — словарь с полями
    species/stone/soil/wet/kernel/genomes/rng. Мутирует state, возвращает
    список населения по видам."""
    species, stone, soil, wet = (state["species"], state["stone"],
                                 state["soil"], state["wet"])
    K, G = state["kernel"], state["genomes"]
    n_sp = cfg.n_species
    rng = state["rng"]

    mobile = cfg.mobile_mask()
    plant_ids = [s for s in range(1, n_sp + 1) if not mobile[s - 1]]

    # растения не видят животных: те не дают опоры, не теснят и не затеняют
    is_animal = xp.zeros(species.shape, dtype=bool)
    for s in range(1, n_sp + 1):
        if mobile[s - 1]:
            is_animal = is_animal | (species == s)
    plants = (species > 0) & ~is_animal

    alive = plants
    nb = correlate(alive.astype(xp.float32), K, mode="constant", cval=0.0)

    idx = xp.clip(species.astype(xp.int32) - 1, 0, n_sp - 1)
    absorb = xp.where(alive, G[idx, 0], 0.0).astype(xp.float32)
    n_fields = G.shape[1]

    L = light_field(alive, absorb, xp)
    W = water_field(alive, stone, soil, wet, cfg, xp, state.get("maxfilter"))

    # вода, дотянувшаяся до пустой клетки: снизу или (для ветвления) сбоку
    Wsup = water_supply(W, cfg, xp)

    # есть ли под пустой клеткой опора — живое тело, камень или почва.
    # С опорой это рост вверх (вес up), без опоры — ветвление вбок
    # (вес branch × свет: ветвиться выгодно только на свету).
    supported = xp.zeros(species.shape, dtype=bool)
    supported[:, :, 1:] = (alive | stone | soil)[:, :, :-1]

    # --- рождение: голосование соседей за то, чей вид займёт пустое место ---
    empty = (~(species > 0)) & (~stone)
    best_score = xp.full(species.shape, -1.0, dtype=xp.float32)
    best_sp = xp.zeros(species.shape, dtype=xp.int8)

    for s in plant_ids:
        g = cfg.genomes[s - 1]
        mine = correlate((species == s).astype(xp.float32), K,
                         mode="constant", cval=0.0)
        R = resource(g, L, Wsup, xp)
        ok = (empty
              & (mine > g[2] * cfg.birth_own)     # свои дали достаточно голосов
              & (nb > g[2])                        # опоры хватает
              & (nb < g[2] + cfg.birth_window)     # но не давка
              & (R > g[3])                         # ресурса хватает виду
              & (Wsup > cfg.water_min))            # вода дотянулась
        branch = float(g[5]) if len(g) > 5 else 0.0
        ok = ok & (supported | (branch > 0))
        # ВАЖНО: сравниваем избыток над собственным порогом, а не абсолют.
        weight = xp.where(supported, g[1], branch * L)
        score = xp.where(ok, mine * weight * (R - g[3]), -1.0).astype(xp.float32)
        upd = score > best_score
        best_score = xp.where(upd, score, best_score)
        best_sp = xp.where(upd, xp.int8(s), best_sp)

    born = best_sp > 0

    # --- выживание: тесно или ресурса не хватает ---
    Rlive = xp.zeros(species.shape, dtype=xp.float32)
    need = xp.zeros(species.shape, dtype=xp.float32)
    for s in plant_ids:
        m = species == s
        g = cfg.genomes[s - 1]
        Rlive = xp.where(m, resource(g, L, W, xp), Rlive)
        need = xp.where(m, g[3] * cfg.surv_factor, need)

    survive = alive & (nb < cfg.crowd_max) & (Rlive > need)

    # --- мутация при делении: пока просто смена вида (в пределах растений) ---
    if plant_ids:
        mut = born & (rng.random(species.shape) < cfg.p_mutate)
        alt = xp.asarray(np.array(plant_ids, dtype=np.int8))
        pick = rng.integers(0, len(plant_ids), species.shape)
        best_sp = xp.where(mut, alt[pick], best_sp)

    # --- растворение камня: клетка, стоящая на камне, превращает его в почву ---
    touch = xp.zeros_like(stone)
    touch[:, :, :-1] = alive[:, :, 1:]
    diss = stone & touch & (rng.random(species.shape) < cfg.p_dissolve)
    stone = stone & ~diss
    soil = soil | diss

    # --- выветривание: не даёт системе застыть в неподвижной точке ---
    shock = alive & (rng.random(species.shape) < cfg.p_shock)
    survive = survive & ~shock

    # животные остаются на местах: их ход считает animals_step
    new_species = xp.where(born, best_sp,
                           xp.where(survive | is_animal, species, xp.int8(0))
                           ).astype(xp.int8)

    energy = state.get("energy")
    if energy is not None:
        energy = xp.where(born, xp.float32(cfg.plant_energy), energy)
        energy = xp.where(new_species == 0, xp.float32(0), energy)
        state["energy"] = energy

    state["species"], state["stone"], state["soil"] = new_species, stone, soil

    if mobile.any():
        animals_step(state, cfg, xp, correlate, rng)

    sp = state["species"]
    return [int((sp == s).sum()) for s in range(1, n_sp + 1)]
