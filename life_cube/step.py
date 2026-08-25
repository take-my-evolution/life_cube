"""Одно поколение мира.

Ключевые механики (важно не сломать при правках):
  * конкуренция по ИЗБЫТКУ (R - need), а не по абсолюту — иначе один вид
    выигрывает везде и ниши не возникают;
  * нет нижнего порога выживания по соседям — одиночка не умирает от
    одиночества, только от нехватки ресурса или тесноты;
  * выветривание (p_shock) не даёт системе застыть в неподвижной точке.
"""

from .config import Config
from .fields import light_field, water_field, water_supply, resource


def step(state: dict, cfg: Config, xp, correlate, gen: int = 0):
    """Одно поколение. state — словарь с полями
    species/stone/soil/wet/kernel/genomes/rng. Мутирует state, возвращает
    список населения по видам."""
    species, stone, soil, wet = (state["species"], state["stone"],
                                 state["soil"], state["wet"])
    K, G = state["kernel"], state["genomes"]
    n_sp = cfg.n_species
    rng = state["rng"]

    alive = species > 0
    nb = correlate(alive.astype(xp.float32), K, mode="constant", cval=0.0)

    idx = xp.clip(species.astype(xp.int32) - 1, 0, n_sp - 1)
    absorb = xp.where(alive, G[idx, 0], 0.0).astype(xp.float32)
    n_fields = G.shape[1]
    has_hunt = n_fields > 6 and float(cfg.genomes[:, 6].max()) > 0
    # взвешенная "хищность" вокруг каждой клетки — сколько её едят
    if has_hunt:
        hunt_map = xp.where(alive, G[idx, 6], 0.0).astype(xp.float32)
        eaten = correlate(hunt_map, K, mode="constant", cval=0.0)
        # добыча — только растения (hunt == 0): хищники друг друга не едят,
        # иначе они питаются собой и размножаются без предела
        plant = alive & (hunt_map <= 0)
        prey_all = correlate(plant.astype(xp.float32), K, mode="constant", cval=0.0)

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
    empty = (~alive) & (~stone)
    best_score = xp.full(species.shape, -1.0, dtype=xp.float32)
    best_sp = xp.zeros(species.shape, dtype=xp.int8)

    for s in range(1, n_sp + 1):
        g = cfg.genomes[s - 1]
        mine = correlate((species == s).astype(xp.float32), K,
                         mode="constant", cval=0.0)
        R = resource(g, L, Wsup, xp)
        hunt = float(g[6]) if n_fields > 6 else 0.0
        if hunt > 0:
            prey = xp.minimum(prey_all, cfg.prey_cap)   # насыщение: больше 2–3 жертв не съесть
            R = R + hunt * cfg.prey_gain * prey
        ok = (empty
              & (mine > g[2] * cfg.birth_own)     # свои дали достаточно голосов
              & (nb > g[2])                        # опоры хватает
              & (nb < g[2] + cfg.birth_window)     # но не давка
              & (R > g[3]))                        # ресурса хватает виду
        if hunt > 0:
            # хищнику вода снизу не обязательна, если рядом есть добыча
            ok = ok & ((Wsup > cfg.water_min) | (prey > 0.5))
        else:
            ok = ok & (Wsup > cfg.water_min)       # вода дотянулась
        branch = float(g[5]) if len(g) > 5 else 0.0
        ok = ok & (supported | (branch > 0) | (hunt > 0))
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
    for s in range(1, n_sp + 1):
        m = species == s
        g = cfg.genomes[s - 1]
        Rs = resource(g, L, W, xp)
        hunt = float(g[6]) if n_fields > 6 else 0.0
        if hunt > 0:
            Rs = Rs + hunt * cfg.prey_gain * xp.minimum(prey_all, cfg.prey_cap)
        Rlive = xp.where(m, Rs, Rlive)
        need = xp.where(m, g[3] * cfg.surv_factor, need)

    survive = alive & (nb < cfg.crowd_max) & (Rlive > need)

    # --- хищничество: жертву съедают с вероятностью ~ хищность соседей ---
    if has_hunt:
        pressure = eaten / xp.maximum(nb, 1e-6)      # доля хищности среди соседей
        p_eat = xp.clip(cfg.kill_rate * pressure, 0.0, 0.95)
        killed = plant & (rng.random(species.shape) < p_eat)
        survive = survive & ~killed

    # --- мутация при делении: пока просто смена вида ---
    mut = born & (rng.random(species.shape) < cfg.p_mutate)
    best_sp = xp.where(mut,
                       rng.integers(1, n_sp + 1, species.shape).astype(xp.int8),
                       best_sp)

    # --- растворение камня: клетка, стоящая на камне, превращает его в почву ---
    touch = xp.zeros_like(stone)
    touch[:, :, :-1] = alive[:, :, 1:]
    diss = stone & touch & (rng.random(species.shape) < cfg.p_dissolve)
    stone = stone & ~diss
    soil = soil | diss

    # --- выветривание: не даёт системе застыть в неподвижной точке ---
    shock = alive & (rng.random(species.shape) < cfg.p_shock)
    survive = survive & ~shock

    new_species = xp.where(born, best_sp,
                           xp.where(survive, species, xp.int8(0))).astype(xp.int8)

    state["species"], state["stone"], state["soil"] = new_species, stone, soil
    return [int((new_species == s).sum()) for s in range(1, n_sp + 1)]
