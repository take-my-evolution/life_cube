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

    # --- дождь: подложка сохнет каждое поколение и пополняется каплями ---
    # `wet` больше не застывший навсегда узор ниш (см. world.build_world) — он
    # эволюционирует сам: без дождя (rain_rate=0) высыхает до нуля за
    # ~ln(0.01)/ln(rain_decay) поколений, и всё, что кормится водой (высокий
    # ген water — например мох), со временем усыхает вместе с ним. Дождь
    # регулируется тремя ручками (rain_rate/rain_amount/rain_decay), а не
    # просто ещё одним фиксированным полем: иначе плотность жизни снова
    # упёрлась бы в потолок и там и осталась.
    # Свой поток случайности (не общий `rng`): иначе одно новое
    # rng.random() здесь сдвигало бы вообще ВСЮ последующую случайность
    # поколения — кто родился, куда пошёл хищник — заново на каждый чих
    # ручки дождя (наступали: ловилось только по неожиданно упавшим тестам).
    rain_rng = state.get("rng_rain")
    if rain_rng is None:
        rain_rng = xp.random.default_rng(cfg.seed_mut ^ 0x7a17a170)
    drop = rain_rng.random(wet.shape) < cfg.rain_rate
    wet = wet * cfg.rain_decay + xp.where(drop, xp.float32(cfg.rain_amount), xp.float32(0.0))
    state["wet"] = wet
    state["rng_rain"] = rain_rng

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

    # --- биомасса растений: клетка копит энергию, деление стоит по массе ---
    # Раньше растения делились КАЖДОЕ поколение, если рядом было место: все
    # шесть видов удваивали биомассу за одно поколение, мох рос ровно так же
    # быстро, как дерево, а гены metabolism/repro/lifespan у растений лежали
    # мёртвым грузом (работали только у животных в motion.py).
    #
    # Теперь у растений та же экономика, что у животных: клетка копит
    # `resource - metabolism`, а новая клетка стоит `repro * mass *
    # growth_cost`. Масса — это плотность ткани: клетка дерева тяжелее клетки
    # мха, поэтому и строится дольше, и весит больше в биомассе, и кормит
    # травоядное сытнее. Скорость роста становится ЭМЕРГЕНТНОЙ: мох во
    # влажной низине растёт быстрее, чем на сухом камне, сам по себе.
    #
    # Совместимость: repro=0 выключает накопление (делится сразу, как раньше),
    # lifespan=0 — не стареет, metabolism=0 — ничего не тратит, mass=0
    # читается как 1. Старые геномы и legacy-сравнение продолжают работать.
    energy = state.get("energy")
    i_met, i_rep, i_life = IDX["metabolism"], IDX["repro"], IDX["lifespan"]
    i_mass = IDX.get("mass")

    def gene_of(g, i, default=0.0):
        return float(g[i]) if (i is not None and i < len(g)) else default

    def cell_mass(g):
        m = gene_of(g, i_mass, 0.0)
        return m if m > 0 else 1.0          # 0 в геноме = «обычная» клетка

    def birth_cost(g):
        """Во что обходится новая клетка этого вида."""
        return gene_of(g, i_rep) * cell_mass(g) * float(getattr(cfg, "growth_cost", 6.0))

    # ресурс живых клеток и порог выживания — нужны и энергии, и смертям ниже
    Rlive = xp.zeros(species.shape, dtype=xp.float32)
    need = xp.zeros(species.shape, dtype=xp.float32)
    for s in plant_ids:
        m = species == s
        g = cfg.genomes[s - 1]
        Rlive = xp.where(m, resource(g, L, W, xp), Rlive)
        need = xp.where(m, g[3] * cfg.surv_factor, need)

    # экономика включается только если её кто-то использует: с нулевыми
    # repro/metabolism/lifespan (старые геномы, legacy-тесты) не тратим ни
    # одной лишней операции на куб
    econ = any(birth_cost(cfg.genomes[s - 1]) > 0 for s in plant_ids)
    dies = any(gene_of(cfg.genomes[s - 1], i_met) > 0
               or gene_of(cfg.genomes[s - 1], i_life) > 0 for s in plant_ids)
    grow = energy is not None and bool(plant_ids) and (econ or dies)
    if grow:
        metab = xp.zeros(species.shape, dtype=xp.float32)
        cap = xp.full(species.shape, 1e9, dtype=xp.float32)
        for s in plant_ids:
            g = cfg.genomes[s - 1]
            m = species == s
            metab = xp.where(m, gene_of(g, i_met), metab)
            cost_s = birth_cost(g)
            if cost_s > 0:
                cap = xp.where(m, cost_s * float(getattr(cfg, "energy_cap", 2.5)), cap)
        # прирост биомассы за поколение: что добыла минус что потратила
        energy = xp.where(alive, xp.minimum(energy + Rlive - metab, cap), energy)

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
        cost_s = birth_cost(g) if grow else 0.0
        if cost_s > 0:
            # соседи своего вида должны НАКОПИТЬ на постройку. Смотрим их
            # СРЕДНЮЮ энергию, а не сумму: по сумме густой ковёр строил бы
            # новую клетку каждое поколение просто потому, что соседей много,
            # и вся экономика опять свелась бы к «делимся всегда»
            emine = correlate(xp.where(species == s, energy, 0.0).astype(xp.float32), K,
                              mode="constant", cval=0.0)
            ok = ok & ((emine / xp.maximum(mine, 1e-6)) > cost_s)
        branch = float(g[5]) if len(g) > 5 else 0.0
        ok = ok & (supported | (branch > 0))
        # ВАЖНО: сравниваем избыток над собственным порогом, а не абсолют.
        weight = xp.where(supported, g[1], branch * L)
        score = xp.where(ok, mine * weight * (R - g[3]), -1.0).astype(xp.float32)
        upd = score > best_score
        best_score = xp.where(upd, score, best_score)
        best_sp = xp.where(upd, xp.int8(s), best_sp)

    born = best_sp > 0

    # --- за постройку платят соседи-родители --------------------------------
    # Энергия на новую клетку списывается с тех, кто её вырастил. Списываем
    # тем же ядром соседства, что и голосовали, нормируя на его вес: если
    # часть соседей — пустота или чужой вид, спишется меньше полной цены.
    # Это приближение (точный дележ стоил бы ещё одной свёртки на вид), но
    # знак и порядок верны: густой и сытый ковёр растёт, разреженный и
    # голодный — нет, и после рывка роста вид «выдыхается».
    if grow and econ and bool(born.any()):
        ksum = float(K.sum()) if hasattr(K, "sum") else 1.0
        for s in plant_ids:
            g = cfg.genomes[s - 1]
            cost_s = birth_cost(g)
            born_s = born & (best_sp == xp.int8(s))
            if cost_s <= 0 or not bool(born_s.any()):
                continue
            paid = correlate(xp.where(born_s, xp.float32(cost_s), 0.0).astype(xp.float32),
                             K, mode="constant", cval=0.0) / max(ksum, 1e-6)
            energy = xp.where(species == s, xp.maximum(energy - paid, 0.0), energy)

    survive = alive & (nb < cfg.crowd_max) & (Rlive > need)

    # --- истощение и старость -----------------------------------------------
    # Гены, которые у растений до сих пор лежали мёртвым грузом: metabolism
    # (нечем платить за обмен — клетка гибнет) и lifespan (0 = не стареет).
    if grow and dies:
        spends = xp.zeros(species.shape, dtype=bool)
        old = xp.zeros(species.shape, dtype=bool)
        age_cur = state.get("age")
        for s in plant_ids:
            g = cfg.genomes[s - 1]
            m = species == s
            if gene_of(g, i_met) > 0:
                spends = spends | m
            life = gene_of(g, i_life)
            if life > 0 and age_cur is not None:
                old = old | (m & (age_cur.astype(xp.float32) > life))
        survive = survive & ~(spends & (energy <= 0)) & ~old

    # --- связь с подложкой: оторванная ткань гибнет ---
    # Вода доходит только по непрерывному телу от камня/почвы, поэтому W == 0
    # означает «нет связи с землёй». Раньше такие клетки жили на одном свете
    # и висели в воздухе (кроны деревьев после гибели ствола) — баг.
    # Новорождённой клетке (age == 0) даётся одно поколение: вода до неё
    # дотечёт на следующем проходе, иначе кромка роста мерцала бы.
    if cfg.require_substrate:
        age = state.get("age")
        newborn = (age == 0) if age is not None else False
        survive = survive & ((W > 0) | newborn)

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

    if energy is not None:
        # Новорождённая клетка стоит столько, сколько весит: её стартовая
        # энергия — это и есть вложенная в неё биомасса. Отсюда же берётся
        # пищевая ценность для травоядного (motion.py ест энергию жертвы):
        # раньше там стояла КОНСТАНТА plant_energy, и клетка мха кормила
        # ровно так же, как клетка дерева. У видов без экономики (repro=0)
        # поведение прежнее — константа.
        if grow and econ:
            newborn_e = xp.full(species.shape, xp.float32(cfg.plant_energy), dtype=xp.float32)
            for s in plant_ids:
                g = cfg.genomes[s - 1]
                cost_s = birth_cost(g)
                if cost_s > 0:
                    newborn_e = xp.where(best_sp == xp.int8(s), xp.float32(cost_s * 0.5), newborn_e)
            energy = xp.where(born, newborn_e, energy)
        else:
            energy = xp.where(born, xp.float32(cfg.plant_energy), energy)
        energy = xp.where(new_species == 0, xp.float32(0), energy)
        state["energy"] = energy
    age = state.get("age")
    if age is not None:
        # возраст растений: новорождённые 0, остальные +1 (животных считает motion)
        age = xp.where(born, xp.int32(0), xp.where(alive & (new_species > 0), age + 1, age))
        age = xp.where(new_species == 0, xp.int32(0), age)
        state["age"] = age

    state["species"], state["stone"], state["soil"] = new_species, stone, soil

    if mobile.any():
        animals_step(state, cfg, xp, correlate, rng)

    sp = state["species"]
    return [int((sp == s).sum()) for s in range(1, n_sp + 1)]
