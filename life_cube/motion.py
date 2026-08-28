"""Подвижные существа: движение по градиенту еды, охота, обмен веществ,
деление по энергии, старость.

Одна клетка = одно существо. Растения остаются сидячими (step.py), животные
живут по этим правилам:

  1. ЧУВСТВО. Поле привлекательности — плотность жертв, размазанная окном
     радиуса `sense`. Существо смотрит на 6 соседей и на «остаться».
  2. ОХОТА. Жертва гибнет с вероятностью 1−(1−hunt·(1−armor))^k, где k —
     сколько хищников к ней прижалось. Её энергия
     делится между напавшими с коэффициентом eat_efficiency.
  3. ДВИЖЕНИЕ. Шаг в пустую клетку с наибольшей привлекательностью (+шум).
     Столкновения разрешаются: в клетку въезжает один, у кого больше ключ.
  4. ТЯЖЕСТЬ. После шага существо падает, если под ним пусто, — поэтому оно
     ходит по поверхности, а не парит.
  5. ОБМЕН. Каждое поколение energy −= metabolism; при нуле — смерть, при
     возрасте больше lifespan — тоже.
  6. ДЕЛЕНИЕ. Если energy > repro и рядом есть место, появляется потомок,
     энергия делится пополам.
"""

import numpy as np

from .backend import sample_event
from .config import IDX

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def shift(a, d, xp, fill=0):
    """Сдвиг массива на вектор d с заполнением края.
    shift(a, d)[i] == a[i - d]."""
    out = xp.full_like(a, fill)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    for ax, k in enumerate(d):
        if k > 0:
            dst[ax] = slice(k, None); src[ax] = slice(None, -k)
        elif k < 0:
            dst[ax] = slice(None, k); src[ax] = slice(-k, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def smooth(field, radius, xp):
    """Запах: поле еды, размазанное гауссианой.

    Именно гауссиана, а не окно: у равномерного окна внутри него градиента
    НЕТ, и существо в пределах чутья не знает, куда идти. На этом уже
    наступали — зверь стоял рядом с едой и бродил наугад."""
    if radius <= 0:
        return field
    if xp.__name__ == "cupy":
        from cupyx.scipy.ndimage import gaussian_filter
    else:
        from scipy.ndimage import gaussian_filter
    sigma = max(float(radius) / 2.0, 0.5)
    return gaussian_filter(field.astype(xp.float32), sigma=sigma,
                           mode="constant", cval=0.0, truncate=3.0)


def animals_step(state, cfg, xp, correlate, rng):
    """Один шаг для всех подвижных видов. Меняет state на месте."""
    species, energy, age = state["species"], state["energy"], state["age"]
    stone, soil = state["stone"], state["soil"]
    G = cfg.genomes
    mobile = cfg.mobile_mask()
    if not mobile.any():
        return {}

    # перкуссия (см. life_cube.backend.sample_event): каждое дискретное
    # событие этого поколения — охота, гибель, деление — отдельный тип
    # удара на клиенте. Собираем по ходу дела, без лишних проходов по миру.
    events = {}

    i_hunt, i_tro = IDX["hunt"], IDX["trophic"]
    i_spd, i_sns = IDX["speed"], IDX["sense"]
    i_met, i_rep = IDX["metabolism"], IDX["repro"]
    i_life, i_arm = IDX["lifespan"], IDX["armor"]

    alive = species > 0
    solid = alive | stone
    trophic = xp.zeros(species.shape, dtype=xp.int32)
    armor = xp.zeros(species.shape, dtype=xp.float32)
    for s in range(1, cfg.n_species + 1):
        m = species == s
        trophic = xp.where(m, int(G[s - 1][i_tro]), trophic)
        armor = xp.where(m, float(G[s - 1][i_arm]), armor)

    # --- 1. охота: кто кого съел --------------------------------------------
    killed = xp.zeros(species.shape, dtype=bool)
    gain = xp.zeros(species.shape, dtype=xp.float32)
    for s in np.flatnonzero(mobile) + 1:
        g = G[s - 1]
        hunt, level = float(g[i_hunt]), int(g[i_tro])
        if hunt <= 0:
            continue
        me = (species == s)
        if not bool(me.any()):
            continue
        prey = alive & (trophic == level - 1) & ~killed
        # сколько хищников этого вида прижались к жертве и наоборот
        pred_around = xp.zeros(species.shape, dtype=xp.float32)
        prey_around = xp.zeros(species.shape, dtype=xp.float32)
        for d in DIRS:
            pred_around += shift(me.astype(xp.float32), d, xp)
            prey_around += shift(prey.astype(xp.float32), d, xp)
        # каждый прижавшийся хищник делает независимую попытку:
        # p = 1 − (1 − hunt·(1−броня))^(число хищников рядом)
        single = xp.clip(hunt * (1.0 - armor), 0.0, 0.99)
        p_kill = 1.0 - (1.0 - single) ** xp.minimum(pred_around, 6.0)
        newly = prey & (pred_around > 0) & (rng.random(species.shape) < p_kill)
        killed = killed | newly
        # энергия жертвы делится между прижавшимися хищниками
        share = xp.where(newly, energy / xp.maximum(pred_around, 1.0), 0.0)
        got = xp.zeros(species.shape, dtype=xp.float32)
        for d in DIRS:
            got += shift(share, tuple(-k for k in d), xp)
        gain = xp.where(me, gain + cfg.eat_efficiency * got, gain)

    ev = sample_event(killed, xp)
    if ev:
        events["kill"] = ev
    if bool(killed.any()):
        species = xp.where(killed, xp.int8(0), species)
        energy = xp.where(killed, xp.float32(0), energy)
        age = xp.where(killed, xp.int32(0), age)
        alive = species > 0
        solid = alive | stone

    energy = energy + gain

    # --- 2. движение ---------------------------------------------------------
    empty = ~solid
    for s in np.flatnonzero(mobile) + 1:
        g = G[s - 1]
        steps = int(g[i_spd])
        me = species == s
        if steps <= 0 or not bool(me.any()):
            continue
        level, sense = int(g[i_tro]), int(g[i_sns])
        prey_now = alive & (trophic == level - 1)
        # рядом с добычей существо остаётся: незачем уходить от еды
        near_prey = xp.zeros(species.shape, dtype=bool)
        for d in DIRS:
            near_prey = near_prey | shift(prey_now, d, xp, fill=False)
        prey_field = prey_now.astype(xp.float32)
        # запах еды, поделённый на тесноту своих: сторонимся сородичей, но
        # поле остаётся НЕОТРИЦАТЕЛЬНЫМ — иначе отношение ниже меняет знак
        # и существо застывает на месте (наступали)
        attract = smooth(prey_field, sense, xp) / (
            1.0 + 2.0 * smooth(me.astype(xp.float32), 1, xp))
        eps = xp.float32(1e-4)
        for _ in range(steps):
            me = species == s
            if not bool(me.any()):
                break
            empty = ~(species > 0) & ~stone
            key = rng.random(species.shape).astype(xp.float32) + 1.0
            # сравниваем ОТНОШЕНИЕ привлекательности цели к своей клетке, а не
            # разность: иначе шум (доли единицы) забивает градиент запаха,
            # который на расстоянии чутья исчисляется сотыми
            here = attract + eps
            best = xp.where(me, xp.float32(1.0), -xp.inf)   # «остаться» = 1.0
            best_dir = xp.full(species.shape, -1, dtype=xp.int32)
            noise = cfg.move_noise
            for di, d in enumerate(DIRS):
                back = tuple(-k for k in d)
                val = (shift(attract, back, xp, fill=0.0) + eps) / here
                free = shift(empty, back, xp, fill=False)
                val = val * (1.0 + noise * (2.0 * rng.random(species.shape).astype(xp.float32) - 1.0))
                cand = me & ~near_prey & free & (val > best)
                best = xp.where(cand, val, best)
                best_dir = xp.where(cand, di, best_dir)
            moving = best_dir >= 0
            if not bool(moving.any()):
                break
            # разрешение столкновений: в клетку въедет тот, у кого ключ больше.
            # внутри одного направления коллизий нет (это сдвиг), поэтому
            # достаточно сравнить победителей по шести направлениям
            claims = []
            winner = xp.zeros(species.shape, dtype=xp.float32)
            for di, d in enumerate(DIRS):
                k = xp.where(moving & (best_dir == di), key, 0.0)
                at_target = shift(k, d, xp)
                claims.append(at_target)
                winner = xp.maximum(winner, at_target)
            new_species, new_energy, new_age = species, energy, age
            vacated = xp.zeros(species.shape, dtype=bool)
            for di, d in enumerate(DIRS):
                took = (claims[di] > 0) & (claims[di] == winner)
                if not bool(took.any()):
                    continue
                back = tuple(-k for k in d)
                src = shift(took, back, xp, fill=False)
                new_species = xp.where(took, xp.int8(int(s)), new_species)
                new_energy = xp.where(took, shift(energy, d, xp), new_energy)
                new_age = xp.where(took, shift(age, d, xp), new_age)
                vacated = vacated | src
            species = xp.where(vacated, xp.int8(0), new_species)
            energy = xp.where(vacated, xp.float32(0), new_energy)
            age = xp.where(vacated, xp.int32(0), new_age)

    # --- 3. тяжесть: существо падает, если под ним пусто ---------------------
    anim = xp.zeros(species.shape, dtype=bool)
    for s in np.flatnonzero(mobile) + 1:
        anim = anim | (species == s)
    if bool(anim.any()):
        down = (0, 0, -1)
        support = shift((species > 0) | stone | soil, (0, 0, 1), xp, fill=True)
        falling = anim & ~support
        if bool(falling.any()):
            tgt = shift(falling, down, xp, fill=False)
            species = xp.where(tgt, shift(species, down, xp), species)
            energy = xp.where(tgt, shift(energy, down, xp), energy)
            age = xp.where(tgt, shift(age, down, xp), age)
            species = xp.where(falling, xp.int8(0), species)
            energy = xp.where(falling, xp.float32(0), energy)
            age = xp.where(falling, xp.int32(0), age)

    # --- 4. обмен веществ, возраст, смерть -----------------------------------
    met = xp.zeros(species.shape, dtype=xp.float32)
    lifespan = xp.zeros(species.shape, dtype=xp.float32)
    anim = xp.zeros(species.shape, dtype=bool)
    for s in np.flatnonzero(mobile) + 1:
        m = species == s
        anim = anim | m
        met = xp.where(m, float(G[s - 1][i_met]), met)
        lifespan = xp.where(m, float(G[s - 1][i_life]), lifespan)
    energy = xp.where(anim, energy - met, energy)
    age = xp.where(anim, age + 1, age)
    dead = anim & ((energy <= 0) | ((lifespan > 0) & (age.astype(xp.float32) > lifespan)))
    ev = sample_event(dead, xp)
    if ev:
        events["starve"] = ev
    species = xp.where(dead, xp.int8(0), species)
    energy = xp.where(dead, xp.float32(0), energy)
    age = xp.where(dead, xp.int32(0), age)

    # --- 5. деление по энергии ----------------------------------------------
    newborn = xp.zeros(species.shape, dtype=bool)      # для перкуссии
    for s in np.flatnonzero(mobile) + 1:
        g = G[s - 1]
        repro = float(g[i_rep])
        if repro <= 0:
            continue
        me = (species == s) & (energy > repro)
        if not bool(me.any()):
            continue
        empty = ~(species > 0) & ~stone
        key = rng.random(species.shape).astype(xp.float32) + 1.0
        placed = xp.zeros(species.shape, dtype=bool)
        winner = xp.zeros(species.shape, dtype=xp.float32)
        claims = []
        for d in DIRS:
            back = tuple(-k for k in d)
            free = shift(empty, back, xp, fill=False)
            k = xp.where(me & free, key, 0.0)
            at_target = shift(k, d, xp)
            claims.append(at_target)
            winner = xp.maximum(winner, at_target)
        for di, d in enumerate(DIRS):
            took = (claims[di] > 0) & (claims[di] == winner)
            if not bool(took.any()):
                continue
            back = tuple(-k for k in d)
            parent = shift(took, back, xp, fill=False) & me & ~placed
            took = shift(parent, d, xp, fill=False)
            half = xp.where(parent, energy * 0.5, 0.0)
            species = xp.where(took, xp.int8(int(s)), species)
            energy = xp.where(took, shift(half, d, xp), energy)
            age = xp.where(took, xp.int32(0), age)
            energy = xp.where(parent, energy * 0.5, energy)
            placed = placed | parent
            newborn = newborn | took

    ev = sample_event(newborn, xp)
    if ev:
        events["birth_animal"] = ev

    state["species"], state["energy"], state["age"] = species, energy, age
    return events
