"""
Трёхмерный клеточный автомат с экологией: камень, свет, вода, конкурирующие виды.

Это РАБОЧАЯ ПРОВЕРЕННАЯ БАЗА, а не псевдокод. Запуск по умолчанию (128^3, 200
поколений) даёт устойчивое сосуществование четырёх видов, разделённых по нишам:
корка в сухих низинах, столбы на влажном холме, потолок роста определяется
транспортом воды.

Ключевые механики, которые заставляют это работать (важно не сломать при правках):

1. АНИЗОТРОПНОЕ СОСЕДСТВО. Ядро 3x3x3 с разными весами: сосед снизу — опора
   (вес 1.6), сосед сверху мешает (0.45), граневые сильнее угловых. Именно это
   превращает бесформенное расползание в рост вверх.
   ВНИМАНИЕ: используется correlate, а не convolve. convolve зеркалит ядро и
   переворачивает анизотропию вверх ногами — на этом уже наступали.

2. ДВА РЕСУРСА С РАЗНЫХ СТОРОН. Свет идёт сверху и поглощается живыми клетками
   (затенение). Вода поднимается снизу из камня и теряет 10% на каждой клетке.
   Компромисс по высоте возникает сам, его не задавали явно.

3. КОНКУРЕНЦИЯ ПО ИЗБЫТКУ, А НЕ ПО АБСОЛЮТУ. Пустую клетку занимает вид с
   максимальным (R - порог_вида), а не с максимальным R. Без этого всегда
   побеждает один глобальный лидер и ниши не возникают.

4. НЕТ НИЖНЕГО ПОРОГА ВЫЖИВАНИЯ ПО СОСЕДЯМ. Одиночная клетка не умирает от
   одиночества — её убивает нехватка ресурса. С классическим порогом Конвея
   засев вымирает на первом же шаге.

5. ВОЗМУЩЕНИЯ. Редкая случайная гибель (выветривание) не даёт системе застыть
   в неподвижной точке.

Запуск:
    python cube_ecology.py                      # 128^3, 200 поколений, CPU
    python cube_ecology.py --n 256 --gens 800 --gpu
    python cube_ecology.py --n 64 --gens 120 --no-render

GPU: с флагом --gpu используется CuPy (cupy + cupyx.scipy.ndimage), код тот же.
На RTX 4080 256^3 идёт в реальном времени.
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Бэкенд: numpy на CPU либо cupy на GPU. Дальше по коду всё через xp.*
# ---------------------------------------------------------------------------

def get_backend(use_gpu: bool):
    if use_gpu:
        import cupy as xp
        from cupyx.scipy.ndimage import correlate
        return xp, correlate, True
    from scipy.ndimage import correlate
    return np, correlate, False


def to_cpu(a):
    """Снять массив с GPU, если он там."""
    return a.get() if hasattr(a, "get") else a


# ---------------------------------------------------------------------------
# Геном вида
# ---------------------------------------------------------------------------
# Пять чисел на вид. Сейчас это параметры (ручки), а не программа поведения.
# Переход к геному вида "если условие -> действие" — следующий этап, см. доку.
#
#   0  absorb    сколько света клетка поглощает (и, значит, отбрасывает тень)
#   1  up        тяга вверх: множитель в оценке привлекательности места
#   2  birth     порог взвешенной суммы соседей для рождения
#   3  need      минимальный ресурс, ниже которого вид не живёт
#   4  water     жадность к воде: 0 = чистый фотосинтетик, 1 = чистый водолюб

GENOME_FIELDS = ("absorb", "up", "birth", "need", "water")

DEFAULT_GENOMES = np.array([
    [0.25, 0.90, 1.60, 0.46, 0.30],   # 1 — корка: стелется, сухо, свет
    [0.60, 1.70, 1.85, 0.42, 0.25],   # 2 — башня: светолюбивая, растёт вверх
    [0.14, 0.80, 1.50, 0.24, 0.85],   # 3 — теневой: живёт влагой, терпит тень
    [0.40, 1.20, 1.80, 0.38, 0.45],   # 4 — универсал: середина по всем осям
], dtype=np.float32)

SPECIES_NAMES = ("корка", "башня", "теневой", "универсал")


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

    # события
    p_mutate: float = 0.0006      # смена вида при делении
    p_dissolve: float = 0.0016    # растворение камня клеткой, стоящей на нём
    p_shock: float = 0.0009       # случайная гибель (выветривание)

    seed_density: float = 0.006   # доля точек поверхности под споры
    genomes: np.ndarray = field(default_factory=lambda: DEFAULT_GENOMES.copy())


# ---------------------------------------------------------------------------
# Мир
# ---------------------------------------------------------------------------

def build_kernel(xp):
    """Анизотропное ядро соседства 3x3x3.

    Веса: снизу 1.6 (опора), сбоку 1.0, сверху 0.45; граневые x1.25,
    угловые x0.6. Центр нулевой — клетка себя не считает.
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


def build_world(cfg: Config, xp):
    """Рельеф камня, карта влажности подложки и стартовый засев спорами.

    Всё детерминировано от cfg.seed_world: один и тот же сид — один и тот же
    ландшафт, независимо от того, что потом произойдёт с мутациями.
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

    return (xp.asarray(stone), xp.asarray(wet), xp.asarray(species),
            relief)


# ---------------------------------------------------------------------------
# Поля ресурсов
# ---------------------------------------------------------------------------

def light_field(alive, absorb, xp):
    """Свет идёт сверху вниз. Каждая живая клетка забирает свою долю по геному,
    остаток достаётся тем, кто ниже. Это и есть затенение — механизм, из-за
    которого высота стоит того, чтобы за неё бороться."""
    n = alive.shape[0]
    L = xp.ones((n, n), dtype=xp.float32)
    out = xp.empty(alive.shape, dtype=xp.float32)
    for z in range(n - 1, -1, -1):
        out[:, :, z] = L
        L = xp.where(alive[:, :, z], L * (1.0 - absorb[:, :, z]), L)
    return out


def water_field(alive, stone, soil, wet, cfg, xp):
    """Вода поднимается из подложки по телу организма, теряя долю на каждом
    шаге. Разрыв в теле обрывает столб воды: висящие в воздухе структуры
    остаются без снабжения и гибнут. Почва (растворённый камень) держит воду
    немного лучше исходного камня."""
    n = alive.shape[0]
    W = xp.zeros(alive.shape, dtype=xp.float32)
    cur = xp.zeros((n, n), dtype=xp.float32)
    for z in range(n):
        s = stone[:, :, z]
        cur = xp.where(s, wet, cur * cfg.water_decay)
        cur = xp.where(soil[:, :, z], xp.maximum(cur, wet * 1.15), cur)
        cur = xp.where(alive[:, :, z] | s | soil[:, :, z], cur, 0.0)
        W[:, :, z] = cur
    return W


def resource(g, L, Wv, xp):
    """Сколько ресурса вид с геномом g получает в данной точке.
    Смесь света и воды в пропорции, заданной жадностью к воде."""
    return (1.0 - g[4]) * L * (0.5 + g[0]) + g[4] * Wv


# ---------------------------------------------------------------------------
# Шаг мира
# ---------------------------------------------------------------------------

def step(state, cfg, xp, correlate, gen):
    """Одно поколение. state — словарь с полями species/stone/soil/wet."""
    species, stone, soil, wet = (state["species"], state["stone"],
                                 state["soil"], state["wet"])
    K, G = state["kernel"], state["genomes"]
    n, n_sp = cfg.n, len(cfg.genomes)
    rng = state["rng"]

    alive = species > 0
    nb = correlate(alive.astype(xp.float32), K, mode="constant", cval=0.0)

    idx = xp.clip(species.astype(xp.int32) - 1, 0, n_sp - 1)
    absorb = xp.where(alive, G[idx, 0], 0.0).astype(xp.float32)

    L = light_field(alive, absorb, xp)
    W = water_field(alive, stone, soil, wet, cfg, xp)

    # вода, дотянувшаяся до пустой клетки снизу: сама пустая клетка воды не
    # содержит, поэтому для рождения смотрим на клетку под ней
    Wsup = xp.zeros_like(W)
    Wsup[:, :, 1:] = W[:, :, :-1] * cfg.water_decay

    # --- рождение: голосование соседей за то, чей вид займёт пустое место ---
    empty = (~alive) & (~stone)
    best_score = xp.full(species.shape, -1.0, dtype=xp.float32)
    best_sp = xp.zeros(species.shape, dtype=xp.int8)

    for s in range(1, n_sp + 1):
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
        # ВАЖНО: сравниваем избыток над собственным порогом, а не абсолют.
        # Иначе один вид выигрывает везде и ниши не возникают.
        score = xp.where(ok, mine * g[1] * (R - g[3]), -1.0).astype(xp.float32)
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
        Rlive = xp.where(m, resource(g, L, W, xp), Rlive)
        need = xp.where(m, g[3] * cfg.surv_factor, need)

    survive = alive & (nb < cfg.crowd_max) & (Rlive > need)

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


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------

def run(cfg: Config, use_gpu=False, verbose=True):
    xp, correlate, on_gpu = get_backend(use_gpu)
    stone, wet, species, relief = build_world(cfg, xp)

    state = {
        "species": species if on_gpu is False else xp.asarray(species),
        "stone": stone,
        "soil": xp.zeros((cfg.n,) * 3, dtype=bool),
        "wet": wet,
        "kernel": build_kernel(xp),
        "genomes": xp.asarray(cfg.genomes),
        # отдельный поток случайности для жизни — не смешивается с сидом мира
        "rng": xp.random.default_rng(cfg.seed_mut),
    }

    hist = []
    t0 = time.time()
    for gen in range(cfg.gens):
        pops = step(state, cfg, xp, correlate, gen)
        hist.append(pops)
        if verbose and (gen % max(1, cfg.gens // 8) == 0 or gen == cfg.gens - 1):
            print(f"поколение {gen:>5}  всего {sum(pops):>8}  "
                  f"по видам {pops}", flush=True)
    if verbose:
        dt = time.time() - t0
        print(f"готово за {dt:.1f} c  ({cfg.gens / dt:.1f} поколений/с, "
              f"{'GPU' if on_gpu else 'CPU'})")

    return {
        "species": to_cpu(state["species"]),
        "stone": to_cpu(state["stone"]),
        "soil": to_cpu(state["soil"]),
        "relief": relief,
        "hist": np.array(hist),
        "config": cfg,
    }


# ---------------------------------------------------------------------------
# Отрисовка
# ---------------------------------------------------------------------------

def render(result, path="cube_ecology.png"):
    """Четыре панели: вертикальный разрез, вид сверху, история популяций,
    атлас горизонтальных слоёв."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    sp, stone, soil = result["species"], result["stone"], result["soil"]
    hist = result["hist"]
    n = sp.shape[0]

    BG = "#0d0f14"
    COL = {1: "#2ec7b8", 2: "#f2c14e", 3: "#c05ce0", 4: "#f2683c"}
    STONE, SOIL = "#2a2d36", "#4a3b2f"
    cmap = ListedColormap([BG, STONE, SOIL] + [COL[i] for i in (1, 2, 3, 4)])
    norm = BoundaryNorm(np.arange(-0.5, 7.5), cmap.N)

    def code(s_sp, s_stone, s_soil):
        out = np.zeros(s_sp.shape, dtype=np.int8)
        out[s_stone] = 1
        out[s_soil] = 2
        for s in (1, 2, 3, 4):
            out[s_sp == s] = 2 + s
        return out

    # верхняя граница жизни — чтобы не рисовать пустоту
    live_z = np.where((sp > 0).any(axis=(0, 1)))[0]
    top_z = int(live_z.max()) + 4 if len(live_z) else n

    fig = plt.figure(figsize=(19, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1.5],
                          hspace=0.16, wspace=0.12,
                          left=0.03, right=0.985, top=0.86, bottom=0.05)

    def style(ax, title):
        ax.set_title(title, color="#c9d1d9", fontsize=12, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("#2a2f3a")

    # 1 — вертикальный разрез слоем толщиной 5 клеток
    ax = fig.add_subplot(gs[:, 0])
    y0 = n // 2
    slab = np.zeros((n, n), dtype=np.int8)
    for k in range(max(0, y0 - 2), min(n, y0 + 3)):
        m = (slab == 0) & (sp[:, k, :] > 0)
        slab[m] = sp[:, k, :][m]
    img = code(slab, stone[:, y0, :], soil[:, y0, :])
    ax.imshow(img.T[:top_z], origin="lower", cmap=cmap, norm=norm,
              interpolation="nearest", aspect="auto")
    style(ax, "Вертикальный разрез: камень, почва, рост вверх")
    ax.set_ylabel("высота (клетки)", color="#8b949e", fontsize=10)

    # 2 — вид сверху: верхний живой слой
    ax = fig.add_subplot(gs[0, 1])
    top = np.zeros((n, n), dtype=np.int8)
    for z in range(n - 1, -1, -1):
        m = (top == 0) & (sp[:, :, z] > 0)
        top[m] = sp[:, :, z][m]
    zeros = np.zeros((n, n), bool)
    ax.imshow(code(top, zeros, zeros).T, origin="lower", cmap=cmap, norm=norm,
              interpolation="nearest")
    style(ax, "Вид сверху: как виды поделили территорию")

    # 3 — история популяций
    ax = fig.add_subplot(gs[1, 1], facecolor="#12151c")
    for s in (1, 2, 3, 4):
        ax.plot(hist[:, s - 1], color=COL[s], lw=1.8)
    ax.set_title("Население по видам", color="#c9d1d9", fontsize=12, pad=8)
    ax.tick_params(colors="#6e7681", labelsize=8)
    ax.grid(alpha=0.12, color="#8b949e")
    ax.set_xlabel("поколение", color="#8b949e", fontsize=9)
    for s_ in ax.spines.values():
        s_.set_color("#2a2f3a")

    # 4 — атлас слоёв
    sub = gs[:, 2].subgridspec(4, 4, hspace=0.22, wspace=0.06)
    zs = np.linspace(1, top_z - 1, 16).astype(int)
    for i, z in enumerate(zs):
        a = fig.add_subplot(sub[i // 4, i % 4])
        a.imshow(code(sp[:, :, z], stone[:, :, z], soil[:, :, z]).T,
                 origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        a.set_title(f"z={z}", color="#8b949e", fontsize=8, pad=2)
        a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_color("#232833")

    total = int((sp > 0).sum())
    fig.suptitle(f"Куб {n}³ · {len(hist)} поколений · четыре вида на камне · "
                 f"{total} живых клеток", color="#e6edf3", fontsize=17, y=0.975)
    labels = [f"{i} · {SPECIES_NAMES[i - 1]}" for i in (1, 2, 3, 4)] + \
             ["камень", "почва (растворён)"]
    cols = [COL[i] for i in (1, 2, 3, 4)] + [STONE, SOIL]
    handles = [plt.Line2D([], [], marker="s", ls="", ms=10, mfc=c, mec="none")
               for c in cols]
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.912), labelcolor="#8b949e", fontsize=10)

    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n", type=int, default=128, help="размер куба")
    p.add_argument("--gens", type=int, default=200, help="число поколений")
    p.add_argument("--seed-world", type=int, default=20260825)
    p.add_argument("--seed-mut", type=int, default=20260825)
    p.add_argument("--gpu", action="store_true", help="считать на CuPy")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--out", default="cube_ecology.png")
    p.add_argument("--save-state", default=None, help="путь для .npz со снимком")
    a = p.parse_args()

    cfg = Config(n=a.n, gens=a.gens,
                 seed_world=a.seed_world, seed_mut=a.seed_mut)
    res = run(cfg, use_gpu=a.gpu)

    if a.save_state:
        np.savez_compressed(a.save_state, species=res["species"],
                            stone=res["stone"], soil=res["soil"],
                            relief=res["relief"], hist=res["hist"])
        print("состояние:", a.save_state)

    if not a.no_render:
        print("картинка:", render(res, a.out))


if __name__ == "__main__":
    main()
