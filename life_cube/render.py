"""Отрисовка: разрез, вид сверху, история популяций, атлас слоёв."""

import numpy as np

from .config import SPECIES_NAMES


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
