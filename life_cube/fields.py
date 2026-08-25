"""Поля ресурсов: свет сверху, вода снизу, и то, как вид их смешивает."""

import numpy as np

from .config import Config


def light_field(alive, absorb, xp=np):
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


def water_field(alive, stone, soil, wet, cfg: Config, xp=np):
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


def resource(g, L, Wv, xp=np):
    """Сколько ресурса вид с геномом g получает в данной точке.
    Смесь света и воды в пропорции, заданной жадностью к воде."""
    return (1.0 - g[4]) * L * (0.5 + g[0]) + g[4] * Wv
