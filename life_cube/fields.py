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


def _spread_lateral(cur, body, steps, decay, xp):
    """Вода растекается по живому телу в пределах слоя: каждый шаг клетка
    берёт максимум из своего значения и (сосед × decay) по четырём сторонам.
    Достаёт на `steps` клеток вбок от источника — это и есть длина ветки,
    которую ствол способен напоить за одно поколение."""
    for _ in range(steps):
        nb = xp.zeros_like(cur)
        nb[1:, :] = xp.maximum(nb[1:, :], cur[:-1, :])
        nb[:-1, :] = xp.maximum(nb[:-1, :], cur[1:, :])
        nb[:, 1:] = xp.maximum(nb[:, 1:], cur[:, :-1])
        nb[:, :-1] = xp.maximum(nb[:, :-1], cur[:, 1:])
        cur = xp.where(body, xp.maximum(cur, nb * decay), cur)
    return cur


def water_field(alive, stone, soil, wet, cfg: Config, xp=np):
    """Вода поднимается из подложки по телу организма, теряя долю на каждом
    шаге вверх и (сильнее) на каждом шаге вбок. Разрыв в теле обрывает
    поток: висящие в воздухе структуры остаются без снабжения и гибнут.
    Почва (растворённый камень) держит воду немного лучше исходного камня."""
    n = alive.shape[0]
    W = xp.zeros(alive.shape, dtype=xp.float32)
    cur = xp.zeros((n, n), dtype=xp.float32)
    lateral = cfg.lateral_steps > 0 and cfg.lateral_decay > 0
    for z in range(n):
        s = stone[:, :, z]
        cur = xp.where(s, wet, cur * cfg.water_decay)
        cur = xp.where(soil[:, :, z], xp.maximum(cur, wet * 1.15), cur)
        body = alive[:, :, z]
        cur = xp.where(body | s | soil[:, :, z], cur, 0.0)
        if lateral:
            cur = _spread_lateral(cur, body, cfg.lateral_steps, cfg.lateral_decay, xp)
        W[:, :, z] = cur
    return W


def water_supply(W, cfg: Config, xp=np):
    """Вода, доступная ПУСТОЙ клетке для рождения: от клетки снизу (как
    раньше) либо от любого живого соседа сбоку (для ветвления), с потерей."""
    Wsup = xp.zeros_like(W)
    Wsup[:, :, 1:] = W[:, :, :-1] * cfg.water_decay
    if cfg.lateral_steps > 0 and cfg.lateral_decay > 0:
        d = cfg.lateral_decay
        Wsup[1:, :, :] = xp.maximum(Wsup[1:, :, :], W[:-1, :, :] * d)
        Wsup[:-1, :, :] = xp.maximum(Wsup[:-1, :, :], W[1:, :, :] * d)
        Wsup[:, 1:, :] = xp.maximum(Wsup[:, 1:, :], W[:, :-1, :] * d)
        Wsup[:, :-1, :] = xp.maximum(Wsup[:, :-1, :], W[:, 1:, :] * d)
    return Wsup


def resource(g, L, Wv, xp=np):
    """Сколько ресурса вид с геномом g получает в данной точке.
    Смесь света и воды в пропорции, заданной жадностью к воде."""
    return (1.0 - g[4]) * L * (0.5 + g[0]) + g[4] * Wv
