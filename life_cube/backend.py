"""Бэкенд: numpy на CPU либо cupy на GPU. Дальше по коду всё через xp.*"""

import numpy as np


def get_backend(use_gpu: bool):
    """Возвращает (xp, correlate, on_gpu).

    ВНИМАНИЕ: везде используется correlate, а не convolve. convolve зеркалит
    ядро и переворачивает анизотропию вверх ногами — на этом уже наступали.
    """
    if use_gpu:
        import cupy as xp
        from cupyx.scipy.ndimage import correlate
        return xp, correlate, True
    from scipy.ndimage import correlate
    return np, correlate, False


def to_cpu(a):
    """Снять массив с GPU, если он там."""
    return a.get() if hasattr(a, "get") else a
