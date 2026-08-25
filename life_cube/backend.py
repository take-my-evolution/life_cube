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


def max_filter_for(xp):
    """4-связный max-фильтр 3x3 для слоя. Одна операция вместо четырёх
    сдвигов — на GPU это разница в числе запусков ядер, а их тут узкое место."""
    if xp.__name__ == "cupy":
        from cupyx.scipy.ndimage import maximum_filter
    else:
        from scipy.ndimage import maximum_filter
    fp = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    def f(a):
        return maximum_filter(a, footprint=fp, mode="constant", cval=0.0)
    return f


def to_cpu(a):
    """Снять массив с GPU, если он там."""
    return a.get() if hasattr(a, "get") else a
