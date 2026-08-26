"""Снимок мира для наблюдателей: разреженная упаковка, связные компоненты
(организмы) и их отслеживание во времени.

Этот модуль — граница между симуляцией и любым рендером. Рендер получает
Snapshot и ничего не знает ни про xp, ни про step().
"""

from dataclasses import dataclass, field

import numpy as np

from .backend import to_cpu

# 26-связность: организм — всё, что касается хотя бы углом
STRUCT26 = np.ones((3, 3, 3), dtype=bool)


@dataclass
class Component:
    cid: int          # устойчивый id организма (переживает кадры)
    species: int
    size: int
    center: tuple     # (x, y, z)
    zmin: int
    zmax: int
    born: int         # поколение, в котором организм впервые замечен


@dataclass
class Snapshot:
    gen: int
    n: int
    pops: list                      # население по видам
    coords: np.ndarray              # (k, 3) uint16 — живые клетки
    species: np.ndarray             # (k,) uint8
    labels: np.ndarray              # (k,) uint32 — устойчивый id организма
    components: list = field(default_factory=list)   # [Component]
    soil_coords: np.ndarray = None  # (m, 3) uint16 — растворённый камень


# ---------------------------------------------------------------------------
# Разреженная упаковка
# ---------------------------------------------------------------------------

def pack_cells(vol):
    """Плотный массив (n,n,n) -> (coords uint16 (k,3), values (k,)) для vol != 0.
    Порядок — C-порядок, так что упаковка детерминирована."""
    vol = to_cpu(vol)
    idx = np.flatnonzero(vol)
    n = vol.shape
    coords = np.stack(np.unravel_index(idx, n), axis=1).astype(np.uint16)
    return coords, vol.reshape(-1)[idx]


def unpack_cells(coords, values, n, dtype=None):
    """Обратно: разреженный список -> плотный массив (n,n,n)."""
    out = np.zeros((n, n, n), dtype=dtype or values.dtype)
    if len(coords):
        c = coords.astype(np.intp)
        out[c[:, 0], c[:, 1], c[:, 2]] = values
    return out


# ---------------------------------------------------------------------------
# Компоненты и отслеживание
# ---------------------------------------------------------------------------

def label_components(species, per_species=True):
    """Метки связных компонент по живым клеткам. Возвращает (labels, count).

    per_species=True: организм — связная область ОДНОГО вида (два вида,
    прижатые друг к другу, считаются двумя организмами).
    """
    from scipy import ndimage
    species = to_cpu(species)
    if not per_species:
        return ndimage.label(species > 0, structure=STRUCT26)
    labels = np.zeros(species.shape, dtype=np.int32)
    total = 0
    for s in np.unique(species[species > 0]):
        lab, k = ndimage.label(species == s, structure=STRUCT26)
        m = lab > 0
        labels[m] = lab[m] + total
        total += int(k)
    return labels, total


class Tracker:
    """Даёт организмам устойчивые id между кадрами.

    Новая компонента наследует id той старой, с которой у неё наибольшее
    пересечение по клеткам (и которая ещё никем не унаследована). Если
    пересечения нет — новый id. Так рост и сдвиг сохраняют идентичность,
    деление даёт одному потомку старый id, другому новый.
    """

    def __init__(self):
        self.next_id = 1
        self.prev = None          # плотный массив устойчивых id прошлого кадра
        self.born = {}            # cid -> поколение появления

    def assign(self, labels, gen):
        """labels: плотный int32 массив временных меток (0 = пусто).
        Возвращает плотный uint32 массив устойчивых id."""
        out = np.zeros(labels.shape, dtype=np.uint32)
        alive = labels > 0
        if not alive.any():
            self.prev = out
            return out
        new_ids = labels[alive]
        mapping = {}
        if self.prev is not None:
            old_ids = self.prev[alive]
            both = old_ids > 0
            if both.any():
                pair = (new_ids[both].astype(np.int64) << 32) | old_ids[both].astype(np.int64)
                keys, cnt = np.unique(pair, return_counts=True)
                order = np.argsort(-cnt, kind="stable")
                taken = set()
                for i in order:
                    nl, ol = int(keys[i] >> 32), int(keys[i] & 0xFFFFFFFF)
                    if nl in mapping or ol in taken:
                        continue
                    mapping[nl] = ol
                    taken.add(ol)
        uniq = np.unique(new_ids)
        lut = np.zeros(int(labels.max()) + 1, dtype=np.uint32)
        for nl in uniq:
            nl = int(nl)
            if nl in mapping:
                lut[nl] = mapping[nl]
            else:
                lut[nl] = self.next_id
                self.born[self.next_id] = gen
                self.next_id += 1
        out[alive] = lut[new_ids]
        self.prev = out
        return out


def describe_components(coords, species, ids, born, max_components=None):
    """Сводка по организмам, отсортированная по размеру."""
    if len(ids) == 0:
        return []
    order = np.argsort(ids, kind="stable")
    ids_s, coords_s, sp_s = ids[order], coords[order].astype(np.int64), species[order]
    uniq, start, counts = np.unique(ids_s, return_index=True, return_counts=True)
    comps = []
    for u, st, c in zip(uniq, start, counts):
        block = coords_s[st:st + c]
        cen = block.mean(axis=0)
        comps.append(Component(
            cid=int(u), species=int(sp_s[st]), size=int(c),
            center=(round(float(cen[0]), 1), round(float(cen[1]), 1), round(float(cen[2]), 1)),
            zmin=int(block[:, 2].min()), zmax=int(block[:, 2].max()),
            born=int(born.get(int(u), 0))))
    comps.sort(key=lambda k: -k.size)
    return comps[:max_components] if max_components else comps


def make_snapshot(state, gen, cfg, tracker=None, with_components=True,
                  max_components=200, n_species=None):
    """Собрать Snapshot из state симуляции (массивы могут быть на GPU)."""
    species = to_cpu(state["species"])
    n = species.shape[0]
    coords, sp = pack_cells(species)
    n_species = n_species or cfg.n_species
    counts = np.bincount(species.ravel().astype(np.int64), minlength=n_species + 1)
    pops = [int(c) for c in counts[1:n_species + 1]]
    soil_coords, _ = pack_cells(state["soil"])

    if with_components:
        lab, _ = label_components(species)
        ids_dense = tracker.assign(lab, gen) if tracker else lab.astype(np.uint32)
        c = coords.astype(np.intp)
        ids = ids_dense[c[:, 0], c[:, 1], c[:, 2]]
        comps = describe_components(coords, sp, ids,
                                    tracker.born if tracker else {}, max_components)
    else:
        ids = np.zeros(len(coords), dtype=np.uint32)
        comps = []

    return Snapshot(gen=gen, n=n, pops=pops, coords=coords,
                    species=sp.astype(np.uint8), labels=ids.astype(np.uint32),
                    components=comps, soil_coords=soil_coords)
