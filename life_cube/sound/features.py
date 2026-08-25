"""Логика озвучивания: Snapshot -> SoundFrame.

Версия 1 (по разделу 8 концепции):
  * 64 гармоники = 64 полосы куба по высоте. Амплитуда полосы — её население
    (нормированное), шумовая примесь — энтропия распределения видов в полосе:
    однородный слой звучит чисто, перемешанный — шипит.
  * Голоса — крупнейшие организмы. Маленький организм — высокая гармоника,
    крупный — низкая. Громкость по размеру, панорама по x, вибрато по скорости
    движения центра масс. Голос живёт, пока жив организм (устойчивый id).
  * События: появление организма, дожившего до порога, и гибель голоса.

Всё это — параметры, а не законы; переписывается свободно. Контракт с
бэкендами — только SoundFrame.
"""

from dataclasses import dataclass, field, asdict

import numpy as np

from ..snapshot import Snapshot


@dataclass
class Voice:
    vid: int          # устойчивый id организма
    harmonic: int     # номер гармоники базовой частоты
    amp: float        # 0..1
    pan: float        # -1..1
    vib: float        # глубина вибрато 0..1 (скорость движения)
    age: int          # поколений с рождения
    species: int


@dataclass
class SoundFrame:
    gen: int
    harmonics: list          # 64 амплитуд 0..1
    noise: list              # 64 шумовых примесей 0..1
    voices: list = field(default_factory=list)   # [Voice]
    births: list = field(default_factory=list)   # id новых голосов
    deaths: list = field(default_factory=list)   # id погибших голосов
    activity: float = 0.0    # доля изменившихся клеток 0..1 — темп «дыхания»

    def to_dict(self):
        d = asdict(self)
        d["harmonics"] = [round(x, 4) for x in self.harmonics]
        d["noise"] = [round(x, 4) for x in self.noise]
        for v in d["voices"]:
            v["amp"] = round(v["amp"], 4); v["pan"] = round(v["pan"], 3); v["vib"] = round(v["vib"], 3)
        d["activity"] = round(self.activity, 4)
        return d


class SoundMapper:
    def __init__(self, n_bands=64, max_voices=12, min_voice_size=8,
                 amp_ref=None):
        self.n_bands = n_bands
        self.max_voices = max_voices
        self.min_voice_size = min_voice_size
        self.amp_ref = amp_ref            # опорное население полосы; None = авто
        self._prev_centers = {}
        self._prev_voice_ids = set()
        self._prev_cells = None
        self._peak = 1.0

    # --- гармоники ----------------------------------------------------------
    def bands(self, snap: Snapshot):
        n, nb = snap.n, self.n_bands
        z = snap.coords[:, 2].astype(np.int64)
        band = np.minimum(z * nb // max(n, 1), nb - 1)
        pop = np.bincount(band, minlength=nb).astype(np.float64)

        # энтропия видов в полосе: 0 — один вид, 1 — все поровну
        n_sp = max(int(snap.species.max()) if len(snap.species) else 1, 1)
        counts = np.zeros((nb, n_sp + 1))
        np.add.at(counts, (band, snap.species.astype(np.int64)), 1)
        counts = counts[:, 1:]
        tot = counts.sum(axis=1, keepdims=True)
        p = np.where(tot > 0, counts / np.maximum(tot, 1), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(p * np.log(np.where(p > 0, p, 1))).sum(axis=1)
        ent = ent / np.log(max(n_sp, 2))

        # нормировка: авто-пик с медленным спадом, чтобы громкость не прыгала
        if self.amp_ref:
            ref = self.amp_ref
        else:
            self._peak = max(pop.max(), self._peak * 0.995, 1.0)
            ref = self._peak
        amp = np.sqrt(np.clip(pop / ref, 0, 1))      # sqrt — ближе к восприятию
        return amp.tolist(), np.clip(ent, 0, 1).tolist()

    # --- голоса -------------------------------------------------------------
    def voices(self, snap: Snapshot):
        comps = [c for c in snap.components if c.size >= self.min_voice_size]
        comps = comps[: self.max_voices]
        biggest = max((c.size for c in comps), default=1)
        out, centers = [], {}
        for c in comps:
            # маленький -> высокая гармоника: размер по кубическому корню
            lin = c.size ** (1 / 3)
            h = int(np.clip(round(24 / max(lin, 1.0)), 1, self.n_bands))
            prev = self._prev_centers.get(c.cid)
            speed = float(np.linalg.norm(np.subtract(c.center, prev))) if prev else 0.0
            out.append(Voice(vid=c.cid, harmonic=h,
                             amp=float(np.sqrt(c.size / biggest)),
                             pan=float(2 * c.center[0] / max(snap.n - 1, 1) - 1),
                             vib=float(np.clip(speed / 2.0, 0, 1)),
                             age=int(snap.gen - c.born), species=c.species))
            centers[c.cid] = c.center
        ids = {v.vid for v in out}
        births = sorted(ids - self._prev_voice_ids)
        deaths = sorted(self._prev_voice_ids - ids)
        self._prev_centers = centers
        self._prev_voice_ids = ids
        return out, births, deaths

    # --- активность ---------------------------------------------------------
    def activity(self, snap: Snapshot):
        n = snap.n
        key = (snap.coords[:, 0].astype(np.int64) * n + snap.coords[:, 1]) * n + snap.coords[:, 2]
        cur = set(key.tolist()) if len(key) < 200_000 else None
        if cur is None or self._prev_cells is None:
            act = 0.0
        else:
            changed = len(cur ^ self._prev_cells)
            act = changed / max(len(cur | self._prev_cells), 1)
        self._prev_cells = cur
        return float(act)

    def map(self, snap: Snapshot) -> SoundFrame:
        amp, noise = self.bands(snap)
        vs, births, deaths = self.voices(snap)
        return SoundFrame(gen=snap.gen, harmonics=amp, noise=noise, voices=vs,
                          births=births, deaths=deaths, activity=self.activity(snap))
