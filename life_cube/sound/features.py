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
    base_hz: float = 55.0                       # основной тон — из населения
    voices: list = field(default_factory=list)   # [Voice]
    births: list = field(default_factory=list)   # id новых голосов
    deaths: list = field(default_factory=list)   # id погибших голосов
    activity: float = 0.0    # доля изменившихся клеток 0..1 — темп «дыхания»
    # доминирующий вид в полосе (1..N, 0 — полоса пуста). Дёшево считается
    # заодно с bands() (argmax по уже собранной матрице counts), используется
    # клиентом для режима "Кристалл": каждая полоса красится по своему виду
    # вместо плоского целочисленного обертона.
    band_species: list = field(default_factory=list)
    # Перкуссия — дискретные события поколения (охота, гибель, деление...),
    # {kind: {"n": сколько клеток затронуто, "pan": [-1..1, до 32]}}.
    # ЕДИНСТВЕННОЕ поле SoundFrame, которое заполняет не SoundMapper.map(), а
    # сам WebViewer (см. server.py: WebViewer._merge_events/broadcaster) —
    # потому что события нужно копить МЕЖДУ снимками (иначе часть терялась
    # бы, когда симуляция быстрее, чем реально уходящие клиенту кадры), а
    # SoundMapper видит только один snap за раз и такой памяти не имеет.
    percussion: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["harmonics"] = [round(x, 4) for x in self.harmonics]
        d["noise"] = [round(x, 4) for x in self.noise]
        for v in d["voices"]:
            v["amp"] = round(v["amp"], 4); v["pan"] = round(v["pan"], 3); v["vib"] = round(v["vib"], 3)
        d["activity"] = round(self.activity, 4)
        d["base_hz"] = round(self.base_hz, 2)
        return d


class SoundMapper:
    # пентатоника: смена доминирующего вида слышна как смена тональности,
    # а не как случайный сдвиг
    PENTA = (0, 3, 5, 7, 10, 12, 15, 17, 19, 22)

    def __init__(self, n_bands=64, max_voices=12, min_voice_size=8,
                 amp_ref=None, base_min=36.0):
        self.n_bands = n_bands
        self.max_voices = max_voices
        self.min_voice_size = min_voice_size
        self.amp_ref = amp_ref            # опорное население полосы; None = авто
        self._prev_centers = {}
        self._prev_voice_ids = set()
        self._prev_cells = None
        self._peak = 1.0
        self.base_min = float(base_min)
        self._base_hz = float(base_min) * 1.5
        self._dominant = None

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

        # доминирующий вид полосы: argmax по уже собранной матрице counts,
        # почти бесплатно (n_bands x n_species уже посчитаны выше). 0 — полоса
        # пуста (tot == 0), иначе 1..n_sp.
        dom = np.where(tot.squeeze(-1) > 0, counts.argmax(axis=1) + 1, 0)

        # нормировка: авто-пик с медленным спадом, чтобы громкость не прыгала
        if self.amp_ref:
            ref = self.amp_ref
        else:
            self._peak = max(pop.max(), self._peak * 0.995, 1.0)
            ref = self._peak
        amp = np.sqrt(np.clip(pop / ref, 0, 1))      # sqrt — ближе к восприятию
        return amp.tolist(), np.clip(ent, 0, 1).tolist(), dom.astype(int).tolist()

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

    def species_voices(self, snap: Snapshot):
        """Голоса из ВИДОВ: работают и там, где организмы не размечаются
        (большие миры). Раньше в таких мирах голосов не было вовсе и звук
        оставался ровным фоном — на это жаловались."""
        pops = list(snap.pops)
        total = sum(pops)
        if total <= 0:
            return [], [], []
        order = [i for i in np.argsort(pops)[::-1] if pops[i] > 0][: self.max_voices]
        sp = snap.species.astype(np.int64)
        n = max(snap.n - 1, 1)
        out, centers = [], {}
        for rank, i in enumerate(order):
            sid = int(i) + 1
            m = sp == sid
            cnt = int(m.sum())
            if cnt == 0:
                continue
            cx = float(snap.coords[m][:, 0].mean()) if cnt else 0.0
            cz = float(snap.coords[m][:, 2].mean()) if cnt else 0.0
            # высота голоса: чем выше вид живёт и чем он малочисленнее, тем выше
            share = pops[i] / total
            h = int(np.clip(round(1 + rank + 6.0 * (cz / n)), 1, self.n_bands))
            prev = self._prev_centers.get(sid)
            speed = abs(cx - prev[0]) + abs(cz - prev[1]) if prev else 0.0
            out.append(Voice(vid=sid, harmonic=h, amp=float(np.sqrt(share)),
                             pan=float(2 * cx / n - 1), vib=float(np.clip(speed, 0, 1)),
                             age=0, species=sid))
            centers[sid] = (cx, cz)
        ids = {v.vid for v in out}
        births = sorted(ids - self._prev_voice_ids)
        deaths = sorted(self._prev_voice_ids - ids)
        self._prev_centers = centers
        self._prev_voice_ids = ids
        return out, births, deaths

    def base_from_world(self, snap: Snapshot):
        """Основной тон вычисляется из населения: тональность задаёт
        доминирующий вид, октаву — насколько густо заселён мир."""
        pops = list(snap.pops)
        total = sum(pops)
        if total <= 0:
            return self._base_hz
        dom = int(np.argmax(pops))
        degree = self.PENTA[dom % len(self.PENTA)]
        octave = 0 if total < 2000 else (1 if total < 40000 else 2)
        target = self.base_min * (2.0 ** octave) * (2.0 ** (degree / 12.0))
        # плавный переход: скачок тона резал бы слух
        self._base_hz += 0.15 * (target - self._base_hz)
        self._dominant = dom + 1
        return self._base_hz

    def map(self, snap: Snapshot) -> SoundFrame:
        amp, noise, band_species = self.bands(snap)
        if snap.components:
            vs, births, deaths = self.voices(snap)
        else:
            vs, births, deaths = self.species_voices(snap)
        return SoundFrame(gen=snap.gen, harmonics=amp, noise=noise, voices=vs,
                          births=births, deaths=deaths, activity=self.activity(snap),
                          base_hz=self.base_from_world(snap), band_species=band_species)
