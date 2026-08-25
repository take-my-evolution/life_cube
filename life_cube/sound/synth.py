"""Оффлайн аддитивный синтез: список SoundFrame -> волна float32 -> WAV.

Та же модель, что в браузере (viewers/web/static/index.html, блок Audio):
    гармоники   sum_h A_h(t) * sin(2π h f0 t)   + шум, взвешенный noise_h
    голоса      sin(2π h_v f0 t + vib) * amp_v, панорама по pan
Амплитуды интерполируются линейно между кадрами. Используется как оракул в
тестах и как способ «послушать прогон» без браузера.
"""

import numpy as np

from .features import SoundFrame, SoundMapper


def render(frames, sr=44100, seconds_per_frame=0.1, base_hz=55.0,
           harmonics_gain=0.6, voices_gain=0.5, noise_gain=0.25, stereo=True):
    if not frames:
        return np.zeros((0, 2 if stereo else 1), np.float32)
    nb = len(frames[0].harmonics)
    spf = int(sr * seconds_per_frame)
    total = spf * len(frames)
    t = np.arange(total) / sr
    rng = np.random.default_rng(0)

    # огибающие гармоник: ступеньки, сглаженные линейной интерполяцией
    A = np.array([f.harmonics for f in frames])           # (F, nb)
    N = np.array([f.noise for f in frames])
    ft = np.arange(len(frames)) * spf
    def envelope(col):
        return np.interp(np.arange(total), ft, col)
    out = np.zeros(total)
    freqs = base_hz * np.arange(1, nb + 1)
    nyq = sr / 2
    for h in range(nb):
        if freqs[h] >= nyq or A[:, h].max() <= 0:
            continue
        env = envelope(A[:, h]) / (h + 1) ** 0.5      # верхние гармоники тише
        out += harmonics_gain * env * np.sin(2 * np.pi * freqs[h] * t)
    # шум: белый, окрашенный средней энтропией (по всем полосам с весом амплитуды)
    ent = envelope((A * N).sum(axis=1) / np.maximum(A.sum(axis=1), 1e-9))
    out += noise_gain * ent * rng.standard_normal(total) * 0.3

    left, right = out.copy(), out.copy()
    # голоса: каждый живёт пока присутствует в кадрах
    per_voice = {}
    for i, f in enumerate(frames):
        for v in f.voices:
            per_voice.setdefault(v.vid, []).append((i, v))
    for vid, items in per_voice.items():
        idx = np.array([i for i, _ in items])
        amp = np.zeros(len(frames)); amp[idx] = [v.amp for _, v in items]
        h = items[0][1].harmonic
        pan = float(np.mean([v.pan for _, v in items]))
        vib = np.zeros(len(frames)); vib[idx] = [v.vib for _, v in items]
        f_v = base_hz * h
        if f_v >= nyq:
            continue
        env = envelope(amp); vb = envelope(vib)
        phase = 2 * np.pi * f_v * t + 0.02 * f_v * vb * np.sin(2 * np.pi * 5.0 * t) / 5.0
        sig = voices_gain * env * np.sin(phase) / max(h, 1) ** 0.3
        left += sig * (1 - pan) / 2 * 1.4
        right += sig * (1 + pan) / 2 * 1.4
    wave = np.stack([left, right], axis=1) if stereo else out[:, None]
    peak = np.abs(wave).max()
    if peak > 0.98:
        wave = wave / peak * 0.98
    return wave.astype(np.float32)


def write_wav(path, wave, sr=44100):
    from scipy.io import wavfile
    wavfile.write(path, sr, (np.clip(wave, -1, 1) * 32767).astype(np.int16))
    return path


def sonify_run(cfg, use_gpu=False, seconds_per_frame=0.1, base_hz=55.0, **kw):
    """Прогнать мир и вернуть (frames, wave): звук всей истории."""
    from ..backend import get_backend
    from ..sim import init_state
    from ..snapshot import Tracker, make_snapshot
    from ..step import step
    xp, corr, _ = get_backend(use_gpu)
    state, _ = init_state(cfg, xp)
    tracker, mapper, frames = Tracker(), SoundMapper(), []
    for gen in range(cfg.gens):
        step(state, cfg, xp, corr, gen)
        frames.append(mapper.map(make_snapshot(state, gen + 1, cfg, tracker)))
    return frames, render(frames, seconds_per_frame=seconds_per_frame, base_hz=base_hz, **kw)

