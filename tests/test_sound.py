import numpy as np
import pytest

from life_cube import Config
from life_cube.snapshot import Snapshot, Component
from life_cube.sound import SoundMapper, SoundFrame
from life_cube.sound.synth import render, sonify_run, write_wav


def snap_from(cells, n=64, gen=0, comps=()):
    """cells: список (x,y,z,species,label)."""
    a = np.array(cells, dtype=np.int64).reshape(-1, 5)
    return Snapshot(gen=gen, n=n, pops=[0, 0, 0, 0],
                    coords=a[:, :3].astype(np.uint16), species=a[:, 3].astype(np.uint8),
                    labels=a[:, 4].astype(np.uint32), components=list(comps),
                    soil_coords=np.zeros((0, 3), np.uint16))


def test_bands_population_and_entropy():
    m = SoundMapper(n_bands=64, amp_ref=10)
    # полоса 0 (z=0): 10 клеток одного вида; полоса 32 (z=32): 4 клетки, два вида поровну
    cells = [(i, 0, 0, 1, 1) for i in range(10)] + [(i, 0, 32, 1 + i % 2, 2) for i in range(4)]
    sf = m.map(snap_from(cells))
    assert len(sf.harmonics) == 64 and len(sf.noise) == 64
    assert sf.harmonics[0] == pytest.approx(1.0)                # 10/10 -> sqrt(1)
    assert sf.harmonics[32] == pytest.approx(np.sqrt(0.4))      # 4/10
    assert sf.harmonics[1] == 0 and sf.harmonics[63] == 0
    assert sf.noise[0] == pytest.approx(0.0)                    # один вид — чисто
    assert sf.noise[32] == pytest.approx(1.0)                   # два вида поровну — макс. энтропия
    # 128^3 -> полосы по 2 слоя: z=1 попадает в полосу 0
    sf2 = SoundMapper(amp_ref=1).map(snap_from([(0, 0, 1, 1, 1)], n=128))
    assert sf2.harmonics[0] == 1.0 and sf2.harmonics[1] == 0.0


def test_voices_pitch_by_size_and_events():
    m = SoundMapper(min_voice_size=4, max_voices=4)
    big = Component(cid=1, species=2, size=1000, center=(0, 5, 5), zmin=0, zmax=9, born=0)
    small = Component(cid=2, species=3, size=8, center=(63, 5, 5), zmin=0, zmax=1, born=3)
    tiny = Component(cid=3, species=1, size=2, center=(5, 5, 5), zmin=0, zmax=0, born=3)
    sf = m.map(snap_from([(0, 0, 0, 1, 1)], gen=5, comps=[big, small, tiny]))
    assert [v.vid for v in sf.voices] == [1, 2]                # tiny отсеян порогом
    v_big, v_small = sf.voices
    assert v_small.harmonic > v_big.harmonic                   # маленький выше
    assert v_big.amp == pytest.approx(1.0) and v_small.amp < 0.2
    assert v_big.pan == pytest.approx(-1.0) and v_small.pan == pytest.approx(1.0)
    assert v_small.age == 2 and sf.births == [1, 2] and sf.deaths == []
    # следующий кадр: big сдвинулся (вибрато), small погиб
    big2 = Component(cid=1, species=2, size=1000, center=(1.5, 5, 5), zmin=0, zmax=9, born=0)
    sf2 = m.map(snap_from([(0, 0, 0, 1, 1)], gen=6, comps=[big2]))
    assert sf2.deaths == [2] and sf2.births == []
    assert sf2.voices[0].vib > 0.5


def test_activity_measures_change():
    m = SoundMapper()
    m.map(snap_from([(i, 0, 0, 1, 1) for i in range(10)]))
    sf = m.map(snap_from([(i, 0, 0, 1, 1) for i in range(5, 15)]))   # половина сменилась
    assert sf.activity == pytest.approx(10 / 15)


def test_to_dict_is_json_friendly():
    import json
    sf = SoundMapper().map(snap_from([(0, 0, 0, 1, 1)]))
    d = sf.to_dict()
    json.dumps(d)
    assert set(d) >= {"gen", "harmonics", "noise", "voices", "births", "deaths", "activity"}


def _spectrum(wave, sr):
    x = wave.mean(axis=1) if wave.ndim == 2 else wave
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr), spec


def test_synth_spectrum_matches_frame():
    """Оракул: в WAV должны торчать ровно те гармоники, что заданы в кадре."""
    sr, base = 22050, 100.0
    h = [0.0] * 64; h[0] = 1.0; h[4] = 0.8            # 100 Гц и 500 Гц
    f = SoundFrame(gen=0, harmonics=h, noise=[0.0] * 64,
                   voices=[dict(vid=1, harmonic=9, amp=1.0, pan=0.0, vib=0.0, age=1, species=1)])
    from life_cube.sound.features import Voice
    f.voices = [Voice(**f.voices[0])]                  # голос на 900 Гц
    wave = render([f] * 10, sr=sr, seconds_per_frame=0.2, base_hz=base, noise_gain=0.0)
    assert wave.shape == (sr * 2, 2) and np.abs(wave).max() <= 0.98
    freqs, spec = _spectrum(wave, sr)
    from scipy.signal import find_peaks
    idx, _ = find_peaks(spec, height=0.2 * spec.max(), distance=int(50 / (freqs[1] - freqs[0])))
    assert sorted(int(round(f / 100)) * 100 for f in freqs[idx]) == [100, 500, 900]
    # ничего лишнего: энергия вне этих трёх частот мала
    mask = np.ones_like(spec, bool)
    for fc in (100, 500, 900):
        mask &= np.abs(freqs - fc) > 15
    assert spec[mask].max() < 0.05 * spec.max()


def test_noise_follows_entropy():
    sr = 22050
    clean = SoundFrame(gen=0, harmonics=[1.0] + [0] * 63, noise=[0.0] * 64)
    dirty = SoundFrame(gen=0, harmonics=[1.0] + [0] * 63, noise=[1.0] * 64)
    w0 = render([clean] * 5, sr=sr, base_hz=100)
    w1 = render([dirty] * 5, sr=sr, base_hz=100)
    f, s0 = _spectrum(w0, sr); _, s1 = _spectrum(w1, sr)
    band = (f > 2000) & (f < 8000)                     # там нет гармоник — только шум
    assert s1[band].mean() > 20 * s0[band].mean()


def test_sonify_run_and_wav(tmp_path):
    frames, wave = sonify_run(Config(n=24, gens=8, seed_density=0.05), seconds_per_frame=0.05)
    assert len(frames) == 8 and wave.shape[0] == 8 * int(44100 * 0.05)
    assert np.abs(wave).max() > 0.01                   # не тишина
    p = write_wav(str(tmp_path / "run.wav"), wave)
    from scipy.io import wavfile
    sr, data = wavfile.read(p)
    assert sr == 44100 and data.shape == wave.shape and data.dtype == np.int16
