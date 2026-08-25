"""Командная строка.

    life-cube run  [--n 128 --gens 200 --gpu --out x.png --save-state x.npz]
    life-cube serve [--n 128 --gpu --port 8765 --rate 10]   # живой веб-просмотр
    python -m life_cube ...                                  # то же самое

Без подкоманды работает как `run` (обратная совместимость).
"""

import argparse
import sys

from .config import Config
from .sim import run, save_state


def _common(p):
    p.add_argument("--n", type=int, default=128, help="размер куба")
    p.add_argument("--seed-world", type=int, default=20260825)
    p.add_argument("--seed-mut", type=int, default=20260825)
    p.add_argument("--gpu", action="store_true", help="считать на CuPy")
    p.add_argument("--seed-density", type=float, default=None)


def build_parser():
    p = argparse.ArgumentParser(
        prog="life-cube",
        description="3D клеточный автомат с экологией: камень, свет, вода, виды")
    sub = p.add_subparsers(dest="mode")

    r = sub.add_parser("run", help="прогон с картинкой в конце")
    _common(r)
    r.add_argument("--gens", type=int, default=200, help="число поколений")
    r.add_argument("--no-render", action="store_true")
    r.add_argument("--out", default="cube_ecology.png")
    r.add_argument("--save-state", default=None, help="путь для .npz со снимком")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--wav", default=None,
                   help="озвучить прогон в WAV (0.1 с на поколение)")
    r.add_argument("--wav-spf", type=float, default=0.1, help="секунд на поколение")
    r.add_argument("--base-hz", type=float, default=55.0)

    s = sub.add_parser("serve", help="веб-просмотр: http://host:port/")
    _common(s)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--rate", type=float, default=0.0,
                   help="целевых поколений/с (0 = без предела)")
    s.add_argument("--snapshot-every", type=int, default=0,
                   help="снимок каждое k-е поколение; 0 = независимо от симуляции, не чаще --fps")
    s.add_argument("--fps", type=float, default=25.0, help="предел кадров/с для зрителей")
    s.add_argument("--components-hz", type=float, default=2.0,
                   help="как часто пересчитывать организмы (дорого)")
    s.add_argument("--yield-ms", type=float, default=0.5,
                   help="сколько мс уступать веб-серверу на каждом шаге при rate=0")
    s.add_argument("--no-components", action="store_true",
                   help="не считать организмы (быстрее)")
    s.add_argument("--paused", action="store_true", help="стартовать на паузе")
    return p


def _cfg(a, **kw):
    extra = {}
    if getattr(a, "seed_density", None) is not None:
        extra["seed_density"] = a.seed_density
    return Config(n=a.n, seed_world=a.seed_world, seed_mut=a.seed_mut, **extra, **kw)


def main(argv=None):
    """Точка входа console_script: возвращает код выхода, не результат."""
    run_cli(argv)
    return 0


def run_cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run"] + argv
    a = build_parser().parse_args(argv)

    if a.mode == "serve":
        from .viewers.web import serve
        serve(_cfg(a), use_gpu=a.gpu, host=a.host, port=a.port, rate=a.rate,
              snapshot_every=a.snapshot_every, components=not a.no_components,
              autostart=not a.paused, fps=a.fps, components_hz=a.components_hz, yield_ms=a.yield_ms)
        return None

    cfg = _cfg(a, gens=a.gens)
    if a.wav:
        from .sound.synth import sonify_run, write_wav
        frames, wave = sonify_run(cfg, use_gpu=a.gpu, seconds_per_frame=a.wav_spf,
                                  base_hz=a.base_hz)
        print("звук:", write_wav(a.wav, wave), f"({len(wave)/44100:.1f} с)")
    res = run(cfg, use_gpu=a.gpu, verbose=not a.quiet)
    if a.save_state:
        print("состояние:", save_state(res, a.save_state))
    if not a.no_render:
        from .viewers.matplotlib import render
        print("картинка:", render(res, a.out))
    return res


if __name__ == "__main__":
    sys.exit(main())
