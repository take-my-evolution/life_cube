"""Командная строка.

    life-cube                                # 128^3, 200 поколений, CPU
    life-cube --n 256 --gens 800 --gpu       # CuPy на GPU
    life-cube --n 64 --gens 120 --no-render
    python -m life_cube ...                  # то же самое
"""

import argparse

from .config import Config
from .sim import run, save_state


def build_parser():
    p = argparse.ArgumentParser(
        prog="life-cube",
        description="3D клеточный автомат с экологией: камень, свет, вода, виды")
    p.add_argument("--n", type=int, default=128, help="размер куба")
    p.add_argument("--gens", type=int, default=200, help="число поколений")
    p.add_argument("--seed-world", type=int, default=20260825)
    p.add_argument("--seed-mut", type=int, default=20260825)
    p.add_argument("--gpu", action="store_true", help="считать на CuPy")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--out", default="cube_ecology.png")
    p.add_argument("--save-state", default=None, help="путь для .npz со снимком")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = Config(n=a.n, gens=a.gens,
                 seed_world=a.seed_world, seed_mut=a.seed_mut)
    res = run(cfg, use_gpu=a.gpu, verbose=not a.quiet)

    if a.save_state:
        print("состояние:", save_state(res, a.save_state))

    if not a.no_render:
        from .render import render
        print("картинка:", render(res, a.out))
    return res


if __name__ == "__main__":
    main()
