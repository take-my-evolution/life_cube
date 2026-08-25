"""Прогон мира на N поколений."""

import time

import numpy as np

from .backend import get_backend, max_filter_for, to_cpu
from .config import Config
from .step import step
from .world import build_kernel, build_world


def init_state(cfg: Config, xp):
    stone, wet, species, relief, energy = build_world(cfg, xp)
    state = {
        "species": species,
        "energy": energy,
        "age": xp.zeros((cfg.n,) * 3, dtype=xp.int32),
        "stone": stone,
        "soil": xp.zeros((cfg.n,) * 3, dtype=bool),
        "wet": wet,
        "kernel": build_kernel(xp),
        "genomes": xp.asarray(cfg.genomes),
        # отдельный поток случайности для жизни — не смешивается с сидом мира
        "rng": xp.random.default_rng(cfg.seed_mut),
        "maxfilter": max_filter_for(xp),
    }
    return state, relief


def run(cfg: Config, use_gpu: bool = False, verbose: bool = True,
        callback=None):
    """Полный прогон. callback(gen, pops, state) вызывается после каждого шага."""
    xp, correlate, on_gpu = get_backend(use_gpu)
    state, relief = init_state(cfg, xp)

    hist = []
    t0 = time.time()
    for gen in range(cfg.gens):
        pops = step(state, cfg, xp, correlate, gen)
        hist.append(pops)
        if callback is not None:
            callback(gen, pops, state)
        if verbose and (gen % max(1, cfg.gens // 8) == 0 or gen == cfg.gens - 1):
            print(f"поколение {gen:>5}  всего {sum(pops):>8}  "
                  f"по видам {pops}", flush=True)
    dt = time.time() - t0
    if verbose:
        print(f"готово за {dt:.1f} c  ({cfg.gens / max(dt, 1e-9):.1f} поколений/с, "
              f"{'GPU' if on_gpu else 'CPU'})")

    return {
        "species": to_cpu(state["species"]),
        "energy": to_cpu(state["energy"]),
        "stone": to_cpu(state["stone"]),
        "soil": to_cpu(state["soil"]),
        "relief": relief,
        "hist": np.array(hist),
        "config": cfg,
        "seconds": dt,
        "gpu": on_gpu,
    }


def save_state(result: dict, path: str):
    np.savez_compressed(path, species=result["species"],
                        stone=result["stone"], soil=result["soil"],
                        relief=result["relief"], hist=result["hist"])
    return path
