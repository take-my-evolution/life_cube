"""life_cube — трёхмерный клеточный автомат с экологией: камень, свет, вода,
конкурирующие виды.

Модули:
    backend  — выбор numpy (CPU) / cupy (GPU)
    config   — геномы видов и параметры мира
    world    — ядро соседства, рельеф, влажность, засев
    fields   — поля света и воды, функция ресурса
    step     — одно поколение
    sim      — прогон на N поколений
    render   — картинка
    cli      — командная строка
"""

from .config import Config, DEFAULT_GENOMES, GENOME_FIELDS, SPECIES_NAMES
from .sim import run
from .step import step

__version__ = "0.4.0"
__all__ = ["Config", "DEFAULT_GENOMES", "GENOME_FIELDS", "SPECIES_NAMES",
           "run", "step", "__version__"]
