# life_cube

Трёхмерный клеточный автомат с эмерджентной экологией: камень, свет сверху,
вода снизу, четыре вида, конкурирующих по избытку ресурса. Прогон 128³ на 200
поколений даёт устойчивое сосуществование видов, разошедшихся по нишам.

Концепция и план проекта — в [`docs/concept.md`](docs/concept.md).
Исходный монолит, от которого отпочковался пакет, — в `legacy/cube_ecology.py`
(тесты проверяют побитовую эквивалентность с ним).

## Установка

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, scipy, matplotlib, pytest
pip install -e ".[gpu]"          # + cupy-cuda12x для --gpu
```

## Запуск

```bash
life-cube                                   # 128^3, 200 поколений, CPU
life-cube --n 256 --gens 800 --gpu          # CuPy на видеокарте
life-cube --n 64 --gens 120 --no-render
life-cube --save-state run.npz --out run.png
python -m life_cube --help
```

Из кода:

```python
from life_cube import Config, run
from life_cube.render import render

res = run(Config(n=64, gens=100), use_gpu=False)
render(res, "cube.png")
```

## Тесты

```bash
pytest
```

## Структура

| Модуль | Что делает |
|---|---|
| `life_cube/backend.py` | numpy на CPU / cupy на GPU, `correlate` (не `convolve`!) |
| `life_cube/config.py` | геномы видов, `Config` мира |
| `life_cube/world.py` | анизотропное ядро 3×3×3, рельеф, влажность, засев |
| `life_cube/fields.py` | поле света (затенение), поле воды (подъём по телу), ресурс |
| `life_cube/step.py` | одно поколение: рождение, выживание, мутация, растворение, шок |
| `life_cube/sim.py` | прогон, история, сохранение `.npz` |
| `life_cube/render.py` | картинка: разрез, вид сверху, население, атлас слоёв |
| `life_cube/cli.py` | командная строка |

## Два сида

`seed_world` задаёт рельеф, влажность и засев; `seed_mut` — весь поток
случайности во время жизни. Один ландшафт — много историй, и наоборот.
