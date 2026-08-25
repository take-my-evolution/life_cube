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

## Живой просмотр в браузере

```bash
pip install -e ".[web]"
life-cube serve --gpu --n 128 --rate 10          # http://<host>:8765/
life-cube serve --n 64 --port 9000 --paused      # стартовать на паузе
```

Симуляция крутится на сервере (Engine), браузер получает разреженные снимки
по WebSocket и рисует их WebGL2-шейдером (instanced-вокселы). В клиенте:
пауза / шаг / скорость / сброс с другими сидами; перспектива или ортографика,
пресеты «сверху / спереди / сбоку / изо», срез по высоте и толстый срез по y;
фильтр по видам, минимальный размер организма, раскраска по виду или по
организму, выделение одного организма из таблицы (остальное — призраком).
Организмы — связные компоненты одного вида (26-связность) с устойчивыми id
между поколениями (`life_cube/snapshot.py`).

## Тесты

```bash
pytest                      # ядро, снимки, движок, протокол
pip install playwright && playwright install chromium
pytest tests/test_web.py    # скриншот-тесты клиента в headless Chromium
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
| `life_cube/snapshot.py` | разреженный снимок, организмы (компоненты), отслеживание id |
| `life_cube/engine.py` | управляемый цикл: пауза, шаг, целевая скорость, подписчики |
| `life_cube/viewers/matplotlib.py` | статичная картинка: разрез, вид сверху, население, атлас слоёв |
| `life_cube/viewers/web/` | сервер (aiohttp + WebSocket) и клиент (WebGL2) живого просмотра |
| `life_cube/cli.py` | командная строка |

## Два сида

`seed_world` задаёт рельеф, влажность и засев; `seed_mut` — весь поток
случайности во время жизни. Один ландшафт — много историй, и наоборот.
