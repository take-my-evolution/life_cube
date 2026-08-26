# Архитектура

```
life_cube/
  engines/            движки (наборы правил) — сменные
    __init__.py       реестр и интерфейс Rules
    ecology.py        обёртка над config/world/fields/step/motion
    lichen.py         самостоятельный движок (свой Config, мир, шаг, гены)
  config.py world.py fields.py step.py motion.py   — модель «экология»
  engine.py           оркестровка: цикл, пауза/шаг/скорость, снимки, смена движка
  snapshot.py         разреженный снимок, организмы (компоненты), устойчивые id
  sound/              что звучит (features) и чем (synth, Web Audio в клиенте)
  viewers/            рендеры: matplotlib (картинка), web (aiohttp + WebGL2)
  cli.py              life-cube run | serve
docs/                 концепты и реализация
tests/                pytest (+ Playwright для браузера)
legacy/               исходный монолит cube_ecology.py (эталон для тестов)
```

## Интерфейс движка (`engines.Rules`)

| Метод | Что делает |
|---|---|
| `Config` | dataclass параметров (обязательно `n`, `seed_world`, `seed_mut`, `genomes`) |
| `init_state(cfg, xp)` | → `(state, relief)`; в `state` обязательно `species` (0 = пусто), `stone`, `soil` |
| `step(state, cfg, xp, correlate, gen)` | одно поколение; → население по видам |
| `species_names/colors(cfg[, state])` | подписи и палитра; у динамических видов — от состояния |
| `to_json(cfg, state)` | описание для панели: `fields`, `labels`, `ranges`, `genomes`, `world`, (`ids`) |
| `apply_genomes`, `randomize`, `world_params` | правка на лету, случайные гены, список ручек мира |
| флаги `dynamic_species`, `terrain_changes` | слать имена видов и рельеф каждый кадр |

Всё остальное — общее: `Engine` крутит любой движок, `snapshot.py` видит только
`species`, рендеры и звук — только `Snapshot`.

## Поток данных
```
Rules.step ──► state ──► Engine.publish ──► Snapshot ──┬─► viewers/web (WebSocket, WebGL2)
                                                       ├─► viewers/matplotlib
                                                       └─► sound.features ──► SoundFrame ──► Web Audio / synth.wav
```
Снимки не тормозят симуляцию: копия состояния берётся под замком, разметка
организмов — вне его, не чаще `--fps`, организмы — не чаще `--components-hz`.

## Что где менять
- новая механика в «экологии» — `step.py`/`motion.py`, тест в `tests/`, запись в `docs/engines/ecology.md`;
- новая концепция — новый модуль в `engines/`, строка в реестре `ENGINES`, `docs/engines/<имя>.md`;
- новый рендер — модуль в `viewers/`, потребляет `Snapshot`;
- новая логика звука — `sound/features.py`, бэкенды не трогаются.
