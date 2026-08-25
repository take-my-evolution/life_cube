"""Рендеры. Каждый рендер — потребитель Snapshot из life_cube.snapshot и
клиент управления Engine из life_cube.engine; ядро о рендерах не знает.

    viewers.matplotlib  — статичная картинка по результату прогона
    viewers.web         — живой просмотр в браузере (WebGL2 + WebSocket)
"""
