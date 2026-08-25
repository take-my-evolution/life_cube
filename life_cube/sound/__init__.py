"""Звук. Разделение обязанностей:

    features.py  — ЧТО звучит: Snapshot -> SoundFrame (чистая функция,
                   здесь живёт логика озвучивания; её можно переписать целиком,
                   не трогая бэкенды)
    synth.py     — ЧЕМ звучит, оффлайн: numpy-аддитивный синтез SoundFrame -> WAV
    viewers/web  — ЧЕМ звучит, живьём: Web Audio в браузере по тому же SoundFrame

Бэкенд получает только SoundFrame и ничего не знает о кубе.
"""

from .features import SoundFrame, SoundMapper

__all__ = ["SoundFrame", "SoundMapper"]
