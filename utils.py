from __future__ import annotations

import time
from typing import Callable


def best_time(func: Callable[[], object], repeat: int = 5, number: int = 1) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            func()
        elapsed = (time.perf_counter() - start) / number
        best = min(best, elapsed)
    return best


def human_bytes(n: int | float) -> str:
    for unit in ("Б", "КиБ", "МиБ", "ГиБ"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}".replace(",", " ")
        n /= 1024
    return f"{n:.1f} ТиБ"
