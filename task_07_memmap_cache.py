from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from utils import best_time, human_bytes

DATA_DIR = Path(__file__).parent / "data"
BIG_FILE = DATA_DIR / "big.dat"
SHAPE = (50_000, 200)
DTYPE = np.float32
BATCH_ROWS = 5_000

rng = np.random.default_rng(42)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# 7.1
def task_7_1() -> None:
    section("7.1. Создание big.dat (50000 x 200, float32)")
    DATA_DIR.mkdir(exist_ok=True)

    mm = np.memmap(BIG_FILE, dtype=DTYPE, mode="w+", shape=SHAPE)

    for start in range(0, SHAPE[0], BATCH_ROWS):
        stop = min(start + BATCH_ROWS, SHAPE[0])
        mm[start:stop] = rng.random((stop - start, SHAPE[1]), dtype=DTYPE)

    mm.flush()
    del mm

    expected = SHAPE[0] * SHAPE[1] * np.dtype(DTYPE).itemsize
    actual = BIG_FILE.stat().st_size
    print(f"ожидаемый размер: {expected:,} байт ({human_bytes(expected)})")
    print(f"фактический размер: {actual:,} байт")
    assert actual == expected == 40_000_000, "размер файла не совпал!"
    print("OK: файл содержит ровно 50000 x 200 x 4 байт")


# 7.2
def task_7_2() -> None:
    section("7.2. Глобальные min, max, mean по батчам (mode='r')")
    if not BIG_FILE.exists():
        print("big.dat не найден — сначала создаём (7.1)")
        task_7_1()

    mm = np.memmap(BIG_FILE, dtype=DTYPE, mode="r", shape=SHAPE)

    g_min, g_max = np.inf, -np.inf
    total = 0.0
    count = 0

    for start in range(0, SHAPE[0], BATCH_ROWS):
        batch = mm[start:start + BATCH_ROWS]
        g_min = min(g_min, float(batch.min()))
        g_max = max(g_max, float(batch.max()))
        total += float(batch.sum(dtype=np.float64))
        count += batch.size

    mean = total / count
    print(f"min  = {g_min:.8f}")
    print(f"max  = {g_max:.8f}")
    print(f"mean = {mean:.8f}   (элементов: {count:,})")

    full = np.fromfile(BIG_FILE, dtype=DTYPE)
    assert g_min == full.min() and g_max == full.max()
    assert np.isclose(mean, full.mean(dtype=np.float64), rtol=1e-9)
    print("OK: результат совпадает с расчётом по полностью загруженному массиву")

    del mm


def py_row_sums(arr: np.ndarray) -> list[float]:
    n_rows, n_cols = arr.shape
    out = []
    for i in range(n_rows):
        s = 0.0
        for j in range(n_cols):
            s += arr[i, j]
        out.append(s)
    return out


def py_col_sums(arr: np.ndarray) -> list[float]:
    n_rows, n_cols = arr.shape
    out = []
    for j in range(n_cols):
        s = 0.0
        for i in range(n_rows):
            s += arr[i, j]
        out.append(s)
    return out


def timed(func, *args, repeat: int = 1) -> tuple[float, list[float]]:
    best, result = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args)
        best = min(best, time.perf_counter() - start)
    return best, result


# 7.3
def task_7_3(n: int = 3000, repeat: int = 3) -> None:
    section(f"7.3. Сумма по строкам/столбцам: C-order vs F-order ({n}x{n})")

    c_arr = rng.random((n, n))
    f_arr = np.asfortranarray(c_arr)
    assert c_arr.flags["C_CONTIGUOUS"] and f_arr.flags["F_CONTIGUOUS"]
    assert np.array_equal(c_arr, f_arr)

    print(f"(чистые Python-циклы с arr[i, j], лучшее из {repeat} — это займёт время...)\n")
    results = {}
    for name, arr in (("C-order", c_arr), ("F-order", f_arr)):
        t_row, rows = timed(py_row_sums, arr, repeat=repeat)
        t_col, cols = timed(py_col_sums, arr, repeat=repeat)
        assert np.allclose(rows, arr.sum(axis=1))
        assert np.allclose(cols, arr.sum(axis=0))
        results[name] = (t_row, t_col)

    print(f"{'Layout':<10}{'по строкам, с':>16}{'по столбцам, с':>18}")
    for name, (t_row, t_col) in results.items():
        print(f"{name:<10}{t_row:>16.3f}{t_col:>18.3f}")

    print("\nДополнение: np.sum(axis=...) на тех же массивах (мс, лучшее из 5)")
    print(f"{'Layout':<10}{'sum(axis=1)':>14}{'sum(axis=0)':>14}")
    for name, arr in (("C-order", c_arr), ("F-order", f_arr)):
        t1 = best_time(lambda: arr.sum(axis=1), repeat=5) * 1e3
        t0 = best_time(lambda: arr.sum(axis=0), repeat=5) * 1e3
        print(f"{name:<10}{t1:>14.2f}{t0:>14.2f}")

    print("\nДополнение: обход срезами arr[i, :] и arr[:, j] (мс, лучшее из 3)")

    def by_rows(a):
        return sum(a[i, :].sum() for i in range(n))

    def by_cols(a):
        return sum(a[:, j].sum() for j in range(n))

    print(f"{'Layout':<10}{'arr[i, :]':>14}{'arr[:, j]':>14}")
    for name, arr in (("C-order", c_arr), ("F-order", f_arr)):
        t_r = best_time(lambda: by_rows(arr), repeat=3) * 1e3
        t_c = best_time(lambda: by_cols(arr), repeat=3) * 1e3
        print(f"{name:<10}{t_r:>14.2f}{t_c:>14.2f}")

    print("\nВывод: в C-order элементы одной строки лежат в памяти подряд, в F-order —"
          "\nподряд лежат элементы столбца. Обход «вдоль» памяти использует кэш процессора"
          "\n(cache line = 64 байта = 8 float64), обход «поперёк» — прыгает с шагом n*8 байт"
          "\nи даёт кэш-промахи. В чистом Python разница мала (порядка шума измерений):"
          "\nнакладные расходы интерпретатора (~сотни нс на arr[i, j]) маскируют"
          "\nстоимость кэш-промахов."
          "\nУ np.sum(axis=...) разница тоже небольшая: NumPy сам выбирает порядок"
          "\nвнутренних циклов под layout массива. Зато при явном обходе срезами"
          "\narr[i, :] / arr[:, j] выигрыш «правильного» направления заметен"
          "\n(обычно в 1.5-2 раза) и меняется местами при смене C <-> F.")


# 7.4
def task_7_4(n: int = 10_000_000) -> None:
    section(f"7.4. (a + b) / (a - b + 1) на {n:,} элементах float64")

    a = rng.random(n)
    b = rng.random(n)

    a_list, b_list = a.tolist(), b.tolist()

    def python_loop() -> list[float]:
        result = []
        append = result.append
        for x, y in zip(a_list, b_list):
            append((x + y) / (x - y + 1))
        return result

    t0 = time.perf_counter()
    res_py = python_loop()
    t_python = time.perf_counter() - t0

    def numpy_expr() -> np.ndarray:
        return (a + b) / (a - b + 1)

    t_numpy = best_time(numpy_expr, repeat=5)

    buf1 = np.empty_like(a)
    buf2 = np.empty_like(a)

    def numpy_out() -> np.ndarray:
        np.add(a, b, out=buf1)
        np.subtract(a, b, out=buf2)
        np.add(buf2, 1.0, out=buf2)
        np.divide(buf1, buf2, out=buf1)
        return buf1

    t_out = best_time(numpy_out, repeat=5)

    expected = numpy_expr()
    assert np.allclose(expected, res_py)
    assert np.allclose(expected, numpy_out())

    print(f"{'Способ':<28}{'время, с':>12}{'ускорение':>12}")
    for name, t in (("Python-цикл + list", t_python),
                    ("NumPy, одно выражение", t_numpy),
                    ("NumPy с out= (буферы)", t_out)):
        print(f"{name:<28}{t:>12.4f}{t_python / t:>11.1f}x")
    ratio = t_numpy / t_out
    print(f"\nОтношение времени «одно выражение» / «с out=» = {ratio:.2f} "
          "(>1 — вариант с out= быстрее).")
    print("Основной выигрыш (десятки раз) даёт сама векторизация. Вариант с out= "
          "\nубирает аллокации временных массивов по 80 МБ и page faults, но его "
          "\nвыигрыш невелик и зависит от машины/аллокатора (от ~0 до ~30%); он "
          "\nзаметнее, когда выражение вычисляется многократно в цикле.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Задача 7: memory-mapped файлы и оптимизация кэша")
    parser.add_argument("tasks", nargs="*", choices=["1", "2", "3", "4"],
                        help="номера подзадач 7.x (по умолчанию — все)")
    parser.add_argument("--n", type=int, default=3000,
                        help="размер стороны массива в 7.3 (по умолчанию 3000)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="число повторов замера Python-циклов в 7.3 (по умолчанию 3)")
    args = parser.parse_args()
    selected = set(args.tasks or ["1", "2", "3", "4"])

    if "1" in selected:
        task_7_1()
    if "2" in selected:
        task_7_2()
    if "3" in selected:
        task_7_3(args.n, args.repeat)
    if "4" in selected:
        task_7_4()
