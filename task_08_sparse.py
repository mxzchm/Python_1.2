from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from utils import best_time, human_bytes

RESULTS_DIR = Path(__file__).parent / "results"
rng = np.random.default_rng(42)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def make_sparse(n: int, density: float, rng: np.random.Generator) -> sparse.csr_matrix:
    nnz = max(1, round(n * n * density))
    flat = rng.choice(n * n, size=nnz, replace=False)
    rows, cols = np.divmod(flat, n)
    data = rng.random(nnz) + 0.5
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def csr_nbytes(m: sparse.csr_matrix) -> int:
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


# 8.1
def task_8_1() -> None:
    section("8.1. COO -> CSR и корректность nnz при совпадающих индексах")
    n, k = 5000, 10_000

    row = rng.integers(0, n, size=k).tolist()
    col = rng.integers(0, n, size=k).tolist()
    data = rng.standard_normal(k).tolist()

    for t in range(3):
        row[-1 - t], col[-1 - t] = row[t], col[t]
    data[-1] = -data[0]

    coo = sparse.coo_matrix((data, (row, col)), shape=(n, n))
    print(f"COO: shape={coo.shape}, nnz (хранимых записей) = {coo.nnz}")

    csr = coo.tocsr()
    unique_pairs = len(set(zip(row, col)))
    print(f"число различных пар (row, col)   = {unique_pairs}")
    print(f"CSR: nnz после конвертации        = {csr.nnz}")
    assert csr.nnz == unique_pairs, "nnz CSR должен равняться числу уникальных позиций"

    assert np.isclose(csr.sum(), np.sum(data))

    true_nonzero = csr.count_nonzero()
    print(f"из них реально ненулевых          = {true_nonzero}")
    csr_clean = csr.copy()
    csr_clean.eliminate_zeros()
    print(f"CSR после eliminate_zeros(): nnz  = {csr_clean.nnz}")
    assert csr_clean.nnz == true_nonzero == unique_pairs - 1

    print(f"\nИтого: {k - unique_pairs} записей COO «слиплись» из-за повторов индексов,"
          f"\nещё {unique_pairs - true_nonzero} стала нулём после сложения.")


# 8.2
def task_8_2() -> None:
    section("8.2. Матрица 4000x4000, плотность 0.5%: dense vs CSR")
    n, density = 4000, 0.005

    a_csr = make_sparse(n, density, rng)
    a_dense = a_csr.toarray()
    x = rng.random(n)

    print(f"nnz = {a_csr.nnz:,}  (плотность {a_csr.nnz / n**2:.3%})")

    mem_dense = a_dense.nbytes
    mem_csr = csr_nbytes(a_csr)
    print("\nПамять:")
    print(f"  dense float64 : {mem_dense:>12,} байт ({human_bytes(mem_dense)})")
    print(f"  CSR           : {mem_csr:>12,} байт ({human_bytes(mem_csr)})")
    print(f"    data    {a_csr.data.nbytes:>10,} Б  ({a_csr.data.dtype}, {a_csr.nnz:,} значений)")
    print(f"    indices {a_csr.indices.nbytes:>10,} Б  ({a_csr.indices.dtype}, номера столбцов)")
    print(f"    indptr  {a_csr.indptr.nbytes:>10,} Б  ({a_csr.indptr.dtype}, n+1 смещений строк)")
    print(f"  CSR меньше dense в {mem_dense / mem_csr:.0f} раз")

    assert np.allclose(a_dense @ x, a_csr @ x)
    t_dense = best_time(lambda: a_dense @ x, repeat=10, number=10)
    t_csr = best_time(lambda: a_csr @ x, repeat=10, number=10)
    print("\nВремя умножения A @ x (лучшее из 10 серий по 10 вызовов):")
    print(f"  dense : {t_dense * 1e3:8.3f} мс")
    print(f"  CSR   : {t_csr * 1e3:8.3f} мс")
    print(f"  CSR быстрее в {t_dense / t_csr:.1f} раз")


# 8.3
def task_8_3() -> None:
    section("8.3. Время A @ x в зависимости от плотности (n = 2000)")
    n = 2000
    densities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    x = rng.random(n)

    rows = []
    for d in densities:
        a_csr = make_sparse(n, d, rng)
        a_dense = a_csr.toarray()
        assert np.allclose(a_dense @ x, a_csr @ x)
        t_dense = best_time(lambda: a_dense @ x, repeat=10, number=10)
        t_csr = best_time(lambda: a_csr @ x, repeat=10, number=10)
        rows.append({
            "density": d,
            "nnz": a_csr.nnz,
            "t_dense_ms": t_dense * 1e3,
            "t_csr_ms": t_csr * 1e3,
            "speedup": t_dense / t_csr,
            "mem_dense_MiB": a_dense.nbytes / 2**20,
            "mem_csr_MiB": csr_nbytes(a_csr) / 2**20,
        })

    print(f"{'density':>8}{'nnz':>11}{'dense, мс':>11}{'CSR, мс':>10}"
          f"{'dense/CSR':>11}{'пам. dense':>12}{'пам. CSR':>10}")
    for r in rows:
        mark = "" if r["speedup"] > 1 else "  <- sparse не выигрывает"
        print(f"{r['density']:>8.3f}{r['nnz']:>11,}{r['t_dense_ms']:>11.3f}"
              f"{r['t_csr_ms']:>10.3f}{r['speedup']:>10.2f}x"
              f"{r['mem_dense_MiB']:>10.1f}МиБ{r['mem_csr_MiB']:>7.1f}МиБ{mark}")

    losing = [r for r in rows if r["speedup"] <= 1]
    if losing:
        d_star = losing[0]["density"]
        prev = [r for r in rows if r["density"] < d_star]
        last_win = prev[-1]["density"] if prev else None
        print(f"\nSparse перестаёт выигрывать по времени начиная с плотности {d_star}"
              f" (последняя выигрышная точка из сетки: {last_win}).")

        if prev:
            r0, r1 = prev[-1], losing[0]
            l0, l1 = np.log(r0["speedup"]), np.log(r1["speedup"])
            frac = l0 / (l0 - l1)
            d_cross = float(np.exp(np.log(r0["density"]) +
                                   frac * (np.log(r1["density"]) - np.log(r0["density"]))))
            print(f"Оценка точки пересечения по интерполяции: плотность ≈ {d_cross:.2f}")
    else:
        print("\nНа всей сетке плотностей CSR оказался быстрее dense.")

    print("По памяти CSR невыгоден при density > 8/12 ≈ 0.67"
          " (12 Б на элемент у CSR против 8 Б у dense).")

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "task_08_3_timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog([r["density"] for r in rows], [r["t_dense_ms"] for r in rows],
              "o-", label="dense (np.ndarray @ x)")
    ax.loglog([r["density"] for r in rows], [r["t_csr_ms"] for r in rows],
              "s-", label="sparse (CSR @ x)")
    ax.set_xlabel("Плотность матрицы")
    ax.set_ylabel("Время A @ x, мс")
    ax.set_title(f"Умножение матрицы на вектор, n = {n}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_path = RESULTS_DIR / "task_08_3_timings.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {csv_path.name}, {png_path.name} (папка results/)")


if __name__ == "__main__":
    task_8_1()
    task_8_2()
    task_8_3()
