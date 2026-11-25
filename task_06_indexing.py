import numpy as np

rng = np.random.default_rng(42)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# 6.1
def task_6_1() -> None:
    section("6.1. Срезы массива np.arange(20)")
    arr = np.arange(20)
    print("arr =", arr)

    every_third = arr[1::3]
    print("каждый третий, начиная с индекса 1:", every_third)

    reversed_5_15 = arr[5:15][::-1]
    assert np.array_equal(reversed_5_15, arr[14:4:-1])
    print("элементы [5, 15) в обратном порядке:", reversed_5_15)

    last5_reversed = arr[-5:][::-1]
    assert np.array_equal(last5_reversed, arr[:-6:-1])
    print("последние 5 в обратном порядке:", last5_reversed)


# 6.2
def task_6_2() -> None:
    section("6.2. Матрица 6x6 случайных целых от 0 до 9")
    m = rng.integers(0, 10, size=(6, 6))
    print("m =\n", m)

    sub = m[2:5, 1:4]
    print("\nподматрица строки 2-4, столбцы 1-3:\n", sub)

    diag = np.diagonal(m)
    print("\nглавная диагональ:", diag)

    i = np.arange(m.shape[0])[:, None]
    j = np.arange(m.shape[1])[None, :]
    mask = (i + j) % 2 == 0
    print("\nмаска (i + j) чётно:\n", mask.astype(int))
    print("элементы с чётной суммой индексов:", m[mask])
    print("матрица, где «чёрные» клетки оставлены, а «белые» обнулены:\n",
          np.where(mask, m, 0))

    check = np.concatenate([m[0::2, 0::2].ravel(), m[1::2, 1::2].ravel()])
    assert np.array_equal(np.sort(m[mask]), np.sort(check))


# 6.3
def task_6_3() -> None:
    section("6.3. Fancy indexing")
    a = np.array([10, 20, 30, 40, 50, 60])
    m = np.arange(12).reshape(3, 4)
    print("a =", a)
    print("m =\n", m)

    print("\na[[0, 2, 4]] =", a[[0, 2, 4]])

    rows = [0, 1, 2]
    cols = [1, 3, 0]
    pairs = m[rows, cols]
    print("\nm[(0,1), (1,3), (2,0)] =", pairs)
    assert list(pairs) == [m[0, 1], m[1, 3], m[2, 0]]

    block = m[np.ix_([0, 2], [1, 3])]
    print("\nблок 2x2 через np.ix_:\n", block)
    block2 = m[[[0], [2]], [[1, 3]]]
    block3 = m[[0, 2]][:, [1, 3]]
    assert np.array_equal(block, block2) and np.array_equal(block, block3)
    print("форма блока:", block.shape)


# 6.4
def task_6_4() -> None:
    section("6.4. Булевы маски на x = randn(1000)")
    x = rng.standard_normal(1000)

    count_gt1 = int((x > 1).sum())
    print("элементов > 1:", count_gt1,
          f"({count_gt1 / x.size:.1%}; теоретически ~15.9%)")

    x_clipped = x.copy()
    x_clipped[x_clipped < 0] = 0
    assert np.array_equal(x_clipped, np.maximum(x, 0))
    print("отрицательных до: ", int((x < 0).sum()),
          "| после замены:", int((x_clipped < 0).sum()))

    x_zeroed = x.copy()
    left_neg = x[:-2] < 0
    right_neg = x[2:] < 0
    sign_change = left_neg != right_neg
    x_zeroed[1:-1][sign_change] = 0

    expected = x.copy()
    for k in range(1, len(x) - 1):
        if (x[k - 1] < 0) != (x[k + 1] < 0):
            expected[k] = 0
    assert np.array_equal(x_zeroed, expected)

    print("позиций со сменой знака у соседей:", int(sign_change.sum()),
          "из", x.size - 2)
    print("первые 10 исходных :", np.round(x[:10], 3))
    print("первые 10 обнулённых:", np.round(x_zeroed[:10], 3))


if __name__ == "__main__":
    task_6_1()
    task_6_2()
    task_6_3()
    task_6_4()
