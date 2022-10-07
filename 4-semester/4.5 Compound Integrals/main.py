import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.16f}'.format


def F(x: float) -> float:
    return x**2 * np.cos(x) + x * np.sin(x)


def IF(x: float) -> float:
    return (x**2 - 1) * np.sin(x) + x * np.cos(x)


a_ = 0
b_ = 4
n_ = 10


def getSum(f, a: float, b: float, n: int) -> float:
    return sum([f(a + i * (b - a) / n) for i in range(n_ + 1)])


def KFL(s: float, a: float, b: float) -> float:
    return s / (b - a)


def printGraph(f, a: float, b: float, expected: float, n: int):

    ptsnum = 1000
    ep = (b - a) / 4

    X = np.linspace(a - ep, b + ep, ptsnum)
    y = [f(x) for x in X]
    Ox = [0 for _ in X]

    fig, ax = plt.subplots()
    ax.plot(X, y, color="blue", alpha=0.5, label=f"{f.__name__}(x)")
    ax.plot(X, Ox, color="black")

    X = [x for x in X if a <= x <= b]
    y = [f(x) for x in X]
    Ox = [0 for _ in X]

    h = (b - a) / n
    for i in range(n_):
        c = a + i * (b - a) / n
        ax.add_patch(ptch.Polygon(((c, 0), (c, f(c + h / 2)), (c+h, f(c + h / 2)), (c+h, 0)), edgecolor="#1207ff", facecolor="#1207ff",
                                    fill=True, alpha=0.2))

    ax.fill_between(X, Ox, y, facecolor="red", alpha=0.2, label=f"{round(expected, 3)}")
    ax.vlines(a, min(f(a), 0), max(f(a), 0), color="blue", alpha=0.2, linestyle="--")
    ax.vlines(b, min(f(b), 0), max(f(b), 0), color="blue", alpha=0.2, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()


printGraph(F, a_, b_, IF(b_) - IF(a_), n_)

