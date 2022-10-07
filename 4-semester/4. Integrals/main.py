import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import pandas as pd
from functions import promptPair

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def F(x: float) -> float:
    return x**2 * np.cos(x) + x * np.sin(x)


def IF(x: float) -> float:
    return (x**2 - 1) * np.sin(x) + x * np.cos(x)


def P0(x: float) -> float:
    return 2


def IP0(x: float) -> float:
    return 2 * x


def P1(x: float) -> float:
    return x + 1


def IP1(x: float) -> float:
    return x ** 2 / 2 + x


def P2(x: float) -> float:
    return 2 * x**2 - 2


def IP2(x: float) -> float:
    return 2 * x ** 3 / 3 - 2 * x


def P3(x: float) -> float:
    return 3 * x ** 3 - 2 * x ** 2 + x


def IP3(x: float) -> float:
    return 3 * x ** 4 / 4 - 2 * x ** 3 / 3 + x ** 2 / 2


a_, b_ = promptPair()


def KFL(f, a: float, b: float) -> float:
    return (b - a) * f(a)


def KFR(f, a: float, b: float) -> float:
    return (b - a) * f(b)


def KFM(f, a: float, b: float) -> float:
    return (b - a) * f((a + b) / 2)


def KFT(f, a: float, b: float) -> float:
    return (b - a) * (f(a) + f(b)) / 2


def KFS(f, a: float, b: float) -> float:
    return (b - a) * (f(a) + 4 * f((a + b) / 2) + f(b)) / 6


def KF38(f, a: float, b: float) -> float:
    h = (b - a) / 3
    return (b - a) * (f(a) / 8 + 3 * f(a + h) / 8 + 3 * f(a + 2 * h) / 8 + f(b) / 8)


def printData(name: str, f, intf, a: float, b: float):
    print(f"\nФункция: f(x) = {name}")
    expected = intf(b) - intf(a)
    data = {
        "Значение интеграла": np.array([expected, 0]),
        "КФ левого прямоугольника": np.array([KFL(f, a, b), abs(expected - KFL(f, a, b))]),
        "КФ правого прямоугольника": np.array([KFR(f, a, b), abs(expected - KFR(f, a, b))]),
        "КФ среднего прямоугольника": np.array([KFM(f, a, b), abs(expected - KFM(f, a, b))]),
        "КФ трапеции": np.array([KFT(f, a, b), abs(expected - KFT(f, a, b))]),
        "КФ Симпсона": np.array([KFS(f, a, b), abs(expected - KFS(f, a, b))]),
        "КФ 3/8": np.array([KF38(f, a, b), abs(expected - KF38(f, a, b))])
    }
    df = pd.DataFrame(data, index=["Значение", "Погрешность"])
    print(df)


def printGraph(f, a: float, b: float, expected: float):

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

    # ax.add_patch(ptch.Rectangle((a, 0), b - a, f(a), edgecolor="#1207ff", facecolor="#1207ff",
    #                             fill=True, alpha=0.2, label=f"{round(KFL(f, a, b), 3)}"))
    # ax.add_patch(ptch.Rectangle((a, 0), b - a, f(b), edgecolor="#ff07eb", facecolor="#ff07eb",
    #                             fill=True, alpha=0.2, label=f"{round(KFR(f, a, b), 3)}"))
    # ax.add_patch(ptch.Rectangle((a, 0), b - a, f((a + b) / 2), edgecolor="#fcff07", facecolor="#fcff07",
    #                             fill=True, alpha=0.2, label=f"{round(KFM(f, a, b), 3)}"))
    # ax.add_patch(ptch.Polygon(((a, 0), (a, f(a)), (b, f(b)), (b, 0)), edgecolor="#1207ff", facecolor="#1207ff",
    #                              fill=True, alpha=0.2, label=f"{round(KFM(f, a, b), 3)}"))
    ax.fill_between(X, Ox, y, facecolor="red", alpha=0.2, label=f"{round(expected, 3)}")
    ax.vlines(a, min(f(a), 0), max(f(a), 0), color="blue", alpha=0.2, linestyle="--")
    ax.vlines(b, min(f(b), 0), max(f(b), 0), color="blue", alpha=0.2, linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()


printData("x^2cos(x) + xsin(x)", F, IF, a_, b_)
printData("2", P0, IP0, a_, b_)
printData("x + 1", P1, IP1, a_, b_)
printData("2x^2 - 2", P2, IP2, a_, b_)
printData("3x^3 - 2x^2 + x", P3, IP3, a_, b_)

printGraph(F, a_, b_, IF(b_) - IF(a_))
# printGraph(P0, a_, b_, IP0(b_) - IP0(a_))
# printGraph(P1, a_, b_, IP1(b_) - IP1(a_))
# printGraph(P2, a_, b_, IP2(b_) - IP2(a_))
# printGraph(P3, a_, b_, IP3(b_) - IP3(a_))
