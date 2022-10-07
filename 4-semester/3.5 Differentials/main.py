import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import promptM, promptA, promptH

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def F(x: float) -> float:
    # return np.exp(1.5 * x)
    return np.cos(x) + 2 * x


def dF(x: float) -> float:
    # return 1.5 * np.exp(1.5 * x)
    return -np.sin(x) + 2


def ddF(x: float) -> float:
    # return 2.25 * np.exp(1.5 * x)
    return -np.cos(x)


def Lagrange(hublist: list):

    def Omega(x: float, xj: float):
        return np.prod([xi[0] - x for xi in hublist if xi[0] != xj])

    def L(x: float):
        return sum([xy[1] * Omega(x, xy[0]) / Omega(xy[0], xy[0]) for xy in hublist])

    return L


def getFirstDerivativeOh2(f, segment: list, h: float):
    left = segment[0]
    right = segment[1]

    def dL2(x: float) -> float:
        if x <= left:
            return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)
        if x >= right:
            return (-3 * f(x) + 4 * f(x - h) - f(x - 2 * h)) / (-2 * h)
        return (f(x + h) - f(x - h)) / (2 * h)

    return dL2


def getFirstDerivativeOh4(f, segment: list, h: float):
    left = segment[0]
    right = segment[1]

    def dL4(x: float) -> float:
        if x <= left:
            return (-25 * f(x) + 48 * f(x + h) - 36 * f(x + 2 * h) + 16 * f(x + 3 * h) - 3 * f(x + 4 * h)) / (12 * h)
        if x >= right:
            return (-25 * f(x) + 48 * f(x - h) - 36 * f(x - 2 * h) + 16 * f(x - 3 * h) - 3 * f(x - 4 * h)) / (-12 * h)
        return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)

    return dL4


def getSecondDerivative(f, h: float):

    def ddL(x: float) -> float:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

    return ddL


m = promptM()
a = promptA()
h = promptH()

# data = {'X': [a + i * h for i in range(m+1)], 'F(X)': [F(a + i * h) for i in range(m+1)]}
# df = pd.DataFrame(data, index=[f"X{i}" for i in range(m+1)])
# print(df)

XY = [[a + i * h, F(a + i * h)] for i in range(m+1)]

L = Lagrange(XY)

dL2 = getFirstDerivativeOh2(L, [a, a + h * m], h)

dL4 = getFirstDerivativeOh4(L, [a, a + h * m], h)

ddL = getSecondDerivative(L, h)

data = {'X': [a + i * h for i in range(m+1)],
        'F': [F(a + i * h) for i in range(m+1)],
        'dL2': [dL2(a + i * h) for i in range(m+1)],
        '|dF - dL2|': [abs(dF(a + i * h) - dL2(a + i * h)) for i in range(m+1)],
        'dL4': [dL4(a + i * h) for i in range(m+1)],
        '|dF - dL4|': [abs(dF(a + i * h) - dL4(a + i * h)) for i in range(m+1)],
        'ddL': [ddL(a + i * h) for i in range(m+1)],
        '|ddF - ddL|': [abs(ddF(a + i * h) - ddL(a + i * h)) for i in range(m+1)]
        }

df = pd.DataFrame(data, index=[f"X{i}" for i in range(m+1)])
print(df)


def printGraph(f, l):
    ptsnum = 1000
    ep = 2 * h
    Ox = np.linspace(a - ep, a + h * m + ep, ptsnum)
    y1 = [f(x) for x in Ox]
    y2 = [l(x) for x in Ox]

    fig, ax = plt.subplots()
    ax.plot(Ox, y1, color="blue", alpha=0.5, label=f"{f.__name__}(x)")
    ax.plot(Ox, y2, color="red", alpha=0.5, label=f"{l.__name__}(x)")
    plt.scatter([xy[0] for xy in XY], [f(xy[0]) for xy in XY],
                color="blue", sizes=[5.0 for _ in XY])
    plt.scatter([xy[0] for xy in XY], [l(xy[0]) for xy in XY],
                color="red", sizes=[5.0 for _ in XY])
    [ax.vlines(xy[0], min(min(y1), min(y2)), max(f(xy[0]), l(xy[0])),
               color="blue", alpha=0.2, linestyle="--") for xy in XY]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    plt.show()


printGraph(dF, dL2)
printGraph(dF, dL4)
printGraph(ddF, ddL)








