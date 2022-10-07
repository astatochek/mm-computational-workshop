import pandas as pd
import numpy as np
from functions import promptPair, promptM, promptN, promptFloat, promptE


def F(x: float) -> float:
    return np.cos(x)


def Lagrange(hublist: list):

    def Omega(x: float, xj: float):
        return np.prod([xi[0] - x for xi in hublist if xi[0] != xj])

    def func(x: float):
        return sum([xy[1] * Omega(x, xy[0]) / Omega(xy[0], xy[0]) for xy in hublist])

    return func


def getF(f, val: float):
    def func(x: float) -> float:
        return f(x) - val
    return func


def getSections(f, section: list):
    left = section[0]
    right = section[1]
    res = []
    num = 100
    pr = left
    for _ in range(num):
        nx = pr + (right - left) / num
        if f(nx) * f(pr) <= 0:
            res.append([pr, nx])
        pr = nx
    return res


def Secants(f, segment: list, ep: float):
    x0 = segment[0]
    x1 = segment[1]
    xp = x0
    xc = x1
    xn = xc - (f(xc) / (f(xc) - f(xp))) * (xc - xp)
    while abs(xn - xc) >= ep:
        xn, xc, xp = xn - (f(xn) / (f(xn) - f(xc))) * (xn - xc), xn, xc
    return xn


def isMonotonic(f, segment: list):
    left = segment[0]
    right = segment[1]
    num = 100
    pr = left
    nx = pr + (right - left) / num
    if f(pr) > f(nx):
        flag = True
    else:
        flag = False
    for _ in range(num-1):
        nx = pr + (right - left) / num
        if (flag and f(pr) > f(nx)) or (not flag and f(pr) <= f(nx)):
            pr = nx
        else:
            return False
    return True


a, b = promptPair()

m = promptM()

h = (b - a) / m
ZYj = []
for i in range(m+1):
    ZYj.append([a + i * h, F(a + i * h)])

data = {'Xi': [zy[0] for zy in ZYj], 'F(Xi)': [zy[1] for zy in ZYj]}
df = pd.DataFrame(data)
print(df)

while True:

    Y = promptFloat()


    def cmp(val):
        return abs(val[0] - Y)


    XYj = [zy for zy in ZYj]

    if isMonotonic(F, [a, b]):
        #                                          1st Method
        n = promptN(m, True)
        YXj = [[zy[1], zy[0]] for zy in ZYj]
        YXj.sort(key=cmp)
        YXj = YXj[:n + 1]

        L = Lagrange(YXj)
        X = L(Y)
        print("Method 1:")
        print(f"L({Y}) = {X}")
        print(f"|F({X}) - {Y}| = {abs(F(X) - Y)}")

        #                                          2nd Method
        n = promptN(m, False)

        XYj.sort(key=cmp)
        XYj = XYj[:n + 1]
    else:
        n = m

    e = promptE()

    L = Lagrange(XYj)
    Func = getF(L, Y)
    Sections = getSections(Func, [a, b])
    Results = [Secants(Func, sec, e) for sec in Sections]

    print("Method 2:")
    data = {'X': Results, '|F(X) - Y|': [abs(F(x) - Y) for x in Results]}
    # print(f"X: {Results[0]}")
    df = pd.DataFrame(data)
    print(df)










