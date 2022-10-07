from random import uniform as rnd
from math import sin, cos


# from task 10:
# f(x) = 1,2∙x^4+2∙x^3‒13∙x^2‒14,2∙x‒24,1
# [A, B] = [-5, 5]
# Epsilon = 10^(-6)


def f(x):
    return 1.2 * (x ** 4) + 2 * (x ** 3) - 13 * (x ** 2) - 14.2 * x - 24.1


def df(x):
    return 4 * 1.2 * (x ** 3) + 3 * 2 * (x ** 2) - 2 * 13 * x - 14.2


def ddf(x):
    return 3 * 4 * 1.2 * (x ** 2) + 2 * 3 * 2 * x - 2 * 13


def get_A():
    return -5


def get_B():
    return 5


def epsilon():
    return 10 ** (-10)


def get_N():
    return 20


def partition(n):
    segments = []
    a = get_A()
    b = get_B()
    h = (b - a) / n
    prev = a

    for i in range(n):
        cur = a + (i + 1) * h
        if f(cur) * f(prev) < 0:
            segments.append([prev, cur])
        if f(cur) != 0:
            prev = cur

    if len(segments) == 0:
        print(f"Warning: {get_N()} is not enough to determine segments")
        return segments

    return segments


def get_x0_n(segment):
    a = segment[0]
    b = segment[1]
    x0 = rnd(a, b)
    while f(x0) * ddf(x0) <= 0 or df(x0) == 0:
        x0 = rnd(a, b)
    return x0


def get_x(segment):
    a = segment[0]
    b = segment[1]
    x = rnd(a, b)
    while x == a or x == b:
        x = rnd(a, b)
    return x


def PrepareOutput(name, beg, cnt, res, delta, nev):
    print(f"{name}:")
    print(f"   Начальное приближение: {beg}")
    print(f"   Количество шагов: {cnt}")
    print(f"   Приближенное решение: {res}")
    print(f"   Дельта: {delta}")
    print(f"   Величина невязки: {nev}")
