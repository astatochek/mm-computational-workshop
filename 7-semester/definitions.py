import numpy as np


def p(x, y):
    return 1 + x / 2

def q(x, y):
    return 1

def mu(x, y):
    return x * y**2 * (1 + y)

def f(x, y):
    dx = y**2 * (y + 1) / 2
    dy = 2 * (3*x*y + x)
    return -(dx + dy)


