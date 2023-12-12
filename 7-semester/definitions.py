import numpy as np


def p(x: np.float64, y: np.float64) -> float:
    return 1. + x / 2


def q(x: np.float64, y: np.float64) -> float:
    return 1.


def mu(x: np.float64, y: np.float64) -> float:
    return x * y ** 2 * (1 + y)


def f(x: np.float64, y: np.float64) -> float:
    dx = y ** 2 * (y + 1) / 2
    dy = 2 * (3 * x * y + x)
    return -(dx + dy)
