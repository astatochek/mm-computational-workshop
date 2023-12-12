import numpy as np
from numpy.typing import NDArray
from definitions import mu, f, p, q


def calc_ksi(c1: np.float64, c2: np.float64, d1: np.float64, d2: np.float64, x_step: np.float64,
             y_step: np.float64) -> np.float64:
    delta1 = c1 * (4 / (x_step ** 2)) * ((np.sin((np.pi * x_step) / 2)) ** 2) + d1 * (4 / (y_step ** 2)) * (
                (np.sin((np.pi * y_step) / (2 * np.pi))) ** 2)
    delta2 = c2 * (4 / (x_step ** 2)) * ((np.cos((np.pi * x_step) / 2)) ** 2) + d2 * (4 / (y_step ** 2)) * (
                (np.cos((np.pi * y_step) / (2 * np.pi))) ** 2)
    return delta1 / delta2


def spectral_radius_approx(ksi: np.float64) -> float:
    return (1 - ksi) / (1 + ksi)


def num_iteration_approx(ksi: np.float64, eps: np.float64) -> int:
    return int(np.log(1 / eps) / (2 * ksi))


def fill_boundary_grid(x_vec, y_vec):
    N, M = len(x_vec), len(y_vec)

    grid = np.zeros((N, M))
    for i, val in enumerate(y_vec):
        grid[0][i] = mu(0, val)
        grid[N - 1][i] = mu(x_vec[N - 1], val)

    for i, val in enumerate(x_vec):
        grid[i][0] = mu(val, 0)
        grid[i][M - 1] = mu(val, y_vec[M - 1])

    return grid


def matrix_norm(matrix: NDArray) -> np.float64:
    return np.max(np.abs(matrix.flatten()))


def get_expected_solution_grid(x_vec: NDArray, y_vec: NDArray) -> NDArray:
    exact = np.zeros((len(x_vec), len(y_vec)))
    for i in range(len(x_vec)):
        for j in range(len(y_vec)):
            exact[i][j] = mu(x_vec[i], y_vec[j])
    return exact


def get_f_grid(x_vec: NDArray, y_vec: NDArray) -> NDArray:
    grid = np.zeros((len(x_vec), len(y_vec)))
    for i in range(len(x_vec)):
        for j in range(len(y_vec)):
            grid[i][j] = f(x_vec[i], y_vec[j])
    return grid


def calc_l_u(u: NDArray, x_vec: NDArray, y_vec: NDArray, x_step: np.float64, y_step: np.float64) -> NDArray:
    l_u = np.zeros((len(x_vec), len(y_vec)))
    for i in range(1, len(x_vec) - 1):
        for j in range(1, len(y_vec) - 1):
            p1 = p(x_vec[i] + x_step / 2, y_vec[j]) * (u[i + 1][j] - u[i][j]) / x_step ** 2
            p2 = p(x_vec[i] - x_step / 2, y_vec[j]) * (u[i][j] - u[i - 1][j]) / x_step ** 2
            q1 = q(x_vec[i], y_vec[j] + y_step / 2) * (u[i][j + 1] - u[i][j]) / y_step ** 2
            q2 = q(x_vec[i], y_vec[j] - y_step / 2) * (u[i][j] - u[i][j - 1]) / y_step ** 2
            l_u[i][j] = p1 - p2 + q1 - q2
    return l_u
