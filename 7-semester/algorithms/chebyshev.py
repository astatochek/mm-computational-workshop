import numpy as np
from numpy.typing import NDArray
from utils import fill_boundary_grid, get_exact_solution, matrix_norm
from definitions import p, q, f


def calc_sigma(
    c1: np.float64,
    c2: np.float64,
    d1: np.float64,
    d2: np.float64,
    step_x: np.float64,
    step_y: np.float64,
) -> np.float64:
    res = c1 * (4 / (step_x**2)) * ((np.sin((np.pi * step_x) / 2)) ** 2) + d1 * (
        4 / (step_y**2)
    ) * ((np.sin((np.pi * step_y) / (2 * np.pi))) ** 2)
    return res


def calc_delta(
    c1: np.float64,
    c2: np.float64,
    d1: np.float64,
    d2: np.float64,
    step_x: np.float64,
    step_y: np.float64,
) -> np.float64:
    res = c2 * (4 / (step_x**2)) * ((np.cos((np.pi * step_x) / 2)) ** 2) + d2 * (
        4 / (step_y**2)
    ) * ((np.cos((np.pi * step_y) / (2 * np.pi))) ** 2)
    return res


def calc_tau(sigma: np.float64, delta: np.float64, k: int) -> list[np.float64]:
    # teta = [1, 31, 15, 17, 7, 25, 9, 23, 3, 29, 13, 19, 5, 27, 11, 21]
    # teta = [1, 3]
    teta = [1, 15, 7, 9, 3, 13, 5, 11]
    n = 8
    tau_k = []
    for i in range(len(teta)):
        term = np.cos(((teta[i] * np.pi) / (2 * n)))
        tau_k.append(2 / (delta + sigma + (delta - sigma) * term))
    return tau_k


def chebyshev_method(
    x: NDArray,
    y: NDArray,
    step_x: np.float64,
    step_y: np.float64,
    iter: int,
    tau_k: list[np.float64],
    eps: np.float64,
):
    k = 0
    u_prev = fill_boundary_grid(x, y)
    u_cur = fill_boundary_grid(x, y)
    u_0 = np.copy(u_prev)
    u_k_list = [u_0]
    exact = get_exact_solution(x, y)

    should_continue = (
        lambda u_cur, exact, eps: matrix_norm(u_cur - exact) > eps
    )
    # while k < iter:
    while should_continue(u_cur, exact, eps):
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                delta_x_sq = step_x**2
                delta_y_sq = step_y**2

                term1 = (
                    p(x[i] + step_x / 2, y[j]) * (u_prev[i + 1][j] - u_prev[i][j])
                ) / delta_x_sq
                term2 = (
                    p(x[i] - step_x / 2, y[j]) * (u_prev[i][j] - u_prev[i - 1][j])
                ) / delta_x_sq
                term3 = (
                    q(x[i], y[j] + step_y / 2) * (u_prev[i][j + 1] - u_prev[i][j])
                ) / delta_y_sq
                term4 = (
                    q(x[i], y[j] - step_y / 2) * (u_prev[i][j] - u_prev[i][j - 1])
                ) / delta_y_sq
                term5 = f(x[i], y[j])

                u_cur[i][j] = u_prev[i][j] + (
                    tau_k[k % 8] * (term1 - term2 + term3 - term4 + term5)
                )
        k += 1

        u_prev = np.copy(u_cur)
        u_k_list.append(np.copy(u_cur))

    return u_k_list, k
