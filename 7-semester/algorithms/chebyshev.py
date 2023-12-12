import numpy as np
from numpy.typing import NDArray
from typing import List
from utils import fill_boundary_grid, get_expected_solution_grid, matrix_norm
from definitions import p, q, f


def calc_delta_1(
    c1: np.float64,
    d1: np.float64,
    step_x: np.float64,
    step_y: np.float64,
) -> np.float64:
    delta1 = c1 * (4 / (step_x**2)) * ((np.sin((np.pi * step_x) / 2)) ** 2) + d1 * (
        4 / (step_y**2)
    ) * ((np.sin((np.pi * step_y) / (2 * np.pi))) ** 2)
    return delta1


def calc_delta_2(
    c2: np.float64,
    d2: np.float64,
    step_x: np.float64,
    step_y: np.float64,
) -> np.float64:
    delta2 = c2 * (4 / (step_x**2)) * ((np.cos((np.pi * step_x) / 2)) ** 2) + d2 * (
        4 / (step_y**2)
    ) * ((np.cos((np.pi * step_y) / (2 * np.pi))) ** 2)
    return delta2


def calc_tau_list(delta1: np.float64, delta2: np.float64, k: int) -> list[np.float64]:
    # sigma_list = [1, 31, 15, 17, 7, 25, 9, 23, 3, 29, 13, 19, 5, 27, 11, 21]
    sigma_list = [1, 15, 7, 9, 3, 13, 5, 11]
    # sigma_list = [1, 7, 3, 5]
    # sigma_list = [1, 3]
    n = len(sigma_list)
    tau_list = []
    for sigma in sigma_list:
        tau_list.append(
            2
            / (
                delta2
                + delta1
                + (delta2 - delta1) * np.cos(((sigma * np.pi) / (2 * n)))
            )
        )
    return tau_list


def method_with_chebyshev_parameters(
    x_vec: NDArray,
    y_vec: NDArray,
    step_x: np.float64,
    step_y: np.float64,
    iter: int,
    tau_list: List[np.float64],
    eps: np.float64,
):
    k = 0
    u_prev = fill_boundary_grid(x_vec, y_vec)
    u_cur = fill_boundary_grid(x_vec, y_vec)
    u_0 = np.copy(u_prev)
    u_list = [u_0]
    expected = get_expected_solution_grid(x_vec, y_vec)

    should_continue = lambda u_cur: matrix_norm(u_cur - expected) > eps
    # while k < iter:
    while should_continue(u_cur):
        for i in range(1, len(x_vec) - 1):
            for j in range(1, len(y_vec) - 1):
                p1 = (
                    p(x_vec[i] + step_x / 2, y_vec[j]) * (u_prev[i + 1][j] - u_prev[i][j])
                ) / step_x**2
                p2 = (
                    p(x_vec[i] - step_x / 2, y_vec[j]) * (u_prev[i][j] - u_prev[i - 1][j])
                ) / step_x**2
                q1 = (
                    q(x_vec[i], y_vec[j] + step_y / 2) * (u_prev[i][j + 1] - u_prev[i][j])
                ) / step_y**2
                q2 = (
                    q(x_vec[i], y_vec[j] - step_y / 2) * (u_prev[i][j] - u_prev[i][j - 1])
                ) / step_y**2
                f_ = f(x_vec[i], y_vec[j])

                u_cur[i][j] = u_prev[i][j] + (
                    tau_list[k % len(tau_list)] * (p1 - p2 + q1 - q2 + f_)
                )
        k += 1

        u_prev = np.copy(u_cur)
        u_list.append(np.copy(u_cur))

    return u_list, k
