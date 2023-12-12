import numpy as np
from numpy.typing import NDArray
from typing import List
from utils import (
    fill_boundary_grid,
    get_expected_solution_grid,
    matrix_norm,
    calc_l_u,
    get_f_grid,
)
from definitions import p, q


def alternational_triangular_iterational_method(
    x_vec: NDArray,
    y_vec: NDArray,
    x_step: np.float64,
    y_step: np.float64,
    omega: np.float64,
    tau: np.float64,
    eps: np.float64,
):
    f_grid = get_f_grid(x_vec, y_vec)
    k = 0
    kappa_1 = omega / x_step**2
    kappa_2 = omega / y_step**2
    u_prev = fill_boundary_grid(x_vec, y_vec)
    u_cur = fill_boundary_grid(x_vec, y_vec)
    u_0 = np.copy(u_prev)
    u_list = [u_0]
    expected = get_expected_solution_grid(x_vec, y_vec)

    should_continue = (
        lambda u_cur: matrix_norm(u_cur - expected) > eps
    )
    # while k < iter:
    while should_continue(u_cur):
        u_cap = np.zeros((len(x_vec), len(y_vec)))
        u_k = np.zeros((len(x_vec), len(y_vec)))
        
        F = calc_l_u(u_prev, x_vec, y_vec, x_step, y_step) + f_grid

        for i in range(1, len(x_vec) - 1):
            for j in range(1, len(y_vec) - 1):
                k1 = kappa_1 * p(x_vec[i] - x_step / 2, y_vec[j]) * u_cap[i - 1][j]
                k2 = kappa_2 * q(x_vec[i], y_vec[j] - y_step / 2) * u_cap[i][j - 1]
                f_ = F[i][j]
                d_ = (
                    1
                    + kappa_1 * p(x_vec[i] - x_step / 2, y_vec[j])
                    + kappa_2 * q(x_vec[i], y_vec[j] - y_step / 2)
                )
                u_cap[i][j] = (k1 + k2 + f_) / d_

        for i in range(len(x_vec) - 2, 0, -1):
            for j in range(len(y_vec) - 2, 0, -1):
                k1 = kappa_1 * p(x_vec[i] + x_step / 2, y_vec[j]) * u_k[i + 1][j]
                k2 = kappa_2 * q(x_vec[i], y_vec[j] + y_step / 2) * u_k[i][j + 1]
                f_ = u_cap[i][j]
                d_ = (
                    1
                    + kappa_1 * p(x_vec[i] + x_step / 2, y_vec[j])
                    + kappa_2 * q(x_vec[i], y_vec[j] + y_step / 2)
                )
                u_k[i][j] = (k1 + k2 + f_) / d_
        u_cur = np.copy(u_prev + tau * u_k)
        u_prev = np.copy(u_cur)
        u_list.append(np.copy(u_cur))
        k += 1

    return u_list, k
