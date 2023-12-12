import numpy as np
from numpy.typing import NDArray
from typing import List
from utils import (
    fill_boundary_grid,
    get_exact_solution,
    matrix_norm,
    calculate_lhu,
    get_f_grid,
)
from definitions import p, q


def triangle_method(
    x_vec: NDArray,
    y_vec: NDArray,
    x_step: np.float64,
    y_step: np.float64,
    omega: np.float64,
    tau: List[np.float64],
    eps: np.float64,
):
    f_h = get_f_grid(x_vec, y_vec)
    k = 0
    kappa_1 = omega / x_step**2
    kappa_2 = omega / y_step**2
    u_prev = fill_boundary_grid(x_vec, y_vec)
    u_cur = fill_boundary_grid(x_vec, y_vec)
    u_0 = np.copy(u_prev)
    u_k_list = [u_0]
    exact = get_exact_solution(x_vec, y_vec)

    stop_condition = (
        lambda u_cur, u_0, exact, eps: matrix_norm(u_cur - exact)
        / matrix_norm(u_0 - exact)
        > eps
    )
    # while k < iter:
    while stop_condition(u_cur, u_0, exact, eps):
        lower_w = np.zeros((len(x_vec), len(y_vec)))
        upper_w = np.zeros((len(x_vec), len(y_vec)))
        
        F = calculate_lhu(u_prev, x_vec, y_vec, x_step, y_step) + f_h

        for i in range(1, len(x_vec) - 1):
            for j in range(1, len(y_vec) - 1):
                term1 = kappa_1 * p(x_vec[i] - x_step / 2, y_vec[j]) * lower_w[i - 1][j]
                term2 = kappa_2 * q(x_vec[i], y_vec[j] - y_step / 2) * lower_w[i][j - 1]
                term3 = F[i][j]
                denominator = (
                    1
                    + kappa_1 * p(x_vec[i] - x_step / 2, y_vec[j])
                    + kappa_2 * q(x_vec[i], y_vec[j] - y_step / 2)
                )
                lower_w[i][j] = (term1 + term2 + term3) / denominator

        for i in range(len(x_vec) - 2, 0, -1):
            for j in range(len(y_vec) - 2, 0, -1):
                term1 = kappa_1 * p(x_vec[i] + x_step / 2, y_vec[j]) * upper_w[i + 1][j]
                term2 = kappa_2 * q(x_vec[i], y_vec[j] + y_step / 2) * upper_w[i][j + 1]
                term3 = lower_w[i][j]
                denominator = (
                    1
                    + kappa_1 * p(x_vec[i] + x_step / 2, y_vec[j])
                    + kappa_2 * q(x_vec[i], y_vec[j] + y_step / 2)
                )
                upper_w[i][j] = (term1 + term2 + term3) / denominator
        u_cur = np.copy(u_prev + tau * upper_w)
        u_prev = np.copy(u_cur)
        u_k_list.append(np.copy(u_cur))
        k += 1

    return u_k_list, k
