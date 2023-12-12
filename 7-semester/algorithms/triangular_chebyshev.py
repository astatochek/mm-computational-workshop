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


def chebyshev_coeffs(gamma_1: np.float64, gamma_2: np.float64) -> list[np.float64]:
    teta = [1, 15, 7, 9, 3, 13, 5, 11]
    n = len(teta)
    tau_k = []
    for i in range(len(teta)):
        term = np.cos((teta[i] / (2 * n)) * np.pi)
        tau_k.append(2 / (gamma_2 + gamma_1 + (gamma_2 - gamma_1) * term))
    return tau_k


def triangular_method_with_chebyshev_parameters(
    x_vec: NDArray,
    y_vec: NDArray,
    x_step: np.float64,
    y_step: np.float64,
    omega: np.float64,
    tau: List[np.float64],
    eps: np.float64,
):
    f_grid = get_f_grid(x_vec, y_vec)
    k = 0
    kappa_1 = omega / x_step**2
    kappa_2 = omega / y_step**2
    u_prev = fill_boundary_grid(x_vec, y_vec)
    u_cur = fill_boundary_grid(x_vec, y_vec)
    u_0 = np.copy(u_prev)
    u_k_list = [u_0]
    exact = get_expected_solution_grid(x_vec, y_vec)

    should_continue = (
        lambda u_cur, exact, eps: matrix_norm(u_cur - exact) > eps
    )
    # while k < iter:
    while should_continue(u_cur, exact, eps):
        lower_w = np.zeros((len(x_vec), len(y_vec)))
        upper_w = np.zeros((len(x_vec), len(y_vec)))
        F = calc_l_u(u_prev, x_vec, y_vec, x_step, y_step) + f_grid
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
        u_cur = np.copy(u_prev + tau[k % len(tau)] * upper_w)
        u_prev = np.copy(u_cur)
        u_k_list.append(np.copy(u_cur))
        k += 1
        
    return u_k_list, k
