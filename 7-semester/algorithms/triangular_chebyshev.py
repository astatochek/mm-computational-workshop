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


def cheba_coef(gamma1: float, gamma2: float) -> list[float]:
    teta = [1, 15, 7, 9, 3, 13, 5, 11]
    n = len(teta)
    tau_k = []
    for i in range(len(teta)):
        term = np.cos((teta[i] / (2 * n)) * np.pi)
        tau_k.append(2 / (gamma2 + gamma1 + (gamma2 - gamma1) * term))
    return tau_k


def triangle_method(
    x: np.ndarray,
    y: np.ndarray,
    h_x: float,
    h_y: float,
    omega: float,
    f_h: np.ndarray,
    tau: list[float],
    eps: float,
):
    k = 0
    kappa_1 = omega / h_x**2
    kappa_2 = omega / h_y**2
    previous_U = fill_boundary_grid(x, y)
    current_U = fill_boundary_grid(x, y)
    U_0 = np.copy(previous_U)
    arrayOfU_K = [U_0]
    exact = get_exact_solution(x, y)

    while matrix_norm(current_U - exact) / matrix_norm(U_0 - exact) > eps:
        lower_w = np.zeros((len(x), len(y)))
        upper_w = np.zeros((len(x), len(y)))
        F = calculate_lhu(previous_U, x_i, y_i, step_x, step_y) + f_h
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                term1 = kappa_1 * p(x[i] - h_x / 2, y[j]) * lower_w[i - 1][j]
                term2 = kappa_2 * q(x[i], y[j] - h_y / 2) * lower_w[i][j - 1]
                term3 = F[i][j]
                denominator = (
                    1
                    + kappa_1 * p(x[i] - h_x / 2, y[j])
                    + kappa_2 * q(x[i], y[j] - h_y / 2)
                )
                lower_w[i][j] = (term1 + term2 + term3) / denominator
        for i in range(len(x) - 2, 0, -1):
            for j in range(len(y) - 2, 0, -1):
                term1 = kappa_1 * p(x[i] + h_x / 2, y[j]) * upper_w[i + 1][j]
                term2 = kappa_2 * q(x[i], y[j] + h_y / 2) * upper_w[i][j + 1]
                term3 = lower_w[i][j]
                denominator = (
                    1
                    + kappa_1 * p(x[i] + h_x / 2, y[j])
                    + kappa_2 * q(x[i], y[j] + h_y / 2)
                )
                upper_w[i][j] = (term1 + term2 + term3) / denominator
        current_U = np.copy(previous_U + tau[k % len(tau)] * upper_w)
        previous_U = np.copy(current_U)
        arrayOfU_K.append(np.copy(current_U))
        k += 1
    return arrayOfU_K, k
