import numpy as np
from utils import fill_boundary_grid, get_exact_solution, matrix_norm
from definitions import p, q, f


def get_tau(c1: float, c2: float, d1: float, d2: float, step_x: float, step_y: float) -> float:
    sigma = c1 * (4 / (step_x**2)) * ((np.sin((np.pi * step_x) / 2)) ** 2) + d1 * (4 / (step_y**2)) * ((np.sin((np.pi * step_y) / (2 * np.pi))) ** 2)
    delta = c2 * (4 / (step_x**2)) * ((np.cos((np.pi * step_x) / 2)) ** 2) + d2 * (4 / (step_y**2)) * ((np.cos((np.pi * step_y) / (2 * np.pi))) ** 2)
    return 2 / (delta + sigma)

def iteration_optimal(x: np.ndarray, y: np.ndarray, step_x: float, step_y: float, iter: int, tau: float, eps: float):
    k = 0
    previous_U = fill_boundary_grid(x, y)
    current_U = fill_boundary_grid(x, y)
    U_0 = np.copy(previous_U)
    arrayOfU_K = [U_0]
    exact = get_exact_solution(x, y)
    # while k < iter:
    while (matrix_norm(current_U - exact) / matrix_norm(U_0 - exact) > eps):
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                delta_x_sq = step_x**2
                delta_y_sq = step_y**2

                term1 = (
                    p(x[i] + step_x / 2, y[j])
                    * (previous_U[i + 1][j] - previous_U[i][j])
                    / delta_x_sq
                )
                term2 = (
                    p(x[i] - step_x / 2, y[j])
                    * (previous_U[i][j] - previous_U[i - 1][j])
                    / delta_x_sq
                )
                term3 = (
                    q(x[i], y[j] + step_y / 2)
                    * (previous_U[i][j + 1] - previous_U[i][j])
                    / delta_y_sq
                )
                term4 = (
                    q(x[i], y[j] - step_y / 2)
                    * (previous_U[i][j] - previous_U[i][j - 1])
                    / delta_y_sq
                )
                term5 = f(x[i], y[j])

                current_U[i][j] = previous_U[i][j] + tau * (
                    term1 - term2 + term3 - term4 + term5
                )
        k += 1

        previous_U = np.copy(current_U)
        arrayOfU_K.append(np.copy(current_U))
    return arrayOfU_K, k
