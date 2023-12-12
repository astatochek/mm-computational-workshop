import numpy as np
from numpy.typing import NDArray
from definitions import p, q, f
from utils import fill_boundary_grid, matrix_norm, get_exact_solution


def get_omega_opt(spec: np.float64) -> np.float64:
    return 2 / (1 + np.sqrt(1 - spec**2))


def upper_relaxation_method(
    x: NDArray,
    y: NDArray,
    step_x: np.float64,
    step_y: np.float64,
    iter: int,
    omega: np.float64,
    eps: np.float64,
):
    k = 0
    u_prev = fill_boundary_grid(x, y)
    u_cur = fill_boundary_grid(x, y)
    u_0 = np.copy(u_prev)
    u_k_list = [u_0]
    exact = get_exact_solution(x, y)

    stop_condition = (
        lambda u_cur, u_0, exact, eps: matrix_norm(u_cur - exact)
        / matrix_norm(u_0 - exact)
        > eps
    )
    # while k < iter:
    while stop_condition(u_cur, u_0, exact, eps):
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                delta_x_sq = step_x**2
                delta_y_sq = step_y**2

                term1 = f(x[i], y[j])
                term2 = (
                    p(x[i] + step_x / 2, y[j]) * (u_prev[i + 1][j] - u_prev[i][j])
                ) / delta_x_sq
                term3 = (
                    p(x[i] - step_x / 2, y[j]) * (u_prev[i][j] - u_cur[i - 1][j])
                ) / delta_x_sq
                term4 = (
                    q(x[i], y[j] + step_y / 2) * (u_prev[i][j + 1] - u_prev[i][j])
                ) / delta_y_sq
                term5 = (
                    q(x[i], y[j] - step_y / 2) * (u_prev[i][j] - u_cur[i][j - 1])
                ) / delta_y_sq

                denominator = (
                    (p(x[i] - step_x / 2, y[j]) / delta_x_sq)
                    + (p(x[i] + step_x / 2, y[j]) / delta_x_sq)
                    + (q(x[i], y[j] - step_y / 2) / delta_y_sq)
                    + (q(x[i], y[j] + step_y / 2) / delta_y_sq)
                )

                u_cur[i][j] = (
                    u_prev[i][j]
                    + (omega * (term1 + term2 - term3 + term4 - term5)) / denominator
                )
        k += 1

        u_prev = np.copy(u_cur)
        u_k_list.append(np.copy(u_cur))

    return u_k_list, k
