import numpy as np
from numpy.typing import NDArray
from utils import fill_boundary_grid, get_expected_solution_grid, matrix_norm
from definitions import p, q, f


def get_tau(
    c1: np.float64,
    c2: np.float64,
    d1: np.float64,
    d2: np.float64,
    step_x: np.float64,
    step_y: np.float64,
) -> np.float64:
    delta1 = c1 * (4 / (step_x**2)) * ((np.sin((np.pi * step_x) / 2)) ** 2) + d1 * (
        4 / (step_y**2)
    ) * ((np.sin((np.pi * step_y) / (2 * np.pi))) ** 2)
    delta2 = c2 * (4 / (step_x**2)) * ((np.cos((np.pi * step_x) / 2)) ** 2) + d2 * (
        4 / (step_y**2)
    ) * ((np.cos((np.pi * step_y) / (2 * np.pi))) ** 2)
    return 2 / (delta2 + delta1)


def iteration_with_optimal_parameter(
    x_vec: NDArray,
    y_vec: NDArray,
    x_step: np.float64,
    y_step: np.float64,
    iter: int,
    tau: np.float64,
    eps: np.float64,
):
    k = 0
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
        for i in range(1, len(x_vec) - 1):
            for j in range(1, len(y_vec) - 1):
                p1 = (
                    p(x_vec[i] + x_step / 2, y_vec[j])
                    * (u_prev[i + 1][j] - u_prev[i][j])
                    / x_step**2
                )
                p2 = (
                    p(x_vec[i] - x_step / 2, y_vec[j])
                    * (u_prev[i][j] - u_prev[i - 1][j])
                    / x_step**2
                )
                p3 = (
                    q(x_vec[i], y_vec[j] + y_step / 2)
                    * (u_prev[i][j + 1] - u_prev[i][j])
                    / y_step**2
                )
                p4 = (
                    q(x_vec[i], y_vec[j] - y_step / 2)
                    * (u_prev[i][j] - u_prev[i][j - 1])
                    / y_step**2
                )
                f_ = f(x_vec[i], y_vec[j])

                u_cur[i][j] = u_prev[i][j] + tau * (
                    p1 - p2 + p3 - p4 + f_
                )
        k += 1

        u_prev = np.copy(u_cur)
        u_list.append(np.copy(u_cur))

    return u_list, k
