import numpy as np
from numpy.typing import NDArray
from definitions import p, q, f
from utils import fill_boundary_grid, get_expected_solution_grid, matrix_norm


def simple_iteration_method(
    x_vec: NDArray,
    y_vec: NDArray,
    step_x: np.float64,
    step_y: np.float64,
    iter: int,
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
                    p(x_vec[i] - step_x / 2, y_vec[j]) * u_prev[i - 1][j]
                ) / step_x**2
                p2 = (
                    p(x_vec[i] + step_x / 2, y_vec[j]) * u_prev[i + 1][j]
                ) / step_x**2
                q1 = (
                    q(x_vec[i], y_vec[j] - step_y / 2) * u_prev[i][j - 1]
                ) / step_y**2
                q2 = (
                    q(x_vec[i], y_vec[j] + step_y / 2) * u_prev[i][j + 1]
                ) / step_y**2
                p1_d = (p(x_vec[i] - step_x / 2, y_vec[j])) / step_x**2
                p2_d = (p(x_vec[i] + step_x / 2, y_vec[j])) / step_x**2
                q1_d = (q(x_vec[i], y_vec[j] - step_y / 2)) / step_y**2
                q2_d = (q(x_vec[i], y_vec[j] + step_y / 2)) / step_y**2
                u_cur[i][j] = (p1 + p2 + q1 + q2 + f(x_vec[i], y_vec[j])) / (
                    p1_d + p2_d + q1_d + q2_d
                )
        k += 1

        u_prev = np.copy(u_cur)
        u_list.append(np.copy(u_cur))

    return u_list, k
