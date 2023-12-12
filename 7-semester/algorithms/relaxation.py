import numpy as np
from numpy.typing import NDArray
from definitions import p, q, f
from utils import fill_boundary_grid, matrix_norm, get_expected_solution_grid


def get_omega(radius: np.float64) -> np.float64:
    return 2 / (1 + np.sqrt(1 - radius**2))


def upper_relaxation_method(
    x_vec: NDArray,
    y_vec: NDArray,
    x_step: np.float64,
    y_step: np.float64,
    iter: int,
    omega: np.float64,
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
                f_ = f(x_vec[i], y_vec[j])
                p1 = (
                    p(x_vec[i] + x_step / 2, y_vec[j]) * (u_prev[i + 1][j] - u_prev[i][j])
                ) / x_step**2
                p2 = (
                    p(x_vec[i] - x_step / 2, y_vec[j]) * (u_prev[i][j] - u_cur[i - 1][j])
                ) / x_step**2
                q1 = (
                    q(x_vec[i], y_vec[j] + y_step / 2) * (u_prev[i][j + 1] - u_prev[i][j])
                ) / y_step**2
                q2 = (
                    q(x_vec[i], y_vec[j] - y_step / 2) * (u_prev[i][j] - u_cur[i][j - 1])
                ) / y_step**2

                d_ = (
                    (p(x_vec[i] - x_step / 2, y_vec[j]) / x_step**2)
                    + (p(x_vec[i] + x_step / 2, y_vec[j]) / x_step**2)
                    + (q(x_vec[i], y_vec[j] - y_step / 2) / y_step**2)
                    + (q(x_vec[i], y_vec[j] + y_step / 2) / y_step**2)
                )

                u_cur[i][j] = (
                    u_prev[i][j]
                    + (omega * (f_ + p1 - p2 + q1 - q2)) / d_
                )
        k += 1

        u_prev = np.copy(u_cur)
        u_list.append(np.copy(u_cur))

    return u_list, k
