import numpy as np
from numpy.typing import NDArray
from utils import fill_boundary_grid, get_expected_solution_grid, matrix_norm
from definitions import p, q, f


def zeydel_method(
    x_vec: NDArray,
    y_vec: NDArray,
    x_step: np.float64,
    y_step: np.float64,
    iter: int,
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
                    p(x_vec[i] - x_step / 2, y_vec[j]) * u_cur[i - 1][j]
                ) / x_step**2
                p2 = (
                    p(x_vec[i] + x_step / 2, y_vec[j]) * u_prev[i + 1][j]
                ) / x_step**2
                q1 = (
                    q(x_vec[i], y_vec[j] - y_step / 2) * u_cur[i][j - 1]
                ) / y_step**2
                q2 = (
                    q(x_vec[i], y_vec[j] + y_step / 2) * u_prev[i][j + 1]
                ) / y_step**2
                f_ = f(x_vec[i], y_vec[j])

                d_ = (
                    (p(x_vec[i] - x_step / 2, y_vec[j]) / x_step**2)
                    + (p(x_vec[i] + x_step / 2, y_vec[j]) / x_step**2)
                    + (q(x_vec[i], y_vec[j] - y_step / 2) / y_step**2)
                    + (q(x_vec[i], y_vec[j] + y_step / 2) / y_step**2)
                )

                u_cur[i][j] = (p1 + p2 + q1 + q2 + f_) / d_
        k += 1

        u_prev = np.copy(u_cur)
        u_list.append(np.copy(u_cur))

    return u_list, k
