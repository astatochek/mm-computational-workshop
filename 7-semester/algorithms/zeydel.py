import numpy as np
from numpy.typing import NDArray
from utils import fill_boundary_grid, get_exact_solution, matrix_norm
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
    u_k_list = [u_0]
    exact = get_exact_solution(x_vec, y_vec)

    should_continue = (
        lambda u_cur, exact, eps: matrix_norm(u_cur - exact) > eps
    )
    # while k < iter:
    while should_continue(u_cur, exact, eps):
        for i in range(1, len(x_vec) - 1):
            for j in range(1, len(y_vec) - 1):
                delta_x_sq = x_step**2
                delta_y_sq = y_step**2

                term1 = (
                    p(x_vec[i] - x_step / 2, y_vec[j]) * u_cur[i - 1][j]
                ) / delta_x_sq
                term2 = (
                    p(x_vec[i] + x_step / 2, y_vec[j]) * u_prev[i + 1][j]
                ) / delta_x_sq
                term3 = (
                    q(x_vec[i], y_vec[j] - y_step / 2) * u_cur[i][j - 1]
                ) / delta_y_sq
                term4 = (
                    q(x_vec[i], y_vec[j] + y_step / 2) * u_prev[i][j + 1]
                ) / delta_y_sq
                term5 = f(x_vec[i], y_vec[j])

                denominator = (
                    (p(x_vec[i] - x_step / 2, y_vec[j]) / delta_x_sq)
                    + (p(x_vec[i] + x_step / 2, y_vec[j]) / delta_x_sq)
                    + (q(x_vec[i], y_vec[j] - y_step / 2) / delta_y_sq)
                    + (q(x_vec[i], y_vec[j] + y_step / 2) / delta_y_sq)
                )

                u_cur[i][j] = (term1 + term2 + term3 + term4 + term5) / denominator
        k += 1

        u_prev = np.copy(u_cur)
        u_k_list.append(np.copy(u_cur))

    return u_k_list, k
