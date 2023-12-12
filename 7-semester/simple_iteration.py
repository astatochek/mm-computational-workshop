import numpy as np
from numpy.typing import NDArray
from definitions import p, q, f
from utils import fill_boundary_grid, get_exact_solution, matrix_norm


def simple_iteration_method(x_vec: NDArray, y_vec: NDArray, step_x: np.float64, step_y: np.float64, iter: int, eps: np.float64):
    k = 0
    u_prev = fill_boundary_grid(x_vec, y_vec)
    u_cur = fill_boundary_grid(x_vec, y_vec)
    u_0 = np.copy(u_prev)
    u_k_list = [u_0]
    exact = get_exact_solution(x_vec, y_vec)

    stop_condition = lambda u_cur, u_0, exact, eps: matrix_norm(u_cur - exact) / matrix_norm(u_0 - exact) > eps
    # while k < iter:
    while stop_condition(u_cur, u_0, exact, eps):
        for i in range(1, len(x_vec) - 1):
            for j in range(1, len(y_vec) - 1):
                a = (p(x_vec[i] - step_x / 2, y_vec[j]) * u_prev[i - 1][j]) / step_x**2
                b = (p(x_vec[i] + step_x / 2, y_vec[j]) * u_prev[i + 1][j]) / step_x**2
                c = (q(x_vec[i], y_vec[j] - step_y / 2) * u_prev[i][j - 1]) / step_y**2
                d = (q(x_vec[i], y_vec[j] + step_y / 2) * u_prev[i][j + 1]) / step_y**2
                a_d = (p(x_vec[i] - step_x / 2, y_vec[j])) / step_x**2
                b_d = (p(x_vec[i] + step_x / 2, y_vec[j])) / step_x**2
                c_d = (q(x_vec[i], y_vec[j] - step_y / 2)) / step_y**2
                d_d = (q(x_vec[i], y_vec[j] + step_y / 2)) / step_y**2
                u_cur[i][j] = (a + b + c + d + f(x_vec[i], y_vec[j])) / (a_d + b_d + c_d + d_d)
        k += 1

        u_prev = np.copy(u_cur)
        u_k_list.append(np.copy(u_cur))

    return u_k_list, k