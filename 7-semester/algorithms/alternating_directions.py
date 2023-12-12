import numpy as np
from numpy.typing import NDArray
from typing import List
from utils import (
    fill_boundary_grid,
    get_expected_solution_grid,
    matrix_norm,
)
from definitions import p, q, f, mu


def A_1(
    x_vec: NDArray, y_vec: NDArray, i: int, j: int, x_step: np.float64
) -> np.float64:
    return (p(x_vec[i] - (x_step / 2), y_vec[j])) / x_step**2


def B_1(
    x_vec: NDArray, y_vec: NDArray, i: int, j: int, x_step: np.float64, tau: np.float64
) -> np.float64:
    term1 = 2 / tau
    term2 = p(x_vec[i] + (x_step / 2), y_vec[j]) / x_step**2
    term3 = p(x_vec[i] - (x_step / 2), y_vec[j]) / x_step**2
    return term1 + term2 + term3


def C_1(
    x_vec: NDArray, y_vec: NDArray, i: int, j: int, x_step: np.float64
) -> np.float64:
    return (p(x_vec[i] + (x_step / 2), y_vec[j])) / x_step**2


def G_1(
    x_vec: NDArray,
    y_vec: NDArray,
    i: int,
    j: int,
    y_step: np.float64,
    tau: np.float64,
    u: NDArray,
) -> np.float64:
    term1 = (2 * u[i][j]) / tau
    term2 = (
        q(x_vec[i], y_vec[j] + (y_step / 2)) * (u[i][j + 1] - u[i][j])
    ) / y_step**2
    term3 = (
        q(x_vec[i], y_vec[j] - (y_step / 2)) * (u[i][j] - u[i][j - 1])
    ) / y_step**2
    term4 = f(x_vec[i], y_vec[j])
    return -(term1 + term2 - term3 + term4)


def A_2(
    x_vec: NDArray, y_vec: NDArray, i: int, j: int, y_step: np.float64
) -> np.float64:
    return (q(x_vec[i], y_vec[j] - (y_step / 2))) / y_step**2


def B_2(
    x_vec: NDArray, y_vec: NDArray, i: int, j: int, y_step: np.float64, tau: np.float64
) -> np.float64:
    term1 = 2 / tau
    term2 = q(x_vec[i], y_vec[j] + (y_step / 2)) / y_step**2
    term3 = q(x_vec[i], y_vec[j] - (y_step / 2)) / y_step**2
    return term1 + term2 + term3


def C_2(
    x_vec: NDArray, y_vec: NDArray, i: int, j: int, y_step: np.float64
) -> np.float64:
    return (q(x_vec[i], y_vec[j] + (y_step / 2))) / y_step**2


def G_2(
    x_vec: NDArray,
    y_vec: NDArray,
    i: int,
    j: int,
    x_step: np.float64,
    tau: np.float64,
    u: NDArray,
):
    term1 = (2 * u[i][j]) / tau
    term2 = (
        p(x_vec[i] + (x_step / 2), y_vec[j]) * (u[i + 1][j] - u[i][j])
    ) / x_step**2
    term3 = (
        p(x_vec[i] - (x_step / 2), y_vec[j]) * (u[i][j] - u[i - 1][j])
    ) / x_step**2
    term4 = f(x_vec[i], y_vec[j])
    return -(term1 + term2 - term3 + term4)


def progonka(A: NDArray, B: NDArray, C: NDArray, G: NDArray) -> NDArray:
    n = len(A)
    s = np.zeros(n)
    t = np.zeros(n)
    y = np.zeros(n)
    s[0] = C[0] / B[0]
    t[0] = -G[0] / B[0]
    for i in range(1, n):
        s[i] = C[i] / (B[i] - (A[i] * s[i - 1]))
        t[i] = ((A[i] * t[i - 1]) - G[i]) / (B[i] - (A[i] * s[i - 1]))
    y[n - 1] = t[n - 1]
    for i in range(n - 2, -1, -1):
        y[i] = (s[i] * y[i + 1]) + t[i]
    return y


def alternating_directions_method(
    x_vec: NDArray,
    y_vec: NDArray,
    x_step: np.float64,
    y_step: np.float64,
    tau: np.float64,
    eps: np.float64,
):
    k = 0
    exact = get_expected_solution_grid(x_vec, y_vec)
    u_prev = fill_boundary_grid(x_vec, y_vec)

    u_0 = np.copy(u_prev)
    u_list = [u_0]

    should_continue = (
        lambda u_cur, exact, eps: matrix_norm(u_cur - exact) > eps
    )
    # while k < iter:
    while should_continue(u_list[k], exact, eps):
        u_half = np.zeros((len(x_vec), len(y_vec)))
        u_cur = np.zeros((len(x_vec), len(y_vec)))

        for i in range(len(x_vec)):  # Условие на крайних столбцах
            u_half[i][0] = mu(x_vec[i], 0)
            u_half[i][len(y_vec) - 1] = mu(x_vec[i], y_vec[len(y_vec) - 1])

        for j in range(1, len(y_vec) - 1):
            A, B, C, G = np.zeros(len(x_vec)), np.zeros(len(x_vec)), np.zeros(len(x_vec)), np.zeros(len(x_vec))

            A[0] = 0
            B[0] = -1
            C[0] = 0
            G[0] = mu(0, y_vec[j])

            A[len(x_vec) - 1] = 0
            B[len(x_vec) - 1] = -1
            C[len(x_vec) - 1] = 0
            G[len(x_vec) - 1] = mu(x_vec[len(x_vec) - 1], y_vec[j])

            for i in range(1, len(x_vec) - 1):
                A[i] = A_1(x_vec, y_vec, i, j, x_step)
                B[i] = B_1(x_vec, y_vec, i, j, x_step, tau)
                C[i] = C_1(x_vec, y_vec, i, j, x_step)
                G[i] = G_1(x_vec, y_vec, i, j, y_step, tau, u_list[k])

            solve = progonka(A, B, C, G)
            for i in range(len(x_vec)):
                u_half[i][j] = solve[i]

        for j in range(len(y_vec)):
            u_cur[0][j] = mu(0, y_vec[j])
            u_cur[len(x_vec) - 1][j] = mu(x_vec[len(x_vec) - 1], y_vec[j])

        for i in range(1, len(x_vec) - 1):
            A = np.zeros(len(y_vec))
            B = np.zeros(len(y_vec))
            C = np.zeros(len(y_vec))
            G = np.zeros(len(y_vec))

            A[0] = 0
            B[0] = -1
            C[0] = 0
            G[0] = mu(x_vec[i], 0)

            A[len(y_vec) - 1] = 0
            B[len(y_vec) - 1] = -1
            C[len(y_vec) - 1] = 0
            G[len(y_vec) - 1] = mu(x_vec[i], y_vec[len(y_vec) - 1])

            for j in range(1, len(y_vec) - 1):
                A[j] = A_2(x_vec, y_vec, i, j, y_step)
                B[j] = B_2(x_vec, y_vec, i, j, y_step, tau)
                C[j] = C_2(x_vec, y_vec, i, j, y_step)
                G[j] = G_2(x_vec, y_vec, i, j, x_step, tau, u_half)
            solve = progonka(A, B, C, G)

            for j in range(len(y_vec)):
                u_cur[i][j] = solve[j]

        u_list.append(np.copy(u_cur))
        k += 1

    return u_list, k
