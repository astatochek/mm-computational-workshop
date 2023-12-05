import numpy as np
import pandas as pd
from typing import Callable
from progonka import progonka

l_x = 1
l_y = 1
p = lambda x, y: np.log(2 + x)
q = lambda x, y: np.log(2 + y)
mu = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
c_1, c_2 = np.log(2), np.log(3)
d_1, d_2 = np.log(2), np.log(3)


def f(x, y):
    dx = np.pi * np.cos(np.pi * y) * (np.cos(np.pi * x) - np.pi * (x + 2) * np.log(x + 2) * np.sin(x * np.pi)) / (x + 2)
    dy = -1 * np.sin(np.pi * x) * (np.sin(np.pi * y) + np.pi * (y + 2) * np.log(y + 2) * np.cos(np.pi * y)) / (y + 2)
    return -1 * (dx + dy)

def alternating_directions_method(N_x, N_y):
    h_x = l_x / N_x
    h_y = l_y / N_y
    x = lambda i: 0. + h_x * i
    y = lambda j: 0. + h_y * j
    get_p_i_plus_half_j = lambda i, j: p(x(i) + h_x / 2, y(j))
    get_p_i_minus_half_j = lambda i, j: p(x(i) - h_x / 2, y(j))
    get_q_i_j_plus_half = lambda i, j: q(x(i), y(j) + h_y / 2)
    get_q_i_j_minus_half = lambda i, j: q(x(i), y(j) - h_y / 2)

    def lambda_1(u: Callable, i, j):
        return get_p_i_plus_half_j(i, j) * (u(i + 1, j) - u(i, j)) / (h_x ** 2) - get_p_i_minus_half_j(i, j) * (
                u(i, j) - u(i - 1, j)) / (h_x ** 2)

    def lambda_2(u: Callable, i, j):
        return get_q_i_j_plus_half(i, j) * (u(i, j + 1) - u(i, j)) / (h_y ** 2) - get_q_i_j_minus_half(i, j) * (
                u(i, j) - u(i, j - 1)) / (h_y ** 2)

    def get_tau():
        delta_x_1 = c_1 * 4 / (h_x ** 2) * np.sin(np.pi * h_x / (2 * l_x)) ** 2
        delta_x_2 = c_2 * 4 / (h_x ** 2) * np.cos(np.pi * h_x / (2 * l_x)) ** 2
        delta_y_1 = d_1 * 4 / (h_y ** 2) * np.sin(np.pi * h_y / (2 * l_y)) ** 2
        delta_y_2 = d_2 * 4 / (h_y ** 2) * np.cos(np.pi * h_y / (2 * l_y)) ** 2

        delta_1 = np.min([delta_x_1, delta_y_1])
        delta_2 = np.min([delta_x_2, delta_y_2])

        return 2 / np.sqrt(delta_1 * delta_2)

    def get_A_x(i, j, tau):
        return get_p_i_minus_half_j(i, j) * tau / (2 * h_x ** 2)

    def get_B_x(i, j, tau):
        return 1 + tau * (get_p_i_minus_half_j(i, j) + get_p_i_plus_half_j(i, j)) / (2 * h_x ** 2)

    def get_C_x(i, j, tau):
        return get_p_i_plus_half_j(i, j) * tau / (2 * h_x ** 2)

    def get_A_y(i, j, tau):
        return get_q_i_j_minus_half(i, j) * tau / (2 * h_y ** 2)

    def get_B_y(i, j, tau):
        return 1 + tau * (get_q_i_j_minus_half(i, j) + get_q_i_j_plus_half(i, j)) / (2 * h_y ** 2)

    def get_C_y(i, j, tau):
        return get_q_i_j_plus_half(i, j) * tau / (2 * h_y ** 2)

    tau = get_tau()
    U = np.zeros((N_x + 1, N_y + 1))
    u = lambda i, j: U[i, j]

    for i in range(N_x + 1):
        for j in range(N_y + 1):
            if i == 0 or j == 0 or i == N_x or j == N_y:
                U[i][j] = mu(x(i), y(j))

    print(pd.DataFrame(U))

    for _ in range(100):
        for j in range(1, N_y):
            A = np.zeros(N_x + 1)
            B = np.zeros(N_x + 1)
            C = np.zeros(N_x + 1)
            G = np.zeros(N_x + 1)
            B[0] = -1
            B[N_x] = -1
            G[0] = mu(x(0), y(j))
            G[N_x] = mu(x(N_x), y(j))

            for i in range(1, N_x):
                G[i] = -U[i][j] - tau / 2 * (lambda_2(u, i, j) + f(x(i), y(j)))
                A[i] = get_A_x(i, j, tau)
                B[i] = get_B_x(i, j, tau)
                C[i] = get_C_x(i, j, tau)

            U_j = progonka(A, B, C, G)

            for i in range(N_x + 1):
                U[i][j] = U_j[i]

        for i in range(1, N_x):
            A = np.zeros(N_y + 1)
            B = np.zeros(N_y + 1)
            C = np.zeros(N_y + 1)
            G = np.zeros(N_y + 1)
            B[0] = -1
            B[N_y] = -1
            G[0] = mu(x(i), y(0))
            G[N_y] = mu(x(i), y(N_y))

            for j in range(1, N_y):
                G[j] = -U[i][j] - tau / 2 * (lambda_1(u, i, j) + f(x(i), y(j)))
                A[j] = get_A_y(i, j, tau)
                B[j] = get_B_y(i, j, tau)
                C[j] = get_C_y(i, j, tau)

            U_i = progonka(A, B, C, G)

            for j in range(N_y + 1):
                U[i][j] = U_i[j]

    print(pd.DataFrame(U))
    print(pd.DataFrame(np.array([
        [
            mu(x(i), y(j)) for j in range(N_y+1)
        ] for i in range(N_x+1)
    ])))


alternating_directions_method(10, 10)
