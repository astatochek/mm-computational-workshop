import numpy as np
from latexifier import latexify


class Matrix(np.ndarray):
    """extends np.ndarray"""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def inverse_rows(self, i: int, j: int):
        """Swaps i-th and j-th rows of a matrix"""
        self[i], self[j] = self[j].copy(), self[i].copy()

    def inverse_cols(self, i: int, j: int):
        """Swaps i-th and j-th columns of a matrix"""
        self = self.transpose()
        self.inverse_rows(i, j)
        self = self.transpose()
    
    def tex(self) -> str:
        """Latexifies the matrix rounding up values up to 6 decimal values"""
        return latexify(self)

    def norm(self) -> float:
        """
        Calculates Matrix norm using np.linalg.norm
        
        ## Returns
        Matrix norm using formulae max(sum(abs(x), axis=0)) either as `string` or a `float`
        """
        return np.linalg.norm(self, 1)


def Gauss(A: Matrix, B: Matrix, n: int) -> Matrix:
    """
    Вычисляет решения линейного уравнения Ax = B методом Гаусса с выбором
    главного элемента по столбцам (с перестановкой строк)

    ## Parameters
    `A`: матрица коэффициентов размерности `n`x`n`
    `B`: столбец свободных членов размерности `n`x`1`
    `n`: размерность

    ## Returns
    `X`: столбец, такой что AX ≈ B
    """
    AB = np.append(A, B)
    X = [[0] for _ in range(n)]
    for k in range(n):
        j = k
        for i in range(k+1, n):
            if abs(AB[j][k]) < abs(AB[i][k]):
                j = i
        AB.inverse_rows(k, j)
        for j in range(k+1, n):
            c = AB[j][k] / AB[k][k]
            for i in range(n+1):
                AB[j][i] -= c * AB[k][i]
    X[n-1][0] = AB[n-1][n] / AB[n-1][n-1]
    for k in range(n-1, -1, -1):
        s = 0
        for i in range(k+1, n):
            s += AB[k][i] * X[i][0]
        X[k][0] = (AB[k][n] - s) / AB[k][k]
    return Matrix(np.array(X))