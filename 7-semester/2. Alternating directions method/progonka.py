import numpy as np

def progonka(a, b, c, g):
    n = len(g)
    s = np.zeros(n)
    t = np.zeros(n)
    s[0] = c[0] / b[0]
    t[0] = -g[0] / b[0]

    for i in range(1, n - 1):
        s[i] = c[i] / (b[i] - s[i - 1] * a[i])
        t[i] = (a[i] * t[i - 1] - g[i]) / (b[i] - a[i] * s[i - 1])
    t[n - 1] = (a[n - 1] * t[n - 2] - g[n - 1]) / (b[n - 1] - a[n - 1] * s[n - 2])
    y = np.zeros(n)
    y[-1] = t[-1]
    for i in range(n - 2, -1, -1):
        y[i] = s[i] * y[i + 1] + t[i]

    return y
