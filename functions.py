import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
from numpy import e as e_num
from numpy import sqrt
from numpy import exp
from numpy import pi
from numpy import cos
from numpy import sin
from math import log


def num_1(x):
    x1 = x[0]
    x2 = x[1]
    a = ((-1.275 * (x1 ** 2) / 3.14 ** 2) + (5 * x1 / 3.14) + x2 - 6) ** 2
    b = ((10 - 5 / (4 * 3.14)) * np.cos(x1) * np.cos(x1))
    c = np.log1p(x1**2 + x2**2 + 1)
    return a + b + c + 10


def plot_3d(func: Callable[[np.array], float], borders: List[int]):
    x = np.linspace(borders[0], borders[1], 100)
    y = np.linspace(borders[0], borders[1], 100)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, func([X, Y]))
    plt.show()


# plot_3d(lambda x: num_1(x), [-15, 15])