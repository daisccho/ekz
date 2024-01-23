import time

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List
from functions import num_1
from matplotlib import pylab
from mpl_toolkits import mplot3d


def temp_xin_yao(T0: int, n: int, i: int, m: int = 5, p: int = 0.2):
    c = m * np.exp(-(p / n))
    T = T0 * np.exp(-1 * c * i ** (1 / n))
    return T


def temp_boltzmann(T0: int, i: int):
    return T0 / np.log(1 + i)


def probability(Fxi: float, Fopt: float, T: int):
    return np.exp(-((Fxi - Fopt) / T))


def xin_yao_algorithm(func: Callable[[np.array], float], x_start: List[float], borders: List[int], T0: int, Tmin: int,
                      Imax: int,
                      Pmax: int = 10, Pmin: int = 0):
    # Инициализация
    dots = [x_start]  # Список для отслеживания точек
    n = len(x_start)
    Xopt = np.zeros(n)  # Начальная точка оптимума
    Fopt = func(Xopt)  # Значение функции в начальной точке
    i = 0
    T = T0

    # Шаг 1: Выбор случайной начальной точки
    x = np.random.uniform(borders[0], borders[1], n)
    Fx = func(x)
    if Fx < Fopt:
        Fopt = Fx
        Xopt = x
        dots.append(Xopt)

    # Шаг 5: Основной цикл
    while T > Tmin and i < Imax:
        # Шаг 2: Генерация новой точки
        i += 1
        a = np.random.uniform(0, 1, 2)
        s = np.sign(a - 0.5)
        z = s * T * (((1 + (1 / T)) ** abs(2 * a - 1)) - 1)
        x_new = x + (Pmax - Pmin) * z

        # Проверка, чтобы новая точка оставалась в пределах границ
        x_new = np.clip(x_new, borders[0], borders[1])

        # Шаг 3: Оценка функции в новой точке
        Fxi = func(x_new)

        # Шаг 4: Принятие или отклонение новой точки
        if Fxi < Fopt:
            Xopt = x_new
            Fopt = Fxi
            dots.append(Xopt)
        else:
            A, B = np.random.uniform(0, 1, 2)
            P = probability(Fxi, Fopt, T)
            if P > A:
                Xopt = x_new
                Fopt = Fxi
                dots.append(Xopt)

        # Обновление температуры
        T = temp_xin_yao(T0, n, i)

    # Возвращение оптимальной точки и списка точек
    return Xopt, dots


def boltzmann_annealing(func: Callable[[np.array], float], x_start: List[float], borders: List[int], T0: int, Tmin: int,
                        Imax: int):
    # Инициализация
    dots = [x_start]  # Список для отслеживания точек
    n = len(x_start)

    Xopt = np.zeros(n)
    Fopt = func(x_start)  # Значение функции в начальной точке
    i = 0
    z = 0
    T = T0
    Ftest = []
    # Шаг 1: Генерация случайной начальной точки
    x = np.random.uniform(borders[0], borders[1], n)
    Fx = func(x)
    if Fx < Fopt:
        Fopt = Fx
        Xopt = x
        z += 1
        Ftest.append(Fopt)
        dots.append(Xopt)

    # Шаг 5: Основной цикл
    while T > Tmin and i < Imax:
        # Шаг 2: Генерация новой точки
        i += 1
        N = np.random.standard_normal(n)
        x_new = x + T * N

        # Проверка, чтобы новая точка оставалась в пределах границ
        # while any(x_new[0] < borders[0][0] or x_new[0] > borders[0][1] or x_new[1] < borders[1][0] or x_new[1] > borders[1][1] for j in range(n)):
        while any(x_new[j] > borders[1] or x_new[j] < borders[0] for j in range(n)):
            N = np.random.standard_normal(n)
            x_new = x + T * N

        # Шаг 3: Оценка функции в новой точке
        Fxi = func(x_new)

        # Шаг 4: Принятие или отклонение новой точки
        if Fxi < Fopt:
            Xopt = x_new
            Fopt = Fxi
            Ftest.append(Fopt)
            dots.append(Xopt)
            z += 1
        else:
            A = np.random.uniform(0, 1, 1)
            P = probability(Fxi, Fopt, T)
            if P > A:
                Xopt = x_new
                Fopt = Fxi
                z += 1
                Ftest.append(Fopt)
                dots.append(Xopt)

        # Обновление температуры
        T = temp_boltzmann(T0, i)
    l = 0
    testing = []
    z = min(Ftest)
    l = Ftest.index(z)
    testing.append(dots[l + 1][0])
    testing.append(dots[l + 1][1])
    dots.pop(l + 1)
    dots.append(testing)

    # Возвращение оптимальной точки и списка точек
    return testing, dots


def error(prompt):
    print(prompt)
    exit()


def func_choise(num):
    #if num == 1:
        #return lambda x: styblinski_tang(x)
    if num == 2:
        return lambda x: num_1(x)
    #if num == 3:
        #return lambda x: cross_in_tray(x)


def plot(func: Callable[[np.array], float], borders: List[int], mass: List[np.array]):
    xx, yy = pylab.meshgrid(
        pylab.linspace(borders[0], borders[1], 500),
        pylab.linspace(borders[0], borders[1], 500))

    zz = pylab.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = func([xx[i, j], yy[i, j]])

    pylab.pcolor(xx, yy, zz)

    pylab.colorbar()

    for i in range(0, len(mass)):
        if i == 0:
            pylab.plot(mass[i][0], mass[i][1], 'go')
        elif i == len(mass) - 1:
            pylab.plot(mass[i][0], mass[i][1], 'ro')
        else:
            pylab.plot(mass[i][0], mass[i][1], 'bx')

    pylab.show()


def plot_3d(func: Callable[[np.array], float], borders: List[int]):
    x = np.linspace(borders[0], borders[1], 100)
    y = np.linspace(borders[0], borders[1], 100)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, func([X, Y]))
    plt.show()


def main():
    print("""Выберите функцию:
            1. Функция Стыбинского-Танга (n=2)                    |
            2. Функция МакКормика                                 |  
            3. Функция "крест на подносе"                         |""")

    function_number = int(input("Номер: "))
    if function_number < 1 or function_number > 3:
        error("Нет такой функции")

    function = func_choise(function_number)

    tochka = [0, 0]
    print("Введите точку:")
    tochka[0] = float(input("x[0] = "))
    tochka[1] = float(input("x[1] = "))

    print(tochka)

    if function_number == 1:
        borders = [-5, 5]
    if function_number == 2:
        borders = [-15, 15]
    if function_number == 3:
        borders = [-10, 10]

    print("""Выберите метод:
            1. Алгоритм Ксин Яо              |
            2. Больцмановский отжиг          |""")

    method = int(input("Номер: "))
    if method == 1:
        print("Введите параметры:")
        T0 = int(input("Начальная температура: "))
        Tmin = int(input("Минимальная температура: "))
        Imax = int(input("Максимальное число итераций: "))
        print("\n")

        result, mass = xin_yao_algorithm(lambda x: function(x), tochka, borders, T0, Tmin, Imax)

        print("Точка оптимума: ")
        print("x = ", result[0])
        print("y = ", result[1])
        print("Значение функции в точке: ")
        print("f(x,y) = ", function(result), "\n")

        plot_3d(function, borders)

        plot(function, borders, mass)

        print("Желаете повторить с теми же параметрами?")
        answ = int(input("Да - 1, нет - 0: "))
        while answ == 1:
            result, mass = xin_yao_algorithm(lambda x: function(x), tochka, borders, T0, Tmin, Imax)

            print("Точка оптимума: ")
            print("x = ", result[0])
            print("y = ", result[1])
            print("Значение функции в точке: ")
            print("f(x,y) = ", function(result), "\n")

            #plot_3d(function, borders)

            plot(function, borders, mass)

            print("Желаете повторить с теми же параметрами?")
            answ = int(input("Да - 1, нет - 0: "))

    if method == 2:
        print("Введите параметры:")
        T0 = int(input("Начальная температура: "))
        Tmin = int(input("Минимальная температура: "))
        Imax = int(input("Максимальное число итераций: "))
        print("\n")

        result, mass = boltzmann_annealing(lambda x: function(x), tochka, borders, T0, Tmin, Imax)

        print("Точка оптимума: ")
        print("x = ", result[0])
        print("y = ", result[1])
        print("Значение функции в точке: ")
        print("f(x,y) = ", function(result), "\n")

        #plot_3d(function, borders)

        plot(function, borders, mass)

        print("Желаете повторить с теми же параметрами?")
        answ = int(input("Да - 1, нет - 0: "))
        while answ == 1:
            result, mass = boltzmann_annealing(lambda x: function(x), tochka, borders, T0, Tmin, Imax)

            print("Точка оптимума: ")
            print("x = ", result[0])
            print("y = ", result[1])
            print("Значение функции в точке: ")
            print("f(x,y) = ", function(result), "\n")

            plot_3d(function, borders)

            plot(function, borders, mass)

            print("Желаете повторить с теми же параметрами?")
            answ = int(input("Да - 1, нет - 0: "))


main()
continuation = 0
print("Желаете протестировать новую функцию?")
continuation = int(input("Да - 1, нет - 0: "))
while continuation == 1:
    main()
    continuation = 0
    print("Желаете протестировать новую функцию?")
    continuation = int(input("Да - 1, нет - 0: "))

print("\nДо связи")