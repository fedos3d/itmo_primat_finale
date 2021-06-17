import math
import matplotlib.pyplot as plt
import numpy as np

H = 10 ** (-7)


def func1(x, y):
    return x ** 2 + 3 * y ** 2 - 2 * x * y + 1


def dichotomy_method(func, x, y, grad_x, grad_y, a, b, eps):
    while (b - a) / 2 > eps:
        m = (a + b) / 2
        f1 = func(x - grad_x * (m - eps / 2), y - grad_y * (m - eps / 2))
        f2 = func(x - grad_x * (m + eps / 2), y - grad_y * (m + eps / 2))
        if f1 > f2:
            a = m - eps / 2
        else:
            b = m + eps / 2
    return (a + b) / 2


def golden_ratio_method(func, x, y, grad_x, grad_y, a, b, eps):
    phi = (math.sqrt(5) + 1) / 2
    x1 = a + (b - a) / (phi + 1)
    x2 = b - (b - a) / (phi + 1)
    f1 = func(x - grad_x * x1, y - grad_y * x1)
    f2 = func(x - grad_x * x2, y - grad_y * x2)
    while b - a > eps:
        if f1 >= f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - (b - a) / (phi + 1)
            f2 = func(x - grad_x * x2, y - grad_y * x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (b - a) / (phi + 1)
            f1 = func(x - grad_x * x1, y - grad_y * x1)
    return (a + b) / 2


def fibonacci(n):
    temp1 = ((1 + math.sqrt(5)) / 2) ** n
    temp2 = ((1 - math.sqrt(5)) / 2) ** n
    return int((temp1 - temp2) / math.sqrt(5))


def fibonacci_method(func, x, y, grad_x, grad_y, a, b, eps):
    n = 0
    while fibonacci(n) <= (b - a) / eps:
        n += 1
    x1 = a + (b - a) * fibonacci(n - 2) / fibonacci(n)
    x2 = a + (b - a) * fibonacci(n - 1) / fibonacci(n)
    f1 = func(x - grad_x * x1, y - grad_y * x1)
    f2 = func(x - grad_x * x2, y - grad_y * x2)
    while n > 1:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b + a - x1
            f2 = func(x - grad_x * x2, y - grad_y * x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b + a - x2
            f1 = func(x - grad_x * x1, y - grad_y * x1)
        n = n - 1
    return (x1 + x2) / 2


def draw(a, b, func, points_x, points_y, name):
    fig, ax = plt.subplots()
    x, y = np.mgrid[a:b:100j, a:b:100j]
    ax.set_title(name)
    ax.contour(x, y, func(x, y), levels=100, colors='b')
    for i in range(len(points_x)):
        ax.scatter(points_x[i], points_y[i], c='y')
    ax.plot([points_x[i] for i in range(len(points_x))], [points_y[i] for i in range(len(points_y))], c='y')
    plt.show()


def draw_func(func, a, b):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = np.mgrid[a:b:100j, a:b:100j]
    ax.contour(x, y, func(x, y), levels=100, colors='b')
    plt.show()