# coding=utf-8
from functions import *


# метод ньютона
def newton(func, x_start, y_start, a, b, it_cnt, eps):
    x, y = x_start, y_start
    x_next, y_next = x, y
    points_x = []
    points_y = []
    for i in range(it_cnt):
        f = func(x, y)
        points_x.append(x_next)
        points_y.append(y_next)
        x_next = x - (func(x + H, y) - func(x - H, y)) / (2 * H) / ((func(x + 2 * H, y) - 2 * func(x, y) + func(x - 2 * H, y)) / (4 * H ** 2))
        y_next = y - (func(x, y + H) - func(x, y - H)) / (2 * H) / ((func(x, y + 2 * H) - 2 * func(x, y) + func(x, y - 2 * H)) / (4 * H ** 2))
        f_next = func(x_next, y_next)
        if abs(x - x_next) < eps and abs(y - y_next) < eps and abs(f - f_next) < eps:
            break
        x, y = x_next, y_next
    print(i)
    draw(a, b, func, points_x, points_y, 'метод ньютона')
    return f