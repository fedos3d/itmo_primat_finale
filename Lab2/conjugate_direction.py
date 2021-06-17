# coding=utf-8
from functions import *


def find_min(func, a, b, x, y, s_x, s_y, eps):
    while b - a > eps:
        f1 = func(x + a * s_x, y + a * s_y)
        f2 = func(x + b * s_x, y + b * s_y)
        c = (a + b) / 2
        if f1 < f2:
            b = c
        else:
            a = c
    return (a + b) / 2


# метод сопряжённых направлений
def conjugate_direction(func, x_start, y_start, a, b, it_cnt, eps):
    x, y = x_start, y_start
    points_x = []
    points_y = []
    s = [[1, 0],
        [0, 1]]
    for i in range(it_cnt):
        points_x.append(x)
        points_y.append(y)
        grad_x = (func(x + H, y) - func(x - H, y)) / (2 * H)
        grad_y = (func(x, y + H) - func(x, y - H)) / (2 * H)
        grad_norm = math.sqrt(grad_x ** 2 + grad_y ** 2)
        if grad_norm < eps:
          break
        lamb = find_min(func, -1000, 1000, x, y, s[0][0], s[0][1], eps)
        x1 = x + lamb * s[0][0]
        y1 = y + lamb * s[0][1]
        lamb = find_min(func, -1000, 1000, x1, y1, s[1][0], s[1][1], eps)
        x2 = x1 + lamb * s[1][0]
        y2 = y1 + lamb * s[1][1]
        lamb = find_min(func, -1000, 1000, x2, y2, s[0][0], s[0][1], eps)
        x3 = x2 + lamb * s[0][0]
        y3 = y2 + lamb * s[0][1]
        s[0][0] = x3 - x1
        s[0][1] = y3 - y1
        x, y = x3, y3
        f = func(x, y)
    print(i)
    draw(a, b, func, points_x, points_y, 'метод сопряжённых направлений')
    return f