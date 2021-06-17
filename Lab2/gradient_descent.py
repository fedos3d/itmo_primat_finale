# coding=utf-8
from functions import *

e = 0.9

class step:
    const = 0
    crash = 1

# метод градиентного спуска
def gradient_descent(func, st, x_start, y_start, a, b, it_cnt, eps):
    x, y = x_start, y_start
    x_next, y_next = x, y
    points_x = []
    points_y = []
    lamb = eps
    for i in range(it_cnt):
        f = func(x, y)
        points_x.append(x_next)
        points_y.append(y_next)
        grad_x = (func(x + H, y) - func(x - H, y)) / (2 * H)
        grad_y = (func(x, y + H) - func(x, y - H)) / (2 * H)
        grad_norm = grad_x ** 2 + grad_y ** 2
        x_next = x - lamb * grad_x
        y_next = y - lamb * grad_y
        f_next = func(x_next, y_next)
        if st == step.crash:
            while f_next > f - e * lamb * grad_norm:
                lamb *= e
        if abs(x - x_next) < eps and abs(y - y_next) < eps and grad_x < eps and grad_y < eps and abs(f - f_next) < eps:
            break
        x = x_next
        y = y_next
    print(i)
    draw(a, b, func, points_x, points_y, 'метод градиентного спуска')
    return f