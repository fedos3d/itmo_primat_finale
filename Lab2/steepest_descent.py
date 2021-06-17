# coding=utf-8
from functions import *


# метод наискорейшего спуска
def steepest_descent(func, methopt, x_start, y_start, a, b, it_cnt, eps):
    grad_x, grad_y = 0, 0
    x, y = x_start, y_start
    x_next, y_next = x, y
    points_x = []
    points_y = []
    for i in range(it_cnt):
        f = func(x, y)
        points_x.append(x_next)
        points_y.append(y_next)
        grad_x = (func(x + H, y) - func(x - H, y)) / (2 * H)
        grad_y = (func(x, y + H) - func(x, y - H)) / (2 * H)
        lamb = methopt(func, x, y, grad_x, grad_y, a, b, eps)
        x_next = x - lamb * grad_x
        y_next = y - lamb * grad_y
        f_next = func(x_next, y_next)
        if abs(x - x_next) < eps and abs(y - y_next) < eps and grad_x < eps and grad_y < eps and abs(f - f_next) < eps:
            break
        x = x_next
        y = y_next
    print(i)
    draw(a, b, func, points_x, points_y, 'метод наискорейшего спуска')
    return f
