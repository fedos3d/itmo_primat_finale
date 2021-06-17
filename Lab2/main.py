from conjugate_direction import *
from steepest_descent import *
from gradient_descent import *
from conjugate_gradient import *
from newton import *
from functions import *

def func(x, y):
    return x ** 2 + 3 * y ** 2 - 2 * x * y + 1

def test(func, x_start, y_start, a, b, it_cnt, eps):
    #draw_func(func, a, b)

    print("Steepest (dich, golden, fibb)")
    print(steepest_descent(func, dichotomy_method, x_start, y_start, a, b, it_cnt, eps))
    print(steepest_descent(func, golden_ratio_method, x_start, y_start, a, b, it_cnt, eps))
    print(steepest_descent(func, fibonacci_method, x_start, y_start, a, b, it_cnt, eps))

    print(gradient_descent(func, step.const, x_start, y_start, a, b, it_cnt, eps))
    print(gradient_descent(func, step.crash, x_start, y_start, a, b, it_cnt, eps))

    print(conjugate_gradient(func, dichotomy_method, x_start, y_start, a, b, it_cnt, eps))
    print(conjugate_gradient(func, golden_ratio_method, x_start, y_start, a, b, it_cnt, eps))
    print(conjugate_gradient(func, fibonacci_method, x_start, y_start, a, b, it_cnt, eps))

    print(conjugate_direction(func, x_start, y_start, a, b, it_cnt, eps))

    print(newton(func, x_start, y_start, a, b, it_cnt, eps))

#test(func, 10, 10, -10, 10, 100, 0.1)
#test(func1, 5, 5, -10, 10, 100, 0.1)
