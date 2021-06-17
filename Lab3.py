import numpy as np
from scipy.sparse import csr_matrix
import time

def get_LU(A):
    n = A.indptr.size - 1
    L = csr_matrix(([], [], [0] * (n + 1)), shape=(n, n), dtype=np.float64)
    U = csr_matrix(([], [], [0] * (n + 1)), shape=(n, n), dtype=np.float64)
    for i in range(n):
        L[i, i] = 1
    for i in range(n):
        for j in range(n):
            if i <= j:
                sum = 0
                for k in range(i):
                    sum += L[i, k] * U[k, j]
                U[i, j] = A[i, j] - sum
            else:
                sum = 0
                for k in range(j):
                    sum += L[i, k] * U[k, j]
                L[i, j] = (A[i, j] - sum) / U[j, j]
    return L, U


def gauss(A, y):
    # Прямой ход
    for i in range(A.indptr.size - 1):
        temp = A.indptr[i]
        while A.indices[temp] != i:
            temp = temp + 1
        temp = A.data[temp]
        y[i, 0] = y[i, 0] / temp
        for j in range(A.indptr[i], A.indptr[i + 1]):
            A[i, A.indices[j]] = A[i, A.indices[j]] / temp

        for j in range(i + 1, A.indptr.size - 1):
            k2 = A.indptr[j]
            if A.indices[k2] <= i:
                k2 += (i - A.indices[k2])
                parameter = A.data[k2]
                for k in range(A.indptr[i], A.indptr[i + 1]):
                    A[j, A.indices[k]] = A[j, A.indices[k]] - parameter * A[i, A.indices[k]]
                y[j, 0] = y[j, 0] - y[i, 0] * parameter
    #Обратный ход
    for i in range(A.indptr.size - 2, 0, -1):
        for j in range(0, i):
            temp = A.indices[A.indptr[j + 1] - A.indptr.size + 1 + i]
            if temp == i:
                y[j, 0] = float(y[j, 0] - y[i, 0] * A[j, i] * A[temp, i])
                A[j, i] = float(A[j, i] - A[j, i] * A[temp, i])
    return y

def LU_gauss(A, y):
    L, U = get_LU(A)
    # Прямой ход
    iter = 0
    for i in range(L.indptr.size - 1):
        for j in range(i + 1, L.indptr.size - 1):
            k2 = L.indptr[j]
            iter += 1
            if L.indices[k2] <= i:
                k2 += (i - L.indices[k2])
                parameter = L.data[k2]
                y[j, 0] = y[j, 0] - y[i, 0] * parameter
    # Обратный ход
    for i in range(U.indptr.size - 2, -1, -1):
        temp = U.indptr[i]
        while U.indices[temp] != i:
            temp = temp + 1
        temp = U.data[temp]
        y[i, 0] = y[i, 0] / temp
        for j in range(0, i):
            iter += 1
            k = U.indices[U.indptr[j + 1] - U.indptr.size + 1 + i]
            if k == i:
                y[j, 0] = float(y[j, 0] - y[i, 0] * U[j, i])
    # print(iter)
    return y


def jacobi(A, x, y, eps):
    iter = 0
    while 1:
        temp = x.copy()
        par = (A * x - y)
        for i in range(x.indptr.size - 1):
            iter = iter + 1
            x[i, 0] = x[i, 0] - par[i, 0] / A[i, i]
        if np.abs(np.linalg.norm(temp.toarray()) - np.linalg.norm(x.toarray())) < eps:
            break
    # print(iter)
    return x


def generate_gilbert(k):
    A = csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            A[i, j] = 1 / (i + j + 1)
    return A


def generate_big_matrix(k):
    A = csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)
    for i in range(k):
        n = 0
        for j in range(k):
            if i == j:
                A[i, j] = k + 1
            else:
                if n < 3:
                    A[i, j] = 1
                    n += 1
    return A


def condition_number_matrix(n, k):
    A = csr_matrix(([], [], [0] * (n + 1)), shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = (n - 1) + 10 ** (-k)
            else:
                A[i, j] = -1

    return A


def x_matrix(n):
    x = csr_matrix(([], [], [0] * (n + 1)), shape=(n, 1), dtype=np.float64)
    for i in range(n):
        x[i, 0] = i + 1
    return x


def start_x_matrix(n):
    x = csr_matrix(([], [], [0] * (n + 1)), shape=(n, 1), dtype=np.float64)
    for i in range(n):
        x[i, 0] = 0
    return x


def reverse_matrix(A):
    n = A.indptr.size - 1
    rev = csr_matrix(([], [], [0] * (n + 1)), shape=(n, n), dtype=np.float64)
    for i in range(n):
        vector_e = csr_matrix(([], [], [0] * (n + 1)), shape=(n, 1), dtype=np.float64)
        vector_e[i, 0] = 1
        reverse_e = LU_gauss(A.copy(), vector_e.copy())
        for j in range(n):
            rev[j, i] += reverse_e[j, 0]
    return rev

#Нахождение обратной матрицы
# A = generate_big_matrix(4)
# rev = reverse_matrix(A)
# print(rev.toarray())
# print((A * rev).toarray())


# Пункт 4
# n = 20
# for i in range(1, 21):
#     k = i
#     A = condition_number_matrix(n, k)
#     x = x_matrix(n)
#     y = A * x
#     print(np.linalg.norm((jacobi(A, start_x_matrix(n), y, 0.001) - x).toarray()))
#     print(np.linalg.norm((LU_gauss(A, y) - x).toarray()))

# Пункт 5
# for i in range(1, 21):
#     n = i
#     A = generate_gilbert(n)
#     x = x_matrix(n)
#     y = A * x
#     print(np.linalg.norm((LU_gauss(A, y) - x).toarray()))

# Пункт 6
# n = 100000
#
# A = generate_big_matrix(n)
# x = x_matrix(n)
# y = A * x
# start = time.time()
# jacobi(A, start_x_matrix(n), y, 0.001)
# # LU_gauss(A, y)
# print(time.time()-start)