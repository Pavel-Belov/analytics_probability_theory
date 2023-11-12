import matplotlib.pyplot as plt
import numpy as np

""" 1.
Даны значения величины заработной платы заемщиков банка (zp)
и значения их поведенческого кредитного скоринга (ks):
zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832].
Используя математические операции, посчитать коэффициенты линейной регрессии,
приняв за X заработную плату (то есть, zp - признак),
а за y - значения скорингового балла (то есть, ks - целевая переменная).
Произвести расчет как с использованием intercept, так и без.
"""
zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# с использованием intercept
b1_i = (np.mean(zp * ks) - np.mean(zp) * np.mean(ks)) / (np.mean(zp ** 2) - np.mean(zp) ** 2)
b0_i = np.mean(ks) - b1_i * np.mean(zp)
print("1.")
print("с использованием intercept")
print("b0:", b0_i, "b1:", b1_i)
R = np.corrcoef(zp, ks)[0, 1] ** 2
print(R)
plt.scatter(zp, ks)
plt.plot(zp, b0_i + b1_i * zp)
plt.show()

# без использованиея intercept
b1_no_i = np.mean(zp) * np.mean(ks) / np.mean(zp ** 2)
print("без использованиея intercept")
print("b1:", b1_no_i)
plt.scatter(zp, ks)
plt.plot(zp, b1_no_i * zp)
plt.show()

""" 2.
Посчитать коэффициент линейной регрессии при заработной плате (zp), 
используя градиентный спуск (без intercept).
"""


def mse_(B1, X1, Y1, n):
    return np.sum((B1 * X1 - Y1) ** 2) / n


print("\n2.")
print("B1 ожидаемое:", b1_no_i)
print("градиентный спуск (без intercept)")
alpha = 5e-9
print("alpha:", alpha)
B1 = 0.1
n = len(zp)
for i in range(150000):
    B1 -= alpha * (2 / n) * np.sum((B1 * zp - ks) * zp)
    if not i % 10000:
        print("итерация:", i, "B1:", B1, "mse:", mse_(B1, zp, ks, len(zp)))

""" 3.
Произвести вычисления как в пункте 2, но с вычислением intercept. 
Учесть, что изменение коэффициентов должно производиться на каждом шаге одновременно 
(то есть изменение одного коэффициента не должно влиять на изменение другого во время одной итерации).
"""


def mse_(B0, B1, X1, Y1, n):
    return np.sum(B0 + B1 * X1 - Y1) ** 2 / n


print("\n3.")
print("B0 ожидаемое:", b0_i, "B1 ожидаемое:", b1_i)
print("градиентный спуск с вычислением intercept")
alpha = 2e-5
B1 = 0.1
B0 = 0.1
n = len(zp)
for i in range(2 * 1000 ** 2 + 1):
    B1 -= alpha * (2 / n) * np.sum((B0 + B1 * zp - ks) * zp)
    B0 -= alpha * (2 / n) * np.sum(B0 + B1 * zp - ks)
    if not i % 200000:
        print("итерация:", i, "B0", B0, "B1:", B1, "mse:", mse_(B0, B1, zp, ks, len(zp)))
