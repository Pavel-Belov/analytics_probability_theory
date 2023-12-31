import numpy as np
from scipy import stats

""" 1.
Известно, что генеральная совокупность распределена нормально 
со средним квадратическим отклонением, равным 16.
Найти доверительный интервал для оценки математического ожидания a 
с надежностью 0.95, если выборочная средняя M = 80, а объем выборки n = 256.
"""
n = 256
M = 80
sigma = 16
z_table = stats.norm.ppf(0.975)
print("1.")
print('Доверительный интервал для оценки математического ожидания:')
print([M - z_table * (sigma / n ** 0.5), M + z_table * (sigma / n ** 0.5)])

""" 2.
В результате 10 независимых измерений некоторой величины X, 
выполненных с одинаковой точностью, получены опытные данные:
6.9, 6.1, 6.2, 6.8, 7.5, 6.3, 6.4, 6.9, 6.7, 6.1
Предполагая, что результаты измерений подчинены нормальному закону распределения вероятностей, 
оценить истинное значение величины X при помощи доверительного интервала, 
покрывающего это значение с доверительной вероятностью 0,95.
"""
measures = np.array([6.9, 6.1, 6.2, 6.8, 7.5, 6.3, 6.4, 6.9, 6.7, 6.1])
x_mean = np.mean(measures)
D = np.var(measures, ddof=1)
t = stats.t.ppf(0.975, 9)
print("\n2.")
print('Оценка истинного значения величины X при помощи доверительного интервала:')
print([x_mean - t * np.sqrt(D / 10), x_mean + t * np.sqrt(D / 10)])

""" 3.
Рост дочерей 175, 167, 154, 174, 178, 148, 160, 167, 169, 170
Рост матерей  178, 165, 165, 173, 168, 155, 160, 164, 178, 175
Используя эти данные построить 95% доверительный интервал 
для разности среднего роста родителей и детей.
"""
growth_of_daughters = np.array([175, 167, 154, 174, 178, 148, 160, 167, 169, 170])
growth_of_mothers = np.array([178, 165, 165, 173, 168, 155, 160, 164, 178, 175])
x_1 = np.mean(growth_of_daughters)
x_2 = np.mean(growth_of_mothers)
delta = np.abs(x_1 - x_2)
D1 = np.var(growth_of_daughters, ddof=1)
D2 = np.var(growth_of_mothers, ddof=1)
D = (D1 + D2) / 2
se = np.sqrt(D / len(growth_of_daughters) + D / len(growth_of_mothers))
t = stats.t.ppf(0.975, 2 * (len(growth_of_daughters) - 1))
print("\n3.")
print('95% доверительный интервал для разности среднего роста родителей и детей:')
print([delta - t * se, delta + t * se])
