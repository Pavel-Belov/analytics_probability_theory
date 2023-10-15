import numpy as np
from scipy import stats
from math import sqrt

""" 1.
Случайная непрерывная величина A имеет равномерное распределение на промежутке (200, 800].
Найдите ее среднее значение и дисперсию.
"""
a = 200
b = 800
m_x = (a + b) / 2
d = (b - a) ** 2 / 12
print('1.')
print('среднее значение:', m_x)
print('дисперсия', d)

""" 2.
О случайной непрерывной равномерно распределенной величине B известно, что ее дисперсия равна 0.2. 
Можно ли найти правую границу величины B и ее среднее значение зная, что левая граница равна 0.5? 
Если да, найдите ее.
"""
d = 0.2
b = 0.5
a = b - sqrt(12 * d)
m_x = (a + b) / 2
print('\n2.')
print('правая граница:', a)
print('среднее значение', m_x)

""" 3.
Непрерывная случайная величина X распределена нормально и задана плотностью распределения 
f(x) = (1 / (4 * sqrt(2pi))) * exp((-(x+2)**2) / 32)
Найдите:
а). M(X)
б). D(X)
в). std(X) (среднее квадратичное отклонение)
"""
m_x = -2  # из перемнной Эйлера видим, что (x - a)**2? nj то есть a = -2 = M(X)
d = 4  # или sqrt(32 / 2)
d_x = d ** 2
std = np.sqrt(d)
print('\n3.')
print('M(X):', a)
print('D(X)', m_x)
print('std', std)

""" 4.
Рост взрослого населения города X имеет нормальное распределение.
Причем, средний рост равен 174 см, а среднее квадратичное отклонение равно 8 см.
Какова вероятность того, что случайным образом выбранный взрослый человек имеет рост:
а). больше 182 см
б). больше 190 см
в). от 166 см до 190 см
г). от 166 см до 182 см
д). от 158 см до 190 см
е). не выше 150 см или не ниже 190 см
ё). не выше 150 см или не ниже 198 см
ж). ниже 166 см.
"""
print('\n4.\nвероятность того, что случайным образом выбранный взрослый человек имеет рост:')
p1 = 1 - stats.norm.cdf(182, 174, sqrt(8))
print('больше 182 см:', p1)

p2 = 1 - stats.norm.cdf(190, 174, sqrt(8))
print('больше 190 см:', p2)

p3 = stats.norm.cdf(166, 174, sqrt(8)) - stats.norm.cdf(190, 174, sqrt(8))
print('от 166 см до 190 см:', p3)

p4 = stats.norm.cdf(166, 174, sqrt(8)) - stats.norm.cdf(182, 174, sqrt(8))
print('от 166 см до 182 см:', p4)

p5 = stats.norm.cdf(158, 174, sqrt(8)) - stats.norm.cdf(190, 174, sqrt(8))
print('от 158 см до 190 см:', p5)

p6 = stats.norm.cdf(150, 174, sqrt(8)) + (1 - stats.norm.cdf(190, 174, sqrt(8)))
print('не выше 150 см или не ниже 190 см:', p6)

p7 = stats.norm.cdf(150, 174, sqrt(8)) + (1 - stats.norm.cdf(198, 174, sqrt(8)))
print('не выше 150 см или не ниже 198 см:', p7)

p8 = stats.norm.cdf(166, 174, sqrt(8))
print('ниже 166 см:', p8)

""" 5.
На сколько сигм (средних квадратичных отклонений) отклоняется рост человека, равный 190 см, 
от математического ожидания роста в популяции, в которой M(X) = 178 см и D(X) = 25 кв.см?
"""
M_X = 178
d = sqrt(25)
X = 190
Z = (X - M_X) / d
print('\n5.')
print('рост человека, равный 190 см, отклоняется от математического ожидания на:')
print(Z, 'сигмы')
