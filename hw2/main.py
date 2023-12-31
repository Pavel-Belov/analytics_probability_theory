from math import factorial, e


def combinations(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))


def bernuli(n, p, k):
    return combinations(n, k) * p ** k * (1 - p) ** (n - k)


def puasson(n, p, k):
    lambd = p * n
    return lambd ** k / factorial(k) * e ** (-lambd)


""" 1.
Вероятность того, что стрелок попадет в мишень, выстрелив один раз, равна 0.8.
Стрелок выстрелил 100 раз. Найдите вероятность того, что стрелок попадет в цель ровно 85 раз.
"""
p = 0.8
n = 100
k = 85
p_k = bernuli(n, p, k)
print('1.\nвероятность того, что стрелок попадет в цель ровно 85 раз\n', p_k)

""" 2. 
Вероятность того, что лампочка перегорит в течение первого дня эксплуатации, равна 0.0004. 
В жилом комплексе после ремонта в один день включили 5000 новых лампочек. 
Какова вероятность, что ни одна из них не перегорит в первый день? 
Какова вероятность, что перегорят ровно две?
"""
p = 0.0004
n = 5000
k1 = 0
k2 = 2

# Пуассон
p_k1_puasson = puasson(n, p, k1)
print('\n2.\nПо Пуассону\nвероятность, что ни одна из ламп не перегорит в первый день\n', p_k1_puasson)
p_k2_puasson = puasson(n, p, k2)
print('вероятность, что перегорят ровно две\n', p_k2_puasson)

# Бернули
p_k1_bernuli = bernuli(n, p, k1)
print('\nПо Бернули\nвероятность, что ни одна из ламп не перегорит в первый день\n', p_k1_bernuli)
p_k2_bernuli = bernuli(n, p, k2)
print('вероятность, что перегорят ровно две\n', p_k2_bernuli)

""" 3. 
Монету подбросили 144 раза. Какова вероятность, что орел выпадет ровно 70 раз?
"""
n = 144
k = 70
p = 0.5
p_k = bernuli(n, p, k)
print('\n3.\nвероятность, что орел выпадет ровно 70 раз\n', p_k)

""" 4. 
В первом ящике находится 10 мячей, из которых 7 - белые. 
Во втором ящике - 11 мячей, из которых 9 белых. 
Из каждого ящика вытаскивают случайным образом по два мяча. 
Какова вероятность того, что все мячи белые? 
Какова вероятность того, что ровно два мяча белые? 
Какова вероятность того, что хотя бы один мяч белый?
"""
# все мячи белые
p_basket1 = combinations(7, 2) / combinations(10, 2)
p_basket2 = combinations(9, 2) / combinations(11, 2)
p1 = p_basket1 * p_basket2
print('\n4.\nвероятность того, что все мячи белые\n', p1)

# ровно два мяча белые
p_basket1_v1 = combinations(7, 2) / combinations(10, 2)  # если оба белых из корзины 1
p_basket2_v1 = combinations(9, 0) / combinations(11, 2)  # значит, из корзины 2 оба не белые

p_basket1_v2 = combinations(7, 0) / combinations(10, 2)  # если оба белых из корзины 2
p_basket2_v2 = combinations(9, 2) / combinations(11, 2)  # значит, из корзины 1 оба не белые

p2 = (p_basket1_v1 * p_basket2_v1) + (p_basket1_v2 * p_basket2_v2)
print('\nвероятность того, что ровно два мяча белые\n', p2)

# хотя бы один мяч белый
p_basket1 = combinations(7, 0) / combinations(10, 2)
p_basket2 = combinations(9, 0) / combinations(11, 2)
p3 = 1 - (p_basket1 * p_basket2)
print('\nвероятность того, что хотя бы один мяч белый\n', p3)
