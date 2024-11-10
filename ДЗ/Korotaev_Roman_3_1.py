import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Генерация выборки для случайных величин xi
N = 1000
xi1 = np.random.normal(0, 1, N)
xi2 = np.random.normal(0, 1, N)
xi3 = np.random.normal(0, 1, N)
xi4 = np.random.normal(0, 1, N)

# Вычисление случайной величины nu
nu = xi1**2 + xi2**2 + xi3**2 + xi4**2
# Эмпирическая функция распределения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(nu, bins=30, density=True, alpha=0.6, color='g')
plt.title('Гистограмма относительных частот группированной выборки')

# Вычисление выборочных моментов
first_moment = np.mean(nu)
second_moment = np.mean(nu**2)
third_moment = np.mean(nu**3)
fourth_moment = np.mean(nu**4)

print("Первый выборочный момент:", first_moment)
print("Второй выборочный момент:", second_moment)
print("Третий выборочный момент:", third_moment)
print("Четвертый выборочный момент:", fourth_moment)

# Теоретические значения моментов
theoretical_first_moment = 4
theoretical_second_moment = 8
theoretical_third_moment = 48
theoretical_fourth_moment = 384

print("Теоретический первый момент:", theoretical_first_moment)
print("Теоретический второй момент:", theoretical_second_moment)
print("Теоретический третий момент:", theoretical_third_moment)
print("Теоретический четвертый момент:", theoretical_fourth_moment)

# Вывод графиков
plt.subplot(1, 2, 2)
x = np.linspace(0, 15, 1000)
plt.plot(x, stats.chi2.pdf(x, df=4), 'r', label='Плотность распределения хи-квадрат, k=4')
plt.legend()
plt.title('График плотности распределения хи-квадрат')
plt.show()