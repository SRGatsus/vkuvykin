import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Заданные интервалы
intervals = [(0, 1), (0, 12), (3, 7)]
N = 1000  # Количество наблюдений

for a, b in intervals:
    # Генерация выборки из равномерного распределения
    data = np.random.uniform(a, b, N)

    # Вычисление эмпирической функции распределения
    sorted_data = np.sort(data)
    ecdf = np.arange(1, N + 1) / N

    # Построение гистограммы
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Гистограмма и ECDF для равномерного распределения на ({a}, {b}]')
    plt.xlabel('Значение')
    plt.ylabel('Плотность')

    # Построение эмпирической функции распределения
    plt.plot(sorted_data, ecdf, color='orange', linestyle='-', linewidth=2)

    # Вычисление выборочного среднего и выборочной дисперсии
    sample_mean = np.mean(data)
    sample_variance = np.var(data)

    plt.legend(['ECDF', f'Значение: {sample_mean:.2f}, Различие: {sample_variance:.2f}'])
    plt.grid(True)
    plt.show()

    # Построение графика плотности распределения и интегральной функции распределения
    x = np.linspace(a - 1, b + 1, 1000)
    pdf = uniform.pdf(x, loc=a, scale=b - a)
    cdf = uniform.cdf(x, loc=a, scale=b - a)

    plt.figure(figsize=(10, 5))
    plt.plot(x, pdf, label='PDF')
    plt.plot(x, cdf, label='CDF')
    plt.title(f'Равномерное распространение: PDF и CDF на ({a}, {b}]')
    plt.xlabel('x')
    plt.ylabel('Вероятность')
    plt.legend()
    plt.grid(True)
    plt.show()