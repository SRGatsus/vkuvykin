import matplotlib.pyplot as plt
import numpy as np

# Параметры распределения
mu = 100
sigma = 10
n_trials = 1000
n_values = range(1, n_trials + 1)

sample_means = []  # Список для хранения выборочного среднего
sample_vars = []  # Список для хранения выборочной дисперсии


def calculate_sample_statistics(mu, sigma, n):
    samples = np.random.normal(mu, sigma, n)
    sample_mean = np.mean(samples)
    sample_means.append(sample_mean)
    sample_variance = np.var(samples)
    sample_means.append(sample_vars)
    sample_std = np.sqrt(sample_variance)
    return sample_std


# Проведение n испытаний и подсчет выборочного стандартного отклонения
sample_stds = [calculate_sample_statistics(mu, sigma, n) for n in n_values]

# Построение графика
plt.figure()
plt.plot(n_values, sample_stds)
plt.xlabel('n - число испытаний')
plt.ylabel('Выборочное стандартное отклонение')
plt.title('Зависимость выборочного стандартного отклонения от числа испытаний')
plt.grid(True)
plt.show()
