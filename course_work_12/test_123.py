import numpy as np

def f(x):
    return np.exp(-x**2)

# Параметры интеграла
a = 0
b = 1
N = 10000

# Генерация случайных точек
x_samples = np.random.uniform(a, b, N)

# Оценка интеграла
integral_estimate = (b - a) * np.mean(f(x_samples))

print(f'Оценка интеграла: {integral_estimate}')
