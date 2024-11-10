import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto
from scipy.stats import moment

# Параметры распределения Парето
a_values = [1, 2, 3]
C0 = 3
x = np.linspace(1, 5, 1000)


# Функции плотности и распределения
def pareto_pdf(x, a, C0):
    return (a / C0) * (C0 / x) ** (a + 1)


def pareto_cdf(x, a, C0):
    return 1 - (C0 / x) ** a


# Построение графиков
plt.figure(figsize=(14, 10))

# График плотности распределения
plt.subplot(2, 1, 1)
for a in a_values:
    plt.plot(x, pareto_pdf(x, a, C0), label=f'a={a}')
plt.xlabel('x')
plt.ylabel('Плотность вероятности f(x)')
plt.title('Плотность распределения Парето')
plt.legend()
plt.grid(True)

# График функции распределения
plt.subplot(2, 1, 2)
for a in a_values:
    plt.plot(x, pareto_cdf(x, a, C0), label=f'a={a}')
plt.xlabel('x')
plt.ylabel('Функция распределения F(x)')
plt.title('Функция распределения Парето')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вычисление математических характеристик
results = {}
for a in a_values:
    if a > 1:
        mean = C0 / (a - 1)
    else:
        mean = np.inf

    if a > 2:
        variance = (C0 ** 2) * a / ((a - 1) ** 2 * (a - 2))
    else:
        variance = np.inf

    skewness = (2 * (1 + a)) / (a - 3) * np.sqrt((a - 2) / a) if a > 3 else np.inf
    kurtosis = 6 * (a ** 2 + a + 1) / (a * (a - 3) * (a - 4)) if a > 4 else np.inf

    results[a] = {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

# Вывод результатов
for a, stats in results.items():
    print(f'a = {a}:')
    print(f'  Математическое ожидание: {stats["mean"]}')
    print(f'  Дисперсия: {stats["variance"]}')
    print(f'  Коэффициент асимметрии: {stats["skewness"]}')
    print(f'  Эксцесс: {stats["kurtosis"]}')
    print()
