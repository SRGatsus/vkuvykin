import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, pareto

# Заданные параметры
alpha = 3
C0 = 10

# Вычисление математического ожидания, дисперсии и среднего квадратического отклонения
if alpha > 1:
    mean = alpha * C0 / (alpha - 1)
else:
    mean = np.inf

if alpha > 2:
    variance = (alpha * C0**2) / ((alpha - 1)**2 * (alpha - 2))
else:
    variance = np.inf

std_dev = np.sqrt(variance)

# Генерация выборки из распределения Парето
sample_size = 1000
samples = pareto.rvs(alpha, scale=C0, size=sample_size)

# Вычисление асимметрии и эксцесса
skewness = skew(samples)
excess_kurtosis = kurtosis(samples)

# Построение графиков
plt.figure(figsize=(14, 10))

# Плотность распределения вероятностей
x = np.linspace(10, 100, 1000)
pdf = (3/10) * (10/x)**4
plt.subplot(3, 1, 1)
plt.plot(x, pdf, label='Плотность вероятности')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Плотность распределения Парето')
plt.legend()
plt.grid(True)

# Функция распределения
cdf = 1 - (10/x)**3
plt.subplot(3, 1, 2)
plt.plot(x, cdf, label='Функция распределения')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Функция распределения Парето')
plt.legend()
plt.grid(True)

# Гистограмма и кумулятивная гистограмма
plt.subplot(3, 1, 3)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Гистограмма')
plt.xlabel('Зарплаты')
plt.ylabel('Частота')
plt.title('Гистограмма распределения зарплат')
plt.grid(True)

# Кумулятивная гистограмма
plt.twinx()
plt.hist(samples, bins=50, density=True, cumulative=True, alpha=0.3, color='b', label='Кумулятивная гистограмма')
plt.ylabel('Кумулятивная частота')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Вывод результатов
print(f'Математическое ожидание: {mean:.4f}')
print(f'Дисперсия: {variance:.4f}')
print(f'Среднее квадратическое отклонение: {std_dev:.4f}')
print(f'Коэффициент асимметрии: {skewness:.4f}')
print(f'Эксцесс: {excess_kurtosis:.4f}')
