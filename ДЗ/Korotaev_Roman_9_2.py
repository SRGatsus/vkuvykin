import numpy as np
from scipy.stats import linregress

# Данные
x = np.array([38, 41, 24, 60, 41, 51, 58, 50, 65, 33])
y = np.array([73, 74, 43, 107, 65, 73, 99, 72, 100, 48])

# Вычисление выборочного коэффициента корреляции
correlation_coefficient = np.corrcoef(x, y)[0, 1]

# Построение линейной регрессии и определение p-value
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Вывод результатов
print(f"Выборочный коэффициент корреляции: {correlation_coefficient:.3f}")
print(f"Уравнение линейной регрессии: y = {slope:.3f}x + {intercept:.3f}")
print(f"Коэффициент детерминации (R^2): {r_value**2:.3f}")
print(f"Значение p-value: {p_value:.6f}")

# H0: Между переменными 𝑥 и 𝑦 нет линейной зависимости (коэффициент корреляции равен 0).
alpha = 0.05
if p_value < alpha:
    print("Нулевая гипотеза H0 отвергается. Существует значимая линейная зависимость между переменными x и y.")
else:
    print("Нулевая гипотеза H0 не отвергается. Нет значимой линейной зависимости между переменными x и y.")
