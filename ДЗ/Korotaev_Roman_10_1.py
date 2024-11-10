import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

# Данные из таблицы
x = np.array([0.085, 0.127, 0.155, 0.180, 0.203, 0.230, 0.253, 0.268, 0.283])
y = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.16, 0.18, 0.20])

# Определение функции для подбора коэффициента p
def parabola(x, p):
    return (x**2) / (2 * p)

# Подбор коэффициента p
p_opt, p_cov = curve_fit(parabola, x, y)
p = p_opt[0]

# Вычисление коэффициента детерминации R^2
y_pred = parabola(x, p)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

# Построение доверительного интервала для p
alpha = 0.05  # Уровень значимости
dof = len(x) - 1  # Число степеней свободы
t_crit = t.ppf(1 - alpha/2, dof)  # Критическое значение t
p_err = np.sqrt(np.diag(p_cov))
ci = t_crit * p_err[0]  # Доверительный интервал

# Вывод результатов
print(f"Оптимальное значение p: {p:.4f}")
print(f"Коэффициент детерминации R^2: {r_squared:.4f}")
print(f"Доверительный интервал для p: [{p - ci:.4f}, {p + ci:.4f}]")

# Построение графика
plt.scatter(x, y, label='Данные')
plt.plot(x, y_pred, label=f'Подбор: y = x^2/(2*{p:.4f})', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Параболический профиль канала')
plt.grid(True)
plt.show()
