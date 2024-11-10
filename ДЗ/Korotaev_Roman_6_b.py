import numpy as np
import matplotlib.pyplot as plt

# База данных реального времени
data = np.array([
    [174, 180, 174, 177, 171, 173, 174, 178],
    [175, 177, 174, 176, 171, 168, 176, 177],
    [172, 172, 174, 179, 171, 176, 174, 173],
    [175, 173, 177, 180, 180, 173, 171, 174],
    [176, 171, 173, 177, 173, 167, 169, 175],
    [174, 175, 175, 170, 178, 173, 167, 179]
])

# Вычисление характеристик стационарной функции

# Среднее значение (взято среднее из всех значений)
mx_stationary = np.mean(data)
# Дисперсия (взята дисперсия из всех значений)
Dx_stationary = np.var(data)
# Ковариационная функция (в данном приближенном случае равна дисперсии)
Kx_stationary = np.var(data)
# Нормированная корреляционная функция (матрица единиц)
Rx_stationary = np.eye(data.shape[1])

print("Характеристики стационарной функции:")
print(f"Среднее значение (mx): {mx_stationary}")
print(f"Дисперсия (Dx): {Dx_stationary}")
print(f"Ковариационная функция (Kx): {Kx_stationary}")
print("Нормированная корреляционная функция (Rx):")
print(Rx_stationary)
t = np.arange(data.shape[1])  # Временная шкала
plt.figure(figsize=(8, 6))
plt.plot(t, np.full_like(t, mx_stationary), linestyle='--', color='m', label='Среднее значение (mx_stationary)')
plt.title('Стационарная случайная функция X(t)')
plt.xlabel('t')
plt.ylabel('X(t)')
plt.legend()
plt.grid(True)
plt.show()