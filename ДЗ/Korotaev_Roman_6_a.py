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


# Вычисление характеристик
def calculate_characteristics(data):
    # Среднее значение
    mx = np.mean(data, axis=0)

    # Дисперсия
    Dx = np.var(data, axis=0)

    # Ковариационная функция
    Kx = np.cov(data.T)

    # Нормированная корреляционная функция
    Rx = np.corrcoef(data.T)

    return mx, Dx, Kx, Rx


# Вычисление характеристик
mx, Dx, Kx, Rx = calculate_characteristics(data)

# Конвертация массивов NumPy в списки для удобной печати
mx_list = mx.tolist()
Dx_list = Dx.tolist()
Kx_list = Kx.tolist()
Rx_list = Rx.tolist()

print("Характеристики:")
for i in range(len(mx_list)):
    print(f"t={i}: mx={mx_list[i]}, Dx={Dx_list[i]}")

print("\nКовариационная функция:")
print(Kx)
print("\nНормированная корреляционная функция:")
print(Rx)
# Построение графиков
t = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Временная шкала

# График среднего значения mx(t)
plt.figure(figsize=(10, 6))
plt.plot(t, mx, marker='o', color='b')
plt.title('Среднее значение функции X(t)')
plt.xlabel('t')
plt.ylabel('mx(t)')
plt.grid(True)
plt.show()

# График дисперсии Dx(t)
plt.figure(figsize=(10, 6))
plt.plot(t, Dx, marker='o', color='r')
plt.title('Дисперсия функции X(t)')
plt.xlabel('t')
plt.ylabel('Dx(t)')
plt.grid(True)
plt.show()

# Графики ковариационной функции Kx(t) и нормированной корреляционной функции Rx(t)
plt.figure(figsize=(12, 6))
plt.imshow(Kx, cmap='coolwarm', interpolation='nearest')
plt.title('Ковариационная функция Kx(t, t\')')
plt.colorbar()
plt.xticks(t, t)
plt.yticks(t, t)
plt.xlabel('t')
plt.ylabel('t\'')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(Rx, cmap='coolwarm', interpolation='nearest')
plt.title('Нормированная корреляционная функция Rx(t, t\')')
plt.colorbar()
plt.xticks(t, t)
plt.yticks(t, t)
plt.xlabel('t')
plt.ylabel('t\'')
plt.show()