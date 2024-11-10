import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Загрузка данных
california = fetch_california_housing()
X, y = california.data, california.target

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение линейной регрессионной модели
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование цен
y_pred = model.predict(X_test)

# Визуализация ожидаемых и прогнозируемых цен
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, label='Прогнозируемые цены')
plt.plot([0, 5], [0, 5], '--r', label='Идеальная линия')
plt.xlabel('Ожидаемые цены')
plt.ylabel('Прогнозируемые цены')
plt.legend()
plt.title('Ожидаемые и прогнозируемые цены на жилье в Калифорнии')
plt.grid(True)
plt.show()

# Вычисление метрик регрессионной модели
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Вывод метрик
print(f"Коэффициент детерминации (R^2): {r2:.4f}")
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
