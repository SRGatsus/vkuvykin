import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = './АВТ_2.xlsx'
count_list = 12
petroleum_raw_materials=[]
light_raw_materials=[]
for i in range(count_list + 1):
    data = pd.read_excel(file_path, sheet_name=i, engine='openpyxl')
    count_day = int(data.at[5, 'Unnamed: 5'])
    for day in range(count_day):
        petroleum_raw_materials.append(float(data.at[5, 'Unnamed: 6']) / count_day)
        light_raw_materials.append(float(data.at[12, 'Unnamed: 6']) / count_day)
print(petroleum_raw_materials)
print(light_raw_materials)
# Выбор признаков и целевой переменной

X = np.array(petroleum_raw_materials).reshape(-1, 1)
y = np.array(light_raw_materials).reshape(-1, 1)
# Разделение данных на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Построение линейной регрессионной модели
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Фактические данные')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Линейная регрессия')
plt.xlabel('Нефтяное сырье')
plt.ylabel('Светлые продукты')
plt.title('Регрессионная зависимость: Светлые продукты vs Нефтяное сырье')
plt.legend()
plt.show()