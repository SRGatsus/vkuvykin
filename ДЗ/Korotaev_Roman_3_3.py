import scipy.stats as stats

k_values = range(1, 16)  # Число узлов от 1 до 15
alpha = 0.05  # Уровень значимости

# Создаем таблицу значений критерия Пирсона
table_data = []
for k in k_values:
    critical_value = stats.chi2.isf(alpha, k-1)
    table_data.append([k, critical_value])

# Выводим таблицу значений критерия Пирсона
print("Число узлов (k) | Критическое значение")
print("---------------------------------------")
for row in table_data:
    print(f"{row[0]}               | {row[1]}")
print("С табличными значениями совподает")