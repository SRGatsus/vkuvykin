import pandas as pd
import matplotlib.pyplot as plt
import glob

# Путь к папке с файлами Excel
folder_path = './АВТ/'

# Поиск всех файлов Excel в указанной папке
excel_files = glob.glob(folder_path + "*.xls")

# Создаем пустой DataFrame для объединения всех данных
all_data = pd.DataFrame()

# Чтение данных из всех файлов и добавление их в общий DataFrame
for i, file in enumerate(excel_files, 1):
    df = pd.read_excel(file)
    df['День'] = i
    all_data = pd.concat([all_data, df], ignore_index=True)

# Пример структуры данных, основанный на изображении
all_data.columns = ['Сырье', 'Газовый конденсат', 'Ловушечный продукт', 'Итого Сырье', 'Фракция бензина', 'Итого Бензин',
                    'Компонент дизельного топлива', 'Итого Светлые', 'Итого Вак', 'Жирный газ', 'Фракция АГД',
                    'Фракция мазута', 'Фракция вак. гудрона', 'Потери', 'Итого Выработка', 'Дебаланс', 'День']

# Создание необходимых графиков

# График зависимости сырья, бензина, светлых продуктов и дизеля от дней
plt.figure(figsize=(10, 6))
plt.plot(all_data['День'], all_data['Сырье'], label='Сырье')
plt.plot(all_data['День'], all_data['Фракция бензина'], label='Бензин')
plt.plot(all_data['День'], all_data['Итого Светлые'], label='Светлые продукты')
plt.plot(all_data['День'], all_data['Компонент дизельного топлива'], label='Дизель')
plt.xlabel('Дни')
plt.ylabel('Количество (т.т.)')
plt.title('Зависимость сырья, бензина, светлых продуктов и дизеля от дней')
plt.legend()
plt.grid(True)
plt.show()

# График зависимости бензина, дизеля, светлых продуктов и дебаланса от сырья
plt.figure(figsize=(10, 6))
plt.plot(all_data['Сырье'], all_data['Фракция бензина'], label='Бензин')
plt.plot(all_data['Сырье'], all_data['Компонент дизельного топлива'], label='Дизель')
plt.plot(all_data['Сырье'], all_data['Итого Светлые'], label='Светлые продукты')
plt.plot(all_data['Сырье'], all_data['Дебаланс'], label='Дебалансы')
plt.xlabel('Сырье (т.т.)')
plt.ylabel('Количество (т.т.)')
plt.title('Зависимость бензина, дизеля, светлых продуктов и дебаланса от сырья')
plt.legend()
plt.grid(True)
plt.show()
