import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

folder_path = '../АВТ'
col_index_plan = 6
col_index_device = 9
col_index_balanced = 11

row_indices = {
    'desalinated_oil': 0,
    'gas_condensate': 1,
    'trap_product': 2,
    'total_materials': 3,
    'gasoline_fraction': 4,
    'total_gasoline': 5,
    'component_diesel_fuel': 8,
    'dt_e_22_and_dt_k_5': 9,
    'total_light': 10,
    'total_wac_shoulder_straps': 15,
    'fatty_gas': 16,
    'agl_fraction': 17,
    'fraction_fuel_oil': 18,
    'the_wac_faction': 19,
    'losses': 20,
    'total_output': 21,
    'debalance': 22
}


def remove_columns_in_range(df, min_val, max_val):
    columns_to_remove = []

    # Проверяем каждый столбец
    for col in df.columns:
        should_remove = True

        # Проверяем каждое значение в столбце
        for val in df[col]:
            if min_val <= val <= max_val:
                should_remove = False
                break

        # Если нужно удалить столбец, добавляем его в список
        if should_remove:
            columns_to_remove.append(col)

    # Удаляем выбранные столбцы из DataFrame
    df_cleaned = df.drop(columns=columns_to_remove)
    return df_cleaned


def read_excel_files(folder_path):
    all_files = os.listdir(folder_path)
    return [file for file in all_files if file.endswith('.xls') or file.endswith('.xlsx')]


def extract_data_from_file(file_path, columns, start_index, offset_index=2):
    data = pd.read_excel(file_path)
    val_1 = [data.at[start_index + i, columns[col_index_plan]] for i in range(offset_index, 23 + offset_index)]
    val_2 = [data.at[start_index + i, columns[col_index_device]] for i in range(offset_index, 23 + offset_index)]
    val_3 = [data.at[start_index + i, columns[col_index_balanced]] for i in range(offset_index, 23 + offset_index)]
    return val_1, val_2, val_3


def find_start_index(data, columns, search_value="АВТ-2", search_column=3, start_row=500, end_row=530):
    for i in range(start_row, end_row):
        if data.at[i, columns[search_column]] == search_value:
            return i
    return -1


def prepare_data(folder_path, row_indices):
    excel_files = read_excel_files(folder_path)
    plan_data, device_data, balanced_data = [], [], []

    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_excel(file_path)
        columns = data.columns
        start_index = find_start_index(data, columns)
        if start_index != -1:
            val_1, val_2, val_3 = extract_data_from_file(file_path, columns, start_index)
            plan_data.append(val_1)
            device_data.append(val_2)
            balanced_data.append(val_3)
    data_list = [".".join(item.replace('.xls','').replace('.xlsx','').split("_")[2:]) for item in excel_files]
    data_dict = {
        'День': data_list,
        'Нефть обессол.': [item for item in [i[row_indices['desalinated_oil']] for i in device_data]],
        'Газовый конденсат': [item for item in [i[row_indices['gas_condensate']] for i in device_data]],
        'Продукт': [item for item in [i[row_indices['trap_product']] for i in device_data]],
        'Сырье': [item for item in [i[row_indices['total_materials']] for i in device_data]],
        'Бензин': [item for item in [i[row_indices['total_gasoline']] for i in device_data]],
        'Дизель': [item for item in [i[row_indices['component_diesel_fuel']] for i in device_data]],
        'ДТ E-22 и ДТ К-5': [item for item in [i[row_indices['dt_e_22_and_dt_k_5']] for i in device_data]],
        'Светлые продукты': [item for item in [i[row_indices['total_light']] for i in device_data]],
        'Вак. погоны': [item for item in [i[row_indices['total_wac_shoulder_straps']] for i in device_data]],
        'Жирный газ': [item for item in [i[row_indices['fatty_gas']] for i in device_data]],
        'Фракция АГЛ': [item for item in [i[row_indices['agl_fraction']] for i in device_data]],
        'Фракция мазута': [item for item in [i[row_indices['fraction_fuel_oil']] for i in device_data]],
        'Фракция вак.гудрона': [item for item in [i[row_indices['the_wac_faction']] for i in device_data]],
        'Потери': [item for item in [i[row_indices['losses']] for i in device_data]],
        'Итог': [i[row_indices['total_output']] for i in device_data],
        'Дебалансы': [i[row_indices['debalance']] for i in device_data],
        'Согл Нефть обессол.': [item for item in [i[row_indices['desalinated_oil']] for i in device_data]],
        'Согл Газовый конденсат': [item for item in [i[row_indices['gas_condensate']] for i in device_data]],
        'Согл Продукт': [item for item in [i[row_indices['trap_product']] for i in device_data]],
        'Согл Сырье': [item for item in [i[row_indices['total_materials']] for i in device_data]],
        'Согл Бензин': [item for item in [i[row_indices['total_gasoline']] for i in device_data]],
        'Согл Дизель': [item for item in [i[row_indices['component_diesel_fuel']] for i in device_data]],
        'Согл ДТ E-22 и ДТ К-5': [item for item in [i[row_indices['dt_e_22_and_dt_k_5']] for i in device_data]],
        'Согл Светлые продукты': [item for item in [i[row_indices['total_light']] for i in device_data]],
        'Согл Вак. погоны': [item for item in [i[row_indices['total_wac_shoulder_straps']] for i in device_data]],
        'Согл Жирный газ': [item for item in [i[row_indices['fatty_gas']] for i in device_data]],
        'Согл Фракция АГЛ': [item for item in [i[row_indices['agl_fraction']] for i in device_data]],
        'Согл Фракция мазута': [item for item in [i[row_indices['fraction_fuel_oil']] for i in device_data]],
        'Согл Фракция вак.гудрона': [item for item in [i[row_indices['the_wac_faction']] for i in device_data]],
        'Согл Потери': [item for item in [i[row_indices['losses']] for i in device_data]],
        'Согл Итог': [i[row_indices['total_output']] for i in device_data],
        'Согл Дебалансы': [i[row_indices['debalance']] for i in device_data]
    }
    df = pd.DataFrame(data_dict)
    condition: pd.DataFrame = (df.iloc[:, 1:] < -0.05) | (df.iloc[:, 1:] > 0.05)
    df_cleaned = df[condition.any(axis=1)]
    df['День'] = pd.to_datetime(df_cleaned['День'], format='%d.%m.%Y')
    df_cleaned = df.sort_values(by='День')


    data_dict = {
        'День': data_list,
        'Нефть обессол.': [item for item in [i[row_indices['desalinated_oil']] for i in plan_data]],
        'Газовый конденсат': [item for item in [i[row_indices['gas_condensate']] for i in plan_data]],
        'Продукт': [item for item in [i[row_indices['trap_product']] for i in plan_data]],
        'Сырье': [item for item in [i[row_indices['total_materials']] for i in plan_data]],
        'Бензин': [item for item in [i[row_indices['total_gasoline']] for i in plan_data]],
        'Дизель': [item for item in [i[row_indices['component_diesel_fuel']] for i in plan_data]],
        'ДТ E-22 и ДТ К-5': [item for item in [i[row_indices['dt_e_22_and_dt_k_5']] for i in plan_data]],
        'Светлые продукты': [item for item in [i[row_indices['total_light']] for i in plan_data]],
        'Вак. погоны': [item for item in [i[row_indices['total_wac_shoulder_straps']] for i in plan_data]],
        'Жирный газ': [item for item in [i[row_indices['fatty_gas']] for i in plan_data]],
        'Фракция АГЛ': [item for item in [i[row_indices['agl_fraction']] for i in plan_data]],
        'Фракция мазута': [item for item in [i[row_indices['fraction_fuel_oil']] for i in plan_data]],
        'Фракция вак.гудрона': [item for item in [i[row_indices['the_wac_faction']] for i in plan_data]],
        'Потери': [item for item in [i[row_indices['losses']] for i in plan_data]],
        'Итог': [i[row_indices['total_output']] for i in plan_data],
        'Дебалансы': [i[row_indices['debalance']] for i in plan_data],
        'Согл Нефть обессол.': [item for item in [i[row_indices['desalinated_oil']] for i in plan_data]],
        'Согл Газовый конденсат': [item for item in [i[row_indices['gas_condensate']] for i in plan_data]],
        'Согл Продукт': [item for item in [i[row_indices['trap_product']] for i in plan_data]],
        'Согл Сырье': [item for item in [i[row_indices['total_materials']] for i in plan_data]],
        'Согл Бензин': [item for item in [i[row_indices['total_gasoline']] for i in plan_data]],
        'Согл Дизель': [item for item in [i[row_indices['component_diesel_fuel']] for i in plan_data]],
        'Согл ДТ E-22 и ДТ К-5': [item for item in [i[row_indices['dt_e_22_and_dt_k_5']] for i in plan_data]],
        'Согл Светлые продукты': [item for item in [i[row_indices['total_light']] for i in plan_data]],
        'Согл Вак. погоны': [item for item in [i[row_indices['total_wac_shoulder_straps']] for i in plan_data]],
        'Согл Жирный газ': [item for item in [i[row_indices['fatty_gas']] for i in plan_data]],
        'Согл Фракция АГЛ': [item for item in [i[row_indices['agl_fraction']] for i in plan_data]],
        'Согл Фракция мазута': [item for item in [i[row_indices['fraction_fuel_oil']] for i in plan_data]],
        'Согл Фракция вак.гудрона': [item for item in [i[row_indices['the_wac_faction']] for i in plan_data]],
        'Согл Потери': [item for item in [i[row_indices['losses']] for i in plan_data]],
        'Согл Итог': [i[row_indices['total_output']] for i in plan_data],
        'Согл Дебалансы': [i[row_indices['debalance']] for i in plan_data]
    }
    df = pd.DataFrame(data_dict)
    condition: pd.DataFrame = (df.iloc[:, 1:] < -0.05) | (df.iloc[:, 1:] > 0.05)
    df_plan_cleaned = df[condition.any(axis=1)]
    df['День'] = pd.to_datetime(df_plan_cleaned['День'], format='%d.%m.%Y')
    df_plan_cleaned = df.sort_values(by='День')

    return df_cleaned,df_plan_cleaned


def plot_graph(ax, df, x_cols: [], y_cols: [], x_label, y_label, title, markers):
    for i in range(min(len(x_cols), len(y_cols), len(markers))):
        ax.plot(df[x_cols[i]], df[y_cols[i]], label=y_cols[i], marker=markers[i])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_dependencies(df):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plot_graph(axs[0, 0], df, ['День'], ['Сырье'], 'День', 'Сырье (т.т.)', 'Зависимость сырья от дней', ['o'])
    plot_graph(axs[0, 1], df, ['День'], ['Бензин'], 'День', 'Бензин (т.т.)', 'Зависимость бензина от дней', ['o'])
    plot_graph(axs[1, 0], df, ['День'], ['Светлые продукты'], 'День', 'Светлые продукты (т.т.)',
               'Зависимость светлых продуктов от дней', ['o'])
    plot_graph(axs[1, 1], df, ['День'], ['Дизель'], 'День', 'Дизель (т.т.)', 'Зависимость дизеля от дней', ['o'])
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plot_graph(axs[0, 0], df, ['День', 'День'], ['Бензин', 'Сырье'], 'День', 'Бензин (т.т.)',
               'Зависимость бензина и сырья от дней', ['o', 'x'])
    plot_graph(axs[0, 1], df, ['День', 'День'], ['Дизель', 'Сырье'], 'День', 'Дизель (т.т.)',
               'Зависимость дизеля и сырья от дней', ['o', 'x'])
    plot_graph(axs[1, 0], df, ['День', 'День'], ['Светлые продукты', 'Сырье'], 'День', 'Светлые продукты (т.т.)',
               'Зависимость светлых и сырья от дней', ['o', 'x'])
    plot_graph(axs[1, 1], df, ['День', 'День'], ['Дебалансы', 'Сырье'], 'День', 'Дебалансы (т.т.)',
               'Зависимость дебаланса и сырья от дней', ['o', 'x'])
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plot_graph(axs[0, 0], df, ['День', 'День'], ['Согл Сырье', 'Сырье'], 'День', 'Сырье (т.т.)',
               'Зависимость согл данных сырья', ['o', 'x'])
    plot_graph(axs[0, 1], df, ['День', 'День'], ['Согл Дизель', 'Дизель'], 'День', 'Дизель (т.т.)',
               'Зависимость согл данных Дизель', ['o', 'x'])
    plot_graph(axs[1, 0], df, ['День', 'День'], ['Согл Светлые продукты', 'Светлые продукты'], 'День',
               'Светлые продукты (т.т.)',
               'Зависимость согл данных Светлые продукты', ['o', 'x'])
    plot_graph(axs[1, 1], df, ['День', 'День'], ['Согл Итог', 'Итог'], 'День', 'Итог (т.т.)',
               'Зависимость согл данных Итог', ['o', 'x'])
    plt.tight_layout()
    plt.show()


def plot_histograms(df):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(df['Светлые продукты'], kde=False, stat='probability', ax=axs[0, 0])
    axs[0, 0].set_xlabel('Светлые продукты (т.т.)')
    axs[0, 0].set_ylabel('Относительная частота')
    axs[0, 0].set_title('Гистограмма относительных частот для светлых продуктов')

    sns.histplot(df['Дебалансы'], kde=False, stat='probability', ax=axs[0, 1])
    axs[0, 1].set_xlabel('Дебалансы (т.т.)')
    axs[0, 1].set_ylabel('Относительная частота')
    axs[0, 1].set_title('Гистограмма относительных частот для дебаланса')

    axs[1, 0].plot(df['День'], df['Дебалансы'], label='Дебалансы', marker='o')
    axs[1, 0].set_xlabel('Дни')
    axs[1, 0].set_ylabel('Дебалансы (т.т.)')
    axs[1, 0].set_title('Линейный график дебаланса')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    sns.histplot(df['Дебалансы'], kde=True, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Дебалансы (т.т.)')
    axs[1, 1].set_ylabel('Частота')
    axs[1, 1].set_title('Гистограмма частот и график распределения дебаланса')

    plt.tight_layout()
    plt.show()


def data_reconciliation(device_array: [], index):
    y = np.array(device_array)
    x0 = np.array(device_array)
    percentage = 0.1  # 10%
    bnds = tuple([[num - num * percentage, num + num * percentage] for num in device_array])

    def fun(x):
        array_input = [x[i] for i in range(len(x) - index)]
        array_input2 = [x[i] for i in range(len(x) - index, len(x))]
        sum_ = sum(array_input)
        sum_2 = sum(array_input2)
        raz = sum_ - sum_2
        return np.array([raz])

    cons = {'type': 'eq',
            'fun': fun
            }

    def value(x):
        val = []
        for i in range(len(x)):
            if y[i] == 0:
                val.append(0)
                continue
            val.append(((y[i] - x[i]) / y[i]) ** 2)
        sum_ = sum(val)
        return sum_

    res = minimize(value, x0, method='SLSQP', constraints=cons,
                   bounds=bnds, options={'maxiter': 1000})
    np.set_printoptions(precision=3)
    # print(res)
    # print('N измеренные согласованные коррекция')
    # for i in range(len(device_array)):
    #     print('{} {:>7.3f} {:12.3f} {:9.3f}'.format(i + 1, y[i], res.x[i], res.x[i] - y[i]))
    return res


def cal_reconciliation(df):
    for i in df['День']:
        if not df['Бензин'].get(i):
            continue

        reconciliation = data_reconciliation(
            [df['Бензин'][i], df['Дизель'][i], df['ДТ E-22 и ДТ К-5'][i], df['Вак. погоны'][i], df['Жирный газ'][i],
             df['Потери'][i], df['Фракция АГЛ'][i], df['Фракция мазута'][i],
             df['Фракция вак.гудрона'][i], df['Нефть обессол.'][i],
             df['Газовый конденсат'][i], df['Продукт'][i]], 3)
        df.loc[i, 'Согл Бензин'] = reconciliation.x[0]
        df.loc[i, 'Согл Дизель'] = reconciliation.x[1]
        df.loc[i, 'Согл ДТ E-22 и ДТ К-5'] = reconciliation.x[2]
        df.loc[i, 'Согл Вак. погоны'] = reconciliation.x[3]
        df.loc[i, 'Согл Жирный газ'] = reconciliation.x[4]
        df.loc[i, 'Согл Потери'] = reconciliation.x[5]
        df.loc[i, 'Согл Фракция АГЛ'] = reconciliation.x[6]
        df.loc[i, 'Согл Фракция мазута'] = reconciliation.x[7]
        df.loc[i, 'Согл Фракция вак.гудрона'] = reconciliation.x[8]
        df.loc[i, 'Согл Нефть обессол.'] = reconciliation.x[9]
        df.loc[i, 'Согл Газовый конденсат'] = reconciliation.x[10]
        df.loc[i, 'Согл Продукт'] = reconciliation.x[11]

        df.loc[i, 'Согл Сырье'] = sum(
            [df['Согл Нефть обессол.'][i], df['Согл Газовый конденсат'][i],
             df['Согл Продукт'][i]])
        df.loc[i, 'Согл Светлые продукты'] = sum(
            [df['Согл Бензин'][i], df['Согл Дизель'][i], df['Согл ДТ E-22 и ДТ К-5'][i]])
        df.loc[i, 'Согл Итог'] = sum([df['Согл Светлые продукты'][i], df['Согл Вак. погоны'][i],
                                      df['Согл Жирный газ'][i],
                                      df['Согл Потери'][i], df['Согл Фракция АГЛ'][i],
                                      df['Согл Фракция мазута'][i],
                                      df['Согл Фракция вак.гудрона'][i]])
    return df

def regression(df,df_plan):
    def linear_regression(x, y):
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Вычисляем коэффициенты
        b1 = (np.sum((x - x_mean) * (y - y_mean))) / (np.sum((x - x_mean) ** 2))
        b0 = y_mean - b1 * x_mean

        return b0, b1

    def predict(x, b0, b1):
        return b0 + b1 * x

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    x = df['Согл Светлые продукты']
    y = df['Согл Сырье']
    x2 = df['Светлые продукты']
    y2 = df['Сырье']
    x3 = df_plan['Светлые продукты']
    y3 = df_plan['Сырье']

    b0, b1 = linear_regression(x, y)
    y_pred = predict(x, b0, b1)
    print(b0, b1)

    b0, b1 = linear_regression(x3, y3)
    y3_pred = predict(x3, b0, b1)
    print(b0, b1)
    axs[0, 0].scatter(x, y, color='blue', label='Согл Данные')
    axs[0, 0].scatter(x2, y2, color='green', label='Данные')
    axs[0, 0].scatter(x3, y3, color='yellow', label='Плановых Данные')
    axs[0, 0].plot(x, y_pred, color='red', label='Линейная регрессия соглс')
    axs[0, 0].plot(x3, y3_pred, color='green', label='Линейная регрессия плановых')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].set_title('Регрессионная зависимость: светлых и сырья')
    axs[0, 0].legend()
    axs[0, 0].grid(True)


    # x = df['Светлые продукты']
    # y = df['Сырье']
    # x2 = df_plan['Светлые продукты']
    # y2 = df_plan['Сырье']
    # b0, b1 = linear_regression(x, y)
    # y_pred = predict(x, b0, b1)
    # print(b0, b1)
    # b0, b1 = linear_regression(x2, y2)
    # y2_pred = predict(x2, b0, b1)
    # print(b0, b1)
    # axs[0, 1].scatter(x, y, color='blue', label='Данные')
    # axs[0, 1].scatter(x2, y2, color='green', label='Плановые Данные')
    # axs[0, 1].plot(x, y_pred, color='red', label=f'Линейная регрессия')
    # axs[0, 1].plot(x2, y2_pred, color='yellow', label=f'Линейная регрессия плановых')
    # axs[0, 1].set_xlabel('X')
    # axs[0, 1].set_ylabel('Y')
    # axs[0, 1].set_title('Регрессионная зависимость: светлых и сырья')
    # axs[0, 1].legend()
    # axs[0, 1].grid(True)
    plt.tight_layout()
    plt.show()
# Основной код
df,df_plan = prepare_data(folder_path, row_indices)
df = cal_reconciliation(df)
df_plan = cal_reconciliation(df_plan)
regression(df,df_plan)
plot_dependencies(df)
plot_histograms(df)




