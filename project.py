import matplotlib.pyplot as plt
import pandas as pd

file_path = './АВТ_2.xlsx'
count_list = 12
final_raw_materials_for_devices = []
final_output_for_devices = []
desalinated_oil_for_devices = []
debalance_for_devices = []
for i in range(count_list + 1):
    data = pd.read_excel(file_path, sheet_name=i, engine='openpyxl')
    count_day = int(data.at[5, 'Unnamed: 5'])
    for day in range(count_day):
        final_raw_materials_for_devices.append(float(data.at[5, 'Unnamed: 6']) / count_day)
        final_output_for_devices.append(float(data.at[23, 'Unnamed: 6']) / count_day)
        desalinated_oil_for_devices.append(float(data.at[2, 'Unnamed: 6']) / count_day)
        debalance_for_devices.append((float(data.at[24, 'Unnamed: 6']) / count_day )* 1000)

dates = pd.interval_range(1,len(final_raw_materials_for_devices),1)
dates = [i for i in range(len(final_raw_materials_for_devices))]


plt.figure(figsize=(min(final_raw_materials_for_devices), max(final_raw_materials_for_devices)))
plt.subplot(2, 2, 1)
plt.plot(dates, final_raw_materials_for_devices, marker='o', linestyle='-', label='Нефть')
plt.plot(dates, final_output_for_devices, marker='o', linestyle='-', label='Итог')
plt.title('Два графика на одной системе координат')
plt.xlabel('Дата')
plt.ylabel('Значения y')
plt.xticks(rotation=45)
plt.legend()

plt.subplot(2, 2, 2)
df = pd.DataFrame({
    "Нефть обессоленная": desalinated_oil_for_devices,
    "Суммарно": final_raw_materials_for_devices
})
df.boxplot()
plt.title('')
plt.legend()

plt.subplot(2, 3, 5)
df = pd.DataFrame({
    "Дебаланс": debalance_for_devices
})
df.boxplot()
plt.title('')
plt.legend()

plt.subplot(2, 3, 6)
df = pd.DataFrame({
    "value": debalance_for_devices
})
df['value'].hist()
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()


plt.subplot(2, 3, 4)
plt.plot(dates, debalance_for_devices, marker='o', linestyle='-', label='Дебаланс')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()

plt.show()
