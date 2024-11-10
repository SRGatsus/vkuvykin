import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Данные Квартета Энскомба
anscombe_data = {
    'I': {
        'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    },
    'II': {
        'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    },
    'III': {
        'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        'y': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    },
    'IV': {
        'x': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19],
        'y': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]
    }
}

# Создание графиков и расчет статистик
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

for i, (dataset, data) in enumerate(anscombe_data.items()):
    x = np.array(data['x'])
    y = np.array(data['y'])

    # Вычисление статистик
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    variance_x = np.var(x, ddof=1)
    variance_y = np.var(y, ddof=1)
    correlation_coefficient = np.corrcoef(x, y)[0, 1]

    # Построение линейной регрессии
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept
    residuals = y - line

    # Построение графиков
    sns.scatterplot(ax=axs[0, i], x=x, y=y)
    axs[0, i].plot(x, line, color='red')
    axs[0, i].set_title(f'Dataset {dataset}')
    axs[0, i].set_xlabel('x')
    axs[0, i].set_ylabel('y')
    axs[0, i].text(0.05, 0.95,
                   f'Mean(x)={mean_x:.2f}\nMean(y)={mean_y:.2f}\nVar(x)={variance_x:.2f}\nVar(y)={variance_y:.2f}\nCorr={correlation_coefficient:.2f}',
                   transform=axs[0, i].transAxes, fontsize=10, verticalalignment='top')

    # Графики остатков
    sns.residplot(ax=axs[1, i], x=x, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
    axs[1, i].hlines(0, min(x), max(x), colors='grey', linestyles='dashed')
    axs[1, i].set_title(f'Residuals {dataset}')
    axs[1, i].set_xlabel('x')
    axs[1, i].set_ylabel('Residuals')

plt.tight_layout()
plt.show()