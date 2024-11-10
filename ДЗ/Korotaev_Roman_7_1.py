from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Параметры нормального распределения
mu = 1.5
sigma = 1
control_answers = {
    6: 3.4,
    5: 233,
    4: 6210,
    3: 66807,
    2: 308537,
    1: 690000
}

fig, ax = plt.subplots()
print("sigma", "табличный", "вычисленый")
for sigma_level, defect_count in control_answers.items():
    x = np.linspace(mu - 4 * sigma_level, mu + 4 * sigma_level, 1000)
    y = norm.pdf(x, mu, sigma_level)
    p_probability = 1 - norm.cdf(sigma_level, mu, sigma)
    ax.plot(x, y, label=f'P(ξ > {sigma_level}σ): { round(p_probability * 1e6, 2)}', linewidth=1.5)

ax.set_xlabel('x')
ax.set_ylabel('Плотность')
ax.set_title('Нормальное распределение')
ax.legend()
plt.show()