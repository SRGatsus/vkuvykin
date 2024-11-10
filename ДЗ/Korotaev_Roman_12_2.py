import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Генерация выборки из нормального распределения
np.random.seed(0)
normal_sample = np.random.normal(loc=0, scale=1, size=1000)

# Генерация выборки из равномерного распределения
uniform_sample = np.random.uniform(low=0, high=1, size=1000)

# Построение графика нормальной вероятности для нормального распределения
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
stats.probplot(normal_sample, dist="norm", plot=plt)
plt.title('Normal Probability Plot (Normal Distribution)')

# Построение графика нормальной вероятности для равномерного распределения
plt.subplot(1, 2, 2)
stats.probplot(uniform_sample, dist="norm", plot=plt)
plt.title('Normal Probability Plot (Uniform Distribution)')

plt.tight_layout()
plt.show()