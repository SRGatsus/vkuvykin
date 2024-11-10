import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
y = np.array([150, 60, 200])  # приборы
k1=0.04
k2=0.04
k3=0.02
def value(x):
    return ((y[0] - x[0]) / k1*y[0]) ** 2 \
        + ((y[1] - x[1]) / k2*y[1]) ** 2 + ((y[2] - x[2]) / k3*y[2]) ** 2


# уравнения и ограничения
cons = {'type': 'eq',
        'fun': lambda x: np.array([x[0] + x[1] - x[2]])
        }
bnds = ([140, 160], [50, 70], [190, 210])  # границы изменения
x0 = np.array([150, 60, 200])
res = minimize(value, x0, method='SLSQP', constraints=cons,
               bounds=bnds, options={'maxiter': 1000})

np.set_printoptions(precision=3)
print('N измеренные согласованные коррекция')
for i in range(3):
    print('{} {:>7.3f} {:12.3f} {:9.3f}'.format(i + 1, y[i], res.x[i], res.x[i] - y[i]))
x=np.arange(0,3,1)
plt.plot(x,res.x,'o',c='g',label='сoгласов')
plt.plot(x,y,'x',label='измеренные')
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.title('согласование данных')
plt.ylabel('значение')
plt.show()
