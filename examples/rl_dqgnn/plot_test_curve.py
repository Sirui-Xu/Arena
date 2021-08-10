from matplotlib import pyplot as plt
import pickle
import numpy as np
import os,sys

x = [1,3,5,7,9,11,13,15,17,19]
y_ub = np.arange(1,21,2)
y_heuristic = [1.0, 3.0, 4.99, 6.83, 8.53, 9.46, 10.89, 12.3, 13.69, 13.64]
y_DDQN = [0.99, 2.88, 4.78, 3.82, 2.37, 2.14, 1.35, 1.01, 0.91,1.09]
y_refactor_max = [0.98, 2.86, 4.64, 5.67, 5.81, 5.82, 5.35, 5.07, 3.34, 3.11]
y_refactor_success_max = [0.99, 3.0, 4.94, 6.55, 7.74, 8.47, 8.48, 7.72, 7.29, 5.85]

plt.xlabel('number of coins')
plt.ylabel('collected coins (mean of 100 runs)')
plt.xlim(0, 19)
plt.xticks(np.arange(1,21,2))
plt.ylim(0, 19)
plt.yticks(np.arange(1,21,2))
plt.plot(x, y_ub, label='max score')
plt.plot(x, y_heuristic, label='heuristic')
plt.plot(x, y_refactor_success_max, label='IL successful traj')
plt.plot(x, y_refactor_max, label='IL all traj')
plt.plot(x, y_DDQN, label='DoubleDQN')
plt.legend(loc='upper left')
plt.show()
