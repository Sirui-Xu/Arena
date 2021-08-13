from matplotlib import pyplot as plt
import pickle
import numpy as np
import os,sys
'''
results = []
for i in range(10):
    with open(f'/home/yiran/pc_mapping/arena-v2/examples/bc_saved_models/refactor_success_max_mine/run{i}/test_result.npy', 'rb') as f:
        result_i = pickle.load(f)
    result_number = [v for (k,v) in result_i.items()]
    results.append(result_number)
results = np.array(results)
result_mean = results.mean(axis=0)
result_std = results.std(axis=0)
print(result_mean, result_std)
exit()
'''

x = [1,3,5,7,9,11,13,15,17,19]
y_ub = np.arange(1,21,2)
y_heuristic = [1.0, 3.0, 4.99, 6.83, 8.53, 9.46, 10.89, 12.3, 13.69, 13.64]
y_DDQN = [0.99, 2.88, 4.78, 3.82, 2.37, 2.14, 1.35, 1.01, 0.91,1.09]
y_refactor_max = [0.98, 2.86, 4.64, 5.67, 5.81, 5.82, 5.35, 5.07, 3.34, 3.11]
y_refactor_success_max = [0.99, 3.0, 4.94, 6.55, 7.74, 8.47, 8.48, 7.72,  7.29, 5.85]
y_refactor_purify_10of10_max = [1.0, 2.97, 4.85, 6.76, 8.05, 8.42, 8.66,  8.03, 7.58, 5.65]
y_refactor_purify_9of10_max =  [1.0, 3.0,  4.94, 6.69, 8.27, 9.27, 9.3,   8.88, 8.87, 7.94]
y_refactor_purify_8of10_max =  [1.0, 3.0,  4.95, 6.76, 7.68, 8.14, 8.11,  8.18, 6.99, 5.09]
y_refactor_purify_7of10_max =  [1.0, 3.0,  4.93, 6.91, 8.32, 9.46, 10.64, 11.7, 11.81, 10.86]
y_refactor_purify_6of10_max =  [1.0, 2.97, 4.93, 6.78, 8.35, 9.87, 10.78, 11.29, 12.0, 11.09]
y_refactor_purify_5of10_max =  [1.0, 2.94, 5.0,  6.59, 8.28, 8.96, 10.22, 10.34, 10.93, 10.56]
y_refactor_purify_4of10_max =  [1.0, 2.97, 5.0,  6.79, 8.16, 9.27, 8.16, 7.82, 7.47, 6.02]
y_refactor_purify_3of10_max =  [1.0, 2.95, 4.96, 6.56, 7.96, 9.14, 8.64, 7.64, 7.36, 4.54]
y_refactor_purify_2of10_max =  [1.0, 3.0,  4.95, 6.75, 8.32, 9.49, 9.55, 9.73, 9.75, 8.04]
y_refactor_purify_1of10_max =  [1.0, 2.97, 4.96, 6.75, 7.92, 8.09, 7.92, 6.62, 5.85, 4.7]

plt.xlabel('number of coins')
plt.ylabel('collected coins (mean of 100 runs)')
plt.xlim(0, 19)
plt.xticks(np.arange(1,21,2))
plt.ylim(0, 19)
plt.yticks(np.arange(1,21,2))
plt.plot(x, y_ub, label='max score')
plt.plot(x, y_heuristic, label='IL heuristic')
plt.plot(x, y_refactor_purify_6of10_max, label='IL purify')
plt.plot(x, y_refactor_success_max, label='IL successful traj')
plt.plot(x, y_refactor_max, label='IL all traj')
plt.plot(x, y_DDQN, label='DoubleDQN')
plt.legend(loc='upper left')
plt.show()
