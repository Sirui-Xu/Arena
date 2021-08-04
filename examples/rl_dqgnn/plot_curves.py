from matplotlib import pyplot as plt
import pickle
import numpy as np

exp_root = '/home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/saved_models/'
exp_names = ['AX0_old', 'AX0_Jul25', 'AX0_Jul28']
runs = ['run0', 'run1', 'run2']

exps = []

for exp_name in exp_names:
    for run in runs:
        exp = exp_root + f"{exp_name}/{run}/"
        exps.append(exp)

xs = []
ys = []

for exp in exps:
    eval_result_fname = exp + 'eval_result_0.0.pkl'
    with open(eval_result_fname, 'rb') as f:
        eval_result = pickle.load(f)
    x=[k for (k,v) in eval_result.items()]
    y=[v for (k,v) in eval_result.items()]
    x=np.array(x)
    y=np.array(y)
    rank=np.argsort(x)

    print(f'exp: {exp}, best score: {np.sort(y)[-1]}')

    xs.append(x[rank])
    ys.append(y[rank])

xs, ys=np.array(xs), np.array(ys)

x=xs[0]
ymean, ystd=ys.mean(axis=0), ys.std(axis=0)

for i in range(9):
    plt.plot(x,ys[i])

#plt.plot(x,ymean)
#plt.fill_between(x, ymean - ystd, ymean + ystd, alpha=0.2)
plt.show()

