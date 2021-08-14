import matplotlib.pyplot as plt
import numpy as np
import os.path as osp

runs_dir = '/home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/saved_models/AX0_double_rerun'
scores = []
max_step=2500
for i in range(10):
    current_run_dir = osp.join(runs_dir, f"run{i}")
    current_scores = np.load(osp.join(current_run_dir, "score.npy"))
    scores.append(current_scores[:max_step])
    print(current_scores[-1])

scores = np.array(scores)
scores_mean = np.mean(scores, axis=0)
scores_std = np.std(scores, axis=0)

x=np.arange(max_step)
plt.ylim(0,5)
plt.xlim(0,10000)
plt.plot(x,scores_mean)
plt.fill_between(x, scores_mean - scores_std, scores_mean + scores_std, alpha=0.2)
plt.show()
