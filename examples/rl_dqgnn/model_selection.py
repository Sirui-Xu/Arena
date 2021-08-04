import os,sys
import argparse
from parse import parse

dqgnn_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.dirname(os.path.dirname(dqgnn_path))
sys.path.append(root_path)
from examples.rl_dqgnn.eval_dqgnn import GNNQEvaluator
from matplotlib import pyplot as plt
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--num_trajs', type=int, default=40)
    parser.add_argument('--env_setting', type=str, default='legacy')
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--nn_name', type=str, default="PointConv")
    parser.add_argument('--num_rewards', type=int, default=5)
    args = parser.parse_args()

    exit()

    files = os.listdir(args.model_dir)
    best_model_fname = None
    best_model_perf = -1
    results = {}
    for file in files:
        if file.endswith(".pth"):
            ckpt_name = parse("ep{}.pth", file)
            if ckpt_name is None:
                continue
            ep_id = int(ckpt_name[0])
            evaluator = GNNQEvaluator(model_path=args.model_dir+file, nn_name=args.nn_name,
                                      env_setting=args.env_setting, num_trajs=args.num_trajs,
                                      eps=args.eps)
            evaluator.update_num_coins(args.num_rewards)
            eval_result = evaluator.evaluate()
            if eval_result['score'] > best_model_perf:
                best_model_perf = eval_result['score']
                best_model_fname = file

            results[ep_id] = eval_result['score']
    print('best model:', best_model_fname)
    print('best score:', best_model_perf)

    x=[k for(k,v) in results.items()]
    y=[v for(k,v) in results.items()]

    plt.plot(x,y)
    plt.savefig(args.model_dir+f'eval_plot_{args.eps}.png')
    with open(args.model_dir+f'eval_result_{args.eps}.pkl', 'wb') as f:
        pickle.dump(results, f)

