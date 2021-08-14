#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python /home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/train_dqgnn.py \
   --model_path /home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/saved_models/AX0_double_assure/run2 \
    --num_episodes 10000 --env_setting AX0 --gnn_aggr add