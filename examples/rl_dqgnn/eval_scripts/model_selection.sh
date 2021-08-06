#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python model_selection.py --num_rewards 5 \
  --model_dir /home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/saved_models/AX0_exp/run9/ \
   --env_setting AX0 --eps 0.0;

CUDA_VISIBLE_DEVICES=2 python model_selection.py --num_rewards 5 \
  --model_dir /home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/saved_models/AX0_Jul28/run2/ \
   --env_setting AX0 --eps 0.3;