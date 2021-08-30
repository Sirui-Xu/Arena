CUDA_VISIBLE_DEVICES=0 python /home/yiran/pc_mapping/arena-v2/examples/rl_ppo/train.py \
  --seed 233 --save_path /home/yiran/pc_mapping/arena-v2/examples/rl_ppo/saved_models/AX0_v1e-3_pi1e-5_233 \
  --v_lr 0.001 --pi_lr 0.00001 --env_setting AX0_fast --steps 600 --epochs 2000