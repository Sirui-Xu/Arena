CUDA_VISIBLE_DEVICES=2 python /home/yiran/pc_mapping/arena-v2/examples/rl_ppo/train.py \
  --seed 235 --save_path /home/yiran/pc_mapping/arena-v2/examples/rl_ppo/saved_models/AX0_v1e-3_pi1e-5_235 \
  --v_lr 0.001 --pi_lr 0.00001 --env_setting AX0 --steps 1200 --epochs 2000