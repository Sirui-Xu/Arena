CUDA_VISIBLE_DEVICES=3 python test.py \
  --num_coins_list 15 17 19 --num_enemies_list 0 --num_obstacles_list 0 --num_bombs 0 \
  --checkpoints_path /home/yiran/pc_mapping/arena-v2/examples/bc_saved_models/refactor_success_max_mine/run1 \
  --aggr max --resume_epoch 10