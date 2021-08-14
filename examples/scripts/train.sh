CUDA_VISIBLE_DEVICES=3 python train.py \
  --data_path /home/yiran/pc_mapping/arena-v2/examples/rl_dqgnn/saved_models/AX0_double/run7/traj_success.json \
  --checkpoints_path /home/yiran/pc_mapping/arena-v2/examples/bc_saved_models/refactor_success_add/run3 \
   --aggr add --num_epochs 25

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path /home/yiran/pc_mapping/arena-v2/examples/bc_filtered_data/5of10.json \
  --checkpoints_path /home/yiran/pc_mapping/arena-v2/examples/bc_saved_models/refactor_success_max_redo/5of10 \
   --aggr max