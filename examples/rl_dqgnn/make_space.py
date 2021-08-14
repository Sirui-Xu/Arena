import os,sys
import json

potential_dirs = [x[0] for x in os.walk('/home/yiran/pc_mapping/arena-v2/examples/bc_saved_models/')]
#potential_dirs = [x[0] for x in os.walk('/home/yiran/pc_mapping/arena-v2/examples/bc_saved_models/IL_greedy_edgeconv')]
for dir in potential_dirs:
    info_path = os.path.join(dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        info['data'] = None
        with open(info_path, 'w') as f:
            json.dump(info, f)
