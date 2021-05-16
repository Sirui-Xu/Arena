import sys
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from torch.distributions import Normal


class GamePatch(Dataset):
    """Provide patches according to GT boxes or proposals"""

    def __init__(self, data, star_shaped=False, std=None):

        self.data = data

        # Note that it is xywh format.
        self.gt_boxes = []
        self.gt_classes = []

        for data in self.data:
            shape = data["state"]["global"]["shape"]
            boxes = []
            classes = []
            for local_info in data["state"]["local"]:
                box = local_info["position"] + local_info["box"] + local_info["velocity"] + [local_info["speed"], local_info["speed"]]
                box = np.array(box, dtype=np.float32)
                box[::2] /= map_shape[0]
                box[1::2] /= map_shape[1]
                boxes.append(box)
                classes.append(local_info["type_index"])
            self.gt_boxes.append(np.array(boxes, dtype=np.float32))
            self.gt_classes.append(np.array(classes, dtype=np.float32))
                
        self.star_shaped = star_shaped
        self.std = std
        # self.class_dim = self.gt_classes[0].shape[0]
        # self.box_dim = self.gt_boxes[0].shape[0]

    def __getitem__(self, index):
        data = self.data[index]  # {image, annotations, indices}

        boxes = self.gt_boxes[index].copy()
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        # add augmentation
        if self.std:
            std_tensor = boxes_tensor.new_tensor(self.std)
            boxes_tensor = Normal(boxes_tensor, std_tensor).sample()
        
        classes_tensor = torch.tensor(self.gt_classes[index], dtype=torch.float32)

        n = boxes_tensor.size(0)
        if self.star_shaped:
            edge_index = torch.tensor([[0, j] for j in range(1, n)], dtype=torch.long).transpose(0, 1)
            # edge_attr = torch.cat([(boxes_tensor[j] - boxes_tensor[0]).unsqueeze(0) for j in range(1, n)], dim=0)
            edge_attr = torch.tensor([[0] for j in range(1, n)], dtype=torch.float32)
        else:
            edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n)], dtype=torch.long).transpose(0, 1)
            # edge_attr = torch.cat([(boxes_tensor[j] - boxes_tensor[i]).unsqueeze(0) for i in range(n) for j in range(n)], dim=0)
            edge_attr = torch.tensor([[0] for i in range(n) for j in range(n)], dtype=torch.float32)
        # get target
        target = data["action"]
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        out = Data(
            x=classes_tensor,
            y=target,
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            pos=boxes_tensor.float(),
            idx=torch.tensor([index], dtype=torch.int64),  # for visualization and dp
            size=torch.tensor([1], dtype=torch.int64),  # indicate batch size
        )
        return out

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    import os.path as osp
    import json
    data_path = osp.join('../algorithm/result/')
    with open(data_path, 'r') as f:
        data = json.load(f)
    dataset = GamePatch(data)
    node_dim = dataset[0].x[0].shape[0]
    pos_dim = dataset[0].pos[0].shape[0]
    print(len(dataset), node_dim, pos_dim)
    for i in tqdm(range(10)):
        data = dataset[i]
        print(data.x) 
        print(data.y)
        print(data.pos)