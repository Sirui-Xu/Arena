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
                box[::2] /= shape[0]
                box[1::2] /= shape[1]
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

class GamePatchReduce(Dataset):
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
                box[::2] /= shape[0]
                box[1::2] /= shape[1]
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

class GamePatchMaze(Dataset):
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
                box[::2] /= shape[0]
                box[1::2] /= shape[1]
                boxes.append(box)
                classes.append([0] + local_info["type_index"])
            maze = data["state"]["global"]["maze"]
            for x in maze:
                for y in maze:
                    if maze[x, y] != 0:
                        box = [(x+0.5), (y+0.5)] + 
                              [x, y, (x+1), (y+1)] +
                              [0, 0, 0, 0]
                        box[::2] /= maze.shape[0]
                        box[1::2] /= maze.shape[1]
                        boxes.append(box)
                        classes.append([maze[x, y]] + [0 for _ in len(data["state"]["local"][0]["type_index"])])
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

class GamePatchLandmark(Dataset):
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
                box[::2] /= shape[0]
                box[1::2] /= shape[1]
                boxes.append(box)
                classes.append([0] + local_info["type_index"])
            maze = data["state"]["global"]["maze"]
            for x in maze:
                for y in maze:
                    if maze[x, y] != 0:
                        box = [(x+0.5), (y+0.5)] + 
                              [x, y, (x+1), (y+1)] +
                              [0, 0, 0, 0]
                        box[::2] /= maze.shape[0]
                        box[1::2] /= maze.shape[1]
                        boxes.append(box)
                        classes.append([maze[x, y]] + [0 for _ in len(data["state"]["local"][0]["type_index"])])
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


class GamePatchLandmark(Dataset):

    def __init__(self, data, star_shaped=False, std=None):
        
        self.data = data

        self.gt_nodes = []  # graph_node
        self.gt_classes = []    # node attribution
        self.gt_edges = []  # graph edge

        for data in self.data:
            shape = data["state"]["global"]["shape"]
            boxes = []
            classes = []
            for local_info in data["state"]["local"]:
                box = local_info["position"] + local_info["box"] + local_info["velocity"] + [local_info["speed"], local_info["speed"]]
                box = np.array(box, dtype=np.float32)
                box[::2] /= shape[0]
                box[1::2] /= shape[1]
                boxes.append(box)
                classes.append(local_info["type_index"])
            self.gt_boxes.append(np.array(boxes, dtype=np.float32))
            self.gt_classes.append(np.array(classes, dtype=np.float32))


            data['maze'] = np.array(data['maze'])
            width, height = data['maze'].shape

            # Coordinates for "player"
            boxes = [[data["player_x"], data["player_y"]]]

            # node attribution follows [type=1, Detonation time(only for bomb)=-1, Direction_x=0, Direction_y=0]
            classes = [[1, -1, 0, 0]]
            
            # Coordinates for "creep"
            boxes += [data["creep_pos"][i] for i in range(len(data["creep_pos"]))]

            # node attribution follows [type=2, Detonation time(only for bomb)=-1, Direction_x, Direction_y] direction in last frame
            classes += [ [2, -1] + data["creep_dir"][i] for i in range(len(data["creep_dir"]))]
            
            # Coordinates for "bomb"
            boxes += [data["bomb_pos"][i] for i in range(len(data["bomb_pos"]))]

            # node attribution follows [type=3, Detonation time, Direction_x=0, Direction_y=0]
            classes += [[3, data["bomb_life"][i], 0, 0] for i in range(len(data["bomb_pos"]))]

            # ---------------------------------------------------------------------------- #
            # Generate graph nodes
            # ---------------------------------------------------------------------------- #
            coor2index = {}
            gt_class = []
            nodes = []

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    pos = [x, y]

                    if pos in boxes:
                        # Generate landmark to represent the maze
                        nodes.append(pos)
                        coor2index[(x, y)] = [len(nodes) - 1]
                        gt_class.append([0, -1, 0, 0])

                        # Generate special node such as player, creep and bomb
                        k = 0
                        while pos in boxes[k:]:
                            k = boxes.index(pos, k)
                            box = boxes[k]
                            nodes.append(box)
                            coor2index[(x, y)].append(len(nodes) - 1)
                            gt_class.append(classes[k])
                            k += 1
                    
                    # ---------------------------------------------------------------------------- #
                    # Generate landmark to represent the maze
                    # ---------------------------------------------------------------------------- #
                    else:
                        if data['maze'][x, y] == 1: # not a wall
                            continue
                        if data['maze'][x, y - 1] != 1 and data['maze'][x, y + 1] != 1 and data['maze'][x - 1, y] == 1 and data['maze'][x + 1, y] == 1: # not a passageway
                            continue
                        if data['maze'][x - 1, y] != 1 and data['maze'][x + 1, y] != 1 and data['maze'][x, y - 1] == 1 and data['maze'][x, y + 1] == 1: # not a passageway
                            continue

                        # If this location is a corner or a fork in the road, it is landmark
                        nodes.append(pos)
                        coor2index[(x, y)] = [len(nodes) - 1]
                        gt_class.append([0, -1, 0, 0])
            
            # ---------------------------------------------------------------------------- #
            # Generate graph edges
            # ---------------------------------------------------------------------------- #

            edges = []

            # Connect all adjacent nodes
            
            # Connect edges in the Y direction
            flag = True
            for x in range(width):
                last_pos = None
                for y in range(height):
                    pos = (x, y)

                    # There is a wall
                    if data['maze'][x, y] == 1:
                        flag = False

                    elif pos in coor2index.keys():
                        
                        # There is no wall between the two sides
                        if flag == True:
                            assert last_pos is not None
                            # connect two adjacent landmark
                            edges.append((coor2index[last_pos][0], coor2index[pos][0]))
                            
                            # connect the all special node with landmark
                            for index in coor2index[pos][1:]:
                                edges.append((coor2index[pos][0], index))
                        
                        # only the first side
                        else:
                            flag = True

                            # connect the all special node with landmark
                            for index in coor2index[pos][1:]:
                                edges.append((coor2index[pos][0], index))

                        last_pos = (x, y)

            # Connect edges in the X direction
            flag = True
            for y in range(data['maze'].shape[1]):
                last_pos = None
                for x in range(data['maze'].shape[0]):
                    pos = (x, y)

                    # There is a wall
                    if data['maze'][x, y] == 1:
                        flag = False

                    elif pos in coor2index.keys():

                        # There is no wall between the two sides
                        if flag == True:
                            assert last_pos is not None
                            edges.append((coor2index[last_pos][0], coor2index[pos][0]))

                        else:
                            flag = True
                        last_pos = (x, y)

            self.gt_classes.append(gt_class)

            # print(boxes, edges, data['maze'])

            nodes = np.array(nodes, dtype=np.float32)
            self.gt_nodes.append(nodes)
            self.gt_edges.append(edges)

        self.std = std
        self.n_actions = n_actions

    def __getitem__(self, index):
        data = self.data[index]

        nodes = self.gt_nodes[index].copy()
        edges = self.gt_edges[index].copy()

        # normalize
        nodes_tensor = torch.tensor(nodes, dtype=torch.float32)

        # add augmentation
        if self.std:
            std_tensor = nodes_tensor.new_tensor(self.std)
            nodes_tensor = Normal(nodes_tensor, std_tensor).sample()
        classes = torch.Tensor(self.gt_classes[index])
        edge_index = torch.tensor(edges, dtype=torch.long).transpose(0, 1)
        edge_attr = torch.cat([(nodes_tensor[edges[i][1]] - nodes_tensor[edges[i][0]]).unsqueeze(0) for i in range(len(edges))], dim=0)

        # get target
        danger_scores = data.get("danger_scores", None)
        explosion_scores = data.get("explosion_scores", None)
        _action = data.get("action", None)
        distance = data.get("distance", None)
        _direction = data.get("direction", None)
        action, direction = None, None
        y = None
        if _action is not None:
            action = torch.zeros(1, self.n_actions, dtype=torch.float32)
            action[0, _action % self.n_actions] = 1
            y = action.clone().detach()
        if distance is not None:
            distance = torch.Tensor([float(distance)]).unsqueeze_(0)
        if _direction is not None:
            direction = torch.zeros(1, 4, dtype=torch.float32)
            direction[0, _direction % 4] = 1
        if explosion_scores is not None:
            explosion_scores = torch.Tensor([explosion_scores]).unsqueeze_(0)
        if danger_scores is not None:
            danger_scores = torch.Tensor([danger_scores]).unsqueeze_(0)

        # print(
        #     action,
        #     distance,
        #     direction,
        #     explosion_scores,
        #     danger_scores,
        # )
        out = Data(
            x=classes,
            y=y,
            action=action,
            distance=distance,
            direction=direction,
            explosion_scores=explosion_scores,
            danger_scores=danger_scores,
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            pos=nodes_tensor.float(),
            idx=torch.tensor([index], dtype=torch.int64),  # for visualization and dp
            size=torch.tensor([1], dtype=torch.int64),  # indicate batch size
        )
        return out

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    import os.path as osp
    import json
    data_path = osp.join('../algorithm/result/data/waterworld_greedycollectv1_[7]_[3, 5, 7, 9]_[20]_513.json')
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