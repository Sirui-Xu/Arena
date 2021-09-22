import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List
import copy

import torch_geometric.nn as gnn
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import Batch as GBatch
from torch_scatter import scatter
from torch_geometric.nn import GENConv, DeepGCNLayer, global_max_pool, global_add_pool, EdgeConv
from arena import Wrapper
from tianshou.data import Batch as TSBatch
from ..dqn_tianshou.graph_batch import Batch as TSGBatch

def get_nn_func(nn_name):
    if nn_name=='PointConv':
        return PointConv
    elif nn_name=='EdgeConvNet':
        return EdgeConvNet
    elif nn_name=='TSPointConv':
        return TSPointConv
    else:
        raise NotImplementedError("Unrecognized nn name")

class MyPointConv(gnn.PointConv):
    def __init__(self, aggr, local_nn=None, global_nn=None, **kwargs):
        super(gnn.PointConv, self).__init__(aggr=aggr, **kwargs)
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.reset_parameters()
        self.add_self_loops = True

class PointConv(nn.Module):

    def __init__(self, aggr='add', input_dim=4, pos_dim=4,
                 edge_dim=None, output_dim=6, dropout=True): # edge_dim is not used!
        super().__init__()

        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.encoder_pos = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        local_nn = nn.Sequential(
            nn.Linear(32 + 64 + pos_dim, 512, bias=False), nn.GroupNorm(8, 512), nn.ReLU(),
            nn.Linear(512, 512, bias=False), nn.GroupNorm(8, 512), nn.ReLU(),
            nn.Linear(512, 512),
        )
        global_nn = nn.Sequential(
            nn.Linear(512, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256),
        )
        gate_nn = nn.Sequential(
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        self.gnn = MyPointConv(aggr=aggr, local_nn=local_nn, global_nn=global_nn)
        self.aggr = aggr
        self.attention = gnn.GlobalAttention(gate_nn)

        self.reset_parameters()

    def forward(self, data, **kwargs):
        batch_size = data.num_graphs
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos

        x = self.encoder(x)
        pos_feature = self.encoder_pos(pos)
        task_feature = x
        feature = torch.cat([task_feature, pos_feature], dim=1)
        x = self.gnn(x=feature, pos=pos, edge_index=edge_index)
        x = self.attention(x, batch, size=batch_size)
        if self.dropout:
            x = F.dropout(x, p=0.1, training=self.training)
        q = self.decoder(x)
        return q

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class TSPointConv(nn.Module):

    def __init__(self, aggr='add', input_dim=4, pos_dim=4,
                 edge_dim=None, output_dim=6, dropout=True, device=None): # edge_dim is not used!
        super().__init__()

        self.dropout = dropout
        self.device=device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.encoder_pos = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        local_nn = nn.Sequential(
            nn.Linear(32 + 64 + pos_dim, 512, bias=False), nn.GroupNorm(8, 512), nn.ReLU(),
            nn.Linear(512, 512, bias=False), nn.GroupNorm(8, 512), nn.ReLU(),
            nn.Linear(512, 512),
        )
        global_nn = nn.Sequential(
            nn.Linear(512, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256),
        )
        gate_nn = nn.Sequential(
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256, bias=False), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.GroupNorm(8, 256), nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        self.gnn = MyPointConv(aggr=aggr, local_nn=local_nn, global_nn=global_nn)
        self.aggr = aggr
        self.attention = gnn.GlobalAttention(gate_nn)

        self.reset_parameters()

    def forward(self, data, state=None, info={}):
        #assert isinstance(data, List) and len(data)>0 and isinstance(data[0], Data)
        assert isinstance(data, np.ndarray) and data.shape[0]>0 and isinstance(data[0], Data)
        data = GBatch.from_data_list(data).to(self.device)
        batch_size = data.num_graphs
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos

        x = self.encoder(x)
        pos_feature = self.encoder_pos(pos)
        task_feature = x
        feature = torch.cat([task_feature, pos_feature], dim=1)
        x = self.gnn(x=feature, pos=pos, edge_index=edge_index)
        x = self.attention(x, batch, size=batch_size)
        if self.dropout:
            x = F.dropout(x, p=0.1, training=self.training)
        q = self.decoder(x)
        return q, state

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class BaseGNN(nn.Module):
    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def compute_losses(self, pd_dict: dict, data_batch: dict, **kwargs):
        loss_dict = dict()
        ce_loss = F.cross_entropy(pd_dict['logits'], data_batch['action'])
        loss_dict['ce_loss'] = ce_loss
        return loss_dict

class EdgeConvNet(BaseGNN):
    def __init__(self, aggr='max', input_dim=4, pos_dim=4, edge_dim=None, output_dim=6):
        super(BaseGNN, self).__init__()

        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.encoder_pos = nn.Sequential(
            nn.Linear(pos_dim, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )

        local_nn = nn.Sequential(
            nn.Linear((64+64) * 2, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
        )
        self.gnn = EdgeConv(local_nn, aggr='add')

        self.encoder2 = nn.Sequential(
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
        )

        self.global_aggr = aggr

        self.fc = nn.Sequential(
            nn.Linear(128, 128, bias=True), nn.ReLU(),
            nn.Linear(128, 128, bias=True), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.reset_parameters()

    def forward(self, data, batch_size=None, **kwargs):
        assert batch_size is not None

        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos

        x_feat = self.encoder(x)
        pos_feat = self.encoder_pos(pos)
        x = torch.cat([x_feat, pos_feat], dim=1)

        x = self.gnn(x=x, edge_index=edge_index)
        x = self.encoder2(x)

        if self.global_aggr == 'max':
            global_feature = gnn.global_max_pool(x, batch, size=batch_size)
        elif self.global_aggr == 'sum':
            global_feature = gnn.global_add_pool(x, batch, size=batch_size)
        else:
            raise NotImplementedError()

        logits = self.fc(global_feature)

        return logits

class EnvStateProcessor:
    def __init__(self, env_kwargs, use_tianshou=False):
        self.env_kwargs=env_kwargs
        self.use_tianshou = use_tianshou

    def process_state(self, state):
        obj_size = self.env_kwargs['object_size']
        local_state = state['local']
        num_objs = len(local_state)
        edges = []
        for i in range(num_objs):
            for j in range(num_objs):
                if i != j:
                    edges.append((i,j))
        edges = (np.array(edges).T).astype(np.int64)
        edge_index = torch.from_numpy(edges)

        x, pos = [], []
        #'''
        p=None
        for node in local_state:
            if node['type'] == 'agent':
                p = node['position']
        if p is None:
            for node in local_state:
                rel_pos = np.array(node['position']) / obj_size # Hard code the size of an object.
                x.append(node['type_index'] + node['velocity'] + rel_pos.tolist())
                pos.append(node['position'] + rel_pos.tolist())
        else:
            for node in local_state:
                rel_pos = (np.array(node['position']) - np.array(p)) / obj_size # Hard code the size of an object.
                x.append(node['type_index'] + node['velocity'] + rel_pos.tolist())
                pos.append(node['position'] + rel_pos.tolist())

        x = torch.tensor(x)
        pos = torch.tensor(pos)

        if not self.use_tianshou:
            return Data(x=x, edge_index=edge_index, pos=pos)
        else:
            return TSBatch(x=x, edge_index=edge_index, pos=pos)

class GraphObservationEnvWrapper(Wrapper):
    def __init__(self, env_func, env_kwargs, seed=233):
        super().__init__(env_func(**env_kwargs), rng=seed)
        self.state_processor = EnvStateProcessor(env_kwargs)
        self._state_raw = None
        self._last_score = None

    def reset(self):
        self._state_raw = super().reset()
        state = self.state_processor.process_state(self._state_raw)
        return state

    def step(self, action):
        state_raw, reward, done, info = super().step(action)
        self._last_score = super().score()
        if(done):
            state = self.reset()
        else:
            self._state_raw = state_raw
            state=self.state_processor.process_state(self._state_raw)
        return state, reward, done, info

    def score(self):
        return self._last_score


class EnvStateProcessorSirui:
    """Provide patches according to GT boxes or proposals"""

    def __init__(self, env_kwargs, use_tianshou=False, star_shaped=False, std=False):
        self.env_kwargs=env_kwargs
        self.use_tianshou = use_tianshou
        self.star_shaped = star_shaped
        self.std = std

    def process_state(self, state):
        data = state

        shape = data["global"]["shape"]
        boxes = []
        classes = []
        for local_info in data["local"]:
            box = local_info["position"] + local_info["box"] + local_info["velocity"] + [local_info["speed"], local_info["speed"]]
            box = np.array(box, dtype=np.float32)
            box[::2] /= shape[0]
            box[1::2] /= shape[1]
            boxes.append(box)
            classes.append(local_info["type_index"])
        gt_boxes = np.array(boxes, dtype=np.float32)
        gt_classes = np.array(classes, dtype=np.float32)

        boxes = gt_boxes.copy()
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        # add augmentation
        #if self.std:
        #    std_tensor = boxes_tensor.new_tensor(self.std)
        #    boxes_tensor = Normal(boxes_tensor, std_tensor).sample()

        classes_tensor = torch.tensor(gt_classes, dtype=torch.float32)

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
        out = Data(
            x=classes_tensor,
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            pos=boxes_tensor.float(),
        )
        return out

class GraphObservationEnvSiruiWrapper(Wrapper):
    def __init__(self, env_func, env_kwargs, seed=233, random_reset=True):
        super().__init__(env_func(**env_kwargs), rng=seed)
        self.state_processor = EnvStateProcessorSirui(env_kwargs, star_shaped=True)
        self._init_state_raw = super().reset()
        self._state_raw = None
        self._last_score = None
        self.random_reset=random_reset

    def reset(self):
        if self.random_reset:
            self._state_raw = super().reset()
        else:
            super().reset()
            super().setGameState(self._init_state_raw)
            #self._state_raw = copy.deepcopy(self._init_state_raw)
            self._state_raw = super().getGameState()
        state = self.state_processor.process_state(self._state_raw)
        return state

    def step(self, action):
        state_raw, reward, done, info = super().step(action)
        self._last_score = super().score()
        if(done):
            state = self.reset()
        else:
            self._state_raw = state_raw
            state=self.state_processor.process_state(self._state_raw)
        return state, reward, done, info

    def score(self):
        return self._last_score

import gym
class GraphObservationEnv(gym.Env):
    def __init__(self, game_func, env_kwargs):
        super().__init__()
        self.game = game_func(env_kwargs)
        self.action_space = gym.spaces.Discrete(6)
        self.state_processor = EnvStateProcessor(env_kwargs, use_tianshou=False)
        self._state_raw = None
        self._last_score = None

    def reset(self):
        self._state_raw = self.game.reset()
        state = self.state_processor.process_state(self._state_raw)
        return state

    def step(self, action):
        state_raw, reward, done, info = self.game.step(action)
        self._last_score = self.game.score()
        if(done):
            state = self.reset()
        else:
            self._state_raw = state_raw
            state=self.state_processor.process_state(self._state_raw)
        return state, reward, done, info

    def score(self):
        return self._last_score

import pickle, os
class ExperienceSaver:
    def __init__(self, save_path):
        self.save_path = save_path
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)
        self.traj_cnt=0
        self.current_traj = []

    def store(self, s, a, r):
        self.current_traj.append({'s': s, 'a': a, 'r': r})

    def close_traj(self):
        fname = self.save_path + f'/traj_{self.traj_cnt}.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(self.current_traj, f, pickle.HIGHEST_PROTOCOL)
        self.current_traj = []
        self.traj_cnt+=1

class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
