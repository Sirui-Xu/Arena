import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torch_geometric.nn as gnn
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from torch_geometric.nn import GENConv, DeepGCNLayer, global_max_pool, global_add_pool, EdgeConv

def get_nn_func(nn_name):
    if nn_name=='PointConv':
        return PointConv
    elif nn_name=='EdgeConvNet':
        return EdgeConvNet
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

    def __init__(self, aggr='add', input_dim=4, pos_dim=4, edge_dim=None, output_dim=6): # edge_dim is not used!
        super().__init__()

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

    def forward(self, data, batch_size=None, **kwargs):
        assert batch_size is not None
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
        x = F.dropout(x, p=0.1, training=self.training)
        q = self.decoder(x)
        return q

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
        self.gnn = EdgeConv(local_nn, aggr='max')

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
    def __init__(self, env_kwargs):
        self.env_kwargs=env_kwargs

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
                rel_pos = np.array(node['position']) - np.array(p) / obj_size # Hard code the size of an object.
                x.append(node['type_index'] + node['velocity'] + rel_pos.tolist())
                pos.append(node['position'] + rel_pos.tolist())

        x = torch.tensor(x)
        pos = torch.tensor(pos)
        return Data(x=x, edge_index=edge_index, pos=pos)