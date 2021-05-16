import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import EdgeConv, PointConv
from torch_scatter import scatter
from torch_geometric.nn import GENConv, DeepGCNLayer, global_max_pool, global_add_pool

class EdgeConv(nn.Module):
    def __init__(self, aggr='max', input_dim=1, edge_dim=4, pos_dim=4, output_dim=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.encoder_pos = nn.Sequential(
            nn.Linear(pos_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )
        local_nn = nn.Sequential(
            nn.Linear((32 + 32) * 2, 1024, bias=False), nn.GroupNorm(16, 1024), nn.ReLU(),
            nn.Linear(1024, 1024, bias=False), nn.GroupNorm(16, 1024), nn.ReLU(),
            nn.Linear(1024, output_dim),
        )
        self.gnn = EdgeConv(local_nn, aggr='add')
        self.aggr = aggr

        self.reset_parameters()

    def forward(self, data, **kwargs):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos

        # infer real batch size
        batch_size = data['size'].sum().item()
        x = self.encoder(x)
        pos_feature = self.encoder_pos(pos)
        task_feature = x
        feature = torch.cat([task_feature, pos_feature], dim=1)
        x = self.gnn(x=feature, edge_index=edge_index)

        if self.aggr == 'max':
            q = gnn.global_max_pool(x, batch, size=batch_size)
        elif self.aggr == 'sum':
            q = gnn.global_add_pool(x, batch, size=batch_size)
        else:
            raise NotImplementedError()

        out_dict = {
            'q': q,
            'feature': feature,
        }
        return out_dict

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class DeeperGCN(torch.nn.Module):
    def __init__(self, aggr='max', input_dim=1, pos_dim=4, edge_dim=4, output_dim=5, hidden_channels=16, num_layers=8):
        super(DeeperGCN, self).__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.GroupNorm(1, 16), nn.ReLU(),
            nn.Linear(16, hidden_channels)
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, 16), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Linear(16, hidden_channels)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 32), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Linear(32, 2*hidden_channels)
        )

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = GENConv(2*hidden_channels, 2*hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(2*hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.decoder = nn.Linear(2*hidden_channels, output_dim)
        self.aggr = aggr

        self.reset_parameters()

    def forward(self, data, batch_size=None):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        pos = data.pos

        # infer real batch size
        if batch_size is None:
            batch_size = data['size'].sum().item()

        x = self.node_encoder(x)
        pos_attr = self.pos_encoder(pos)
        # print(x.shape, pos_attr.shape)
        x = torch.cat([x, pos_attr], dim=1)
        edge_attr = self.edge_encoder(edge_attr)
        # print(x.shape, edge_attr.shape, edge_index.shape)
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            # print(x.shape, edge_attr.shape)
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.decoder(x)
        if self.aggr == 'max':
            q = global_max_pool(x, batch, size=batch_size)
        elif self.aggr == 'sum':
            q = global_add_pool(x, batch, size=batch_size)
        else:
            raise NotImplementedError()

        out_dict = {
            'q': q,
        }

        return out_dict

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class MyPointConv(PointConv):
    def __init__(self, aggr, local_nn=None, global_nn=None, **kwargs):
        super(PointConv, self).__init__(aggr=aggr, **kwargs)
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.reset_parameters()
        self.add_self_loops = True


class PointConv(nn.Module):
    
    def __init__(self, aggr='max', input_dim=1, edge_dim=4, pos_dim=4, output_dim=5):
        super().__init__()

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
            nn.Linear(64 + 64 + 4, 1024, bias=False), nn.GroupNorm(16, 1024), nn.ReLU(),
            nn.Linear(1024, 1024, bias=False), nn.GroupNorm(16, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        global_nn = nn.Sequential(
            nn.Linear(1024, 1024, bias=False), nn.GroupNorm(16, 1024), nn.ReLU(),
            nn.Linear(1024, 1024, bias=False), nn.GroupNorm(16, 1024), nn.ReLU(),
            nn.Linear(1024, output_dim),
        )
        self.gnn = MyPointConv(aggr='add', local_nn=local_nn, global_nn=global_nn)
        self.aggr = aggr

        self.reset_parameters()

    def forward(self, data, batch_size=None, **kwargs):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos

        # infer real batch size
        if batch_size is None:
            batch_size = data['size'].sum().item()

        x = self.encoder(x)
        pos_feature = self.encoder_pos(pos)
        task_feature = x
        feature = torch.cat([task_feature, pos_feature], dim=1)
        # print(x.shape, pos.shape, pos_feature.shape, feature.shape)

        x = self.gnn(x=feature, pos=pos, edge_index=edge_index)
        if self.aggr == 'max':
            q = gnn.global_max_pool(x, batch, size=batch_size)
        elif self.aggr == 'sum':
            q = gnn.global_add_pool(x, batch, size=batch_size)
        else:
            raise NotImplementedError()

        out_dict = {
            'q': q,
            'task_feature': task_feature,
        }

        return out_dict

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)