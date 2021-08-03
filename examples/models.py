import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter
from torch_geometric.nn import GENConv, DeepGCNLayer, global_max_pool, global_add_pool
from torch_geometric.nn import EdgeConv as EdgeConvGNN

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

class EdgeConvTongzhou(BaseGNN):
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
        self.gnn = EdgeConvGNN(local_nn, aggr='max')

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
        if batch_size is None:
            batch_size = data['size'].sum().item()

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

        out_dict = {
            'q': logits
        }
        return out_dict

class EdgeConv(nn.Module):
    def __init__(self, aggr='max', input_dim=1, pos_dim=4, edge_dim=4, output_dim=4):
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
        self.gnn = gnn.EdgeConv(local_nn, aggr='add')
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

class MyPointConv(gnn.PointConv):
    def __init__(self, aggr, local_nn=None, global_nn=None, **kwargs):
        super(gnn.PointConv, self).__init__(aggr=aggr, **kwargs)
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.reset_parameters()
        self.add_self_loops = True


class PointConv(nn.Module):
    
    def __init__(self, aggr='add', input_dim=1, pos_dim=4, edge_dim=4, output_dim=5):
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
        self.attention = gnn.GlobalAttention(gate_nn)

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
        x = self.attention(x, batch, size=batch_size)
        x = F.dropout(x, p=0.1, training=self.training)
        q = self.decoder(x)
        '''
        if self.aggr == 'max':
            q = gnn.global_max_pool(x, batch, size=batch_size)
        elif self.aggr == 'sum':
            q = gnn.global_add_pool(x, batch, size=batch_size)
        else:
            raise NotImplementedError()
        '''
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