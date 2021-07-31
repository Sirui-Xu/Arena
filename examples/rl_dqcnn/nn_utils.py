import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class BaseCNN(nn.Module):
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

class PlainCNN(BaseCNN):
    """Plain CNN (for 128x128 images)"""

    def __init__(self,
                 in_channels=3,
                 output_dim=3,
                 max_pooling=True,
                 ):
        super().__init__()

        self.output_dim = output_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(128, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Linear(128, 256, bias=True), nn.ReLU(inplace=True),
                nn.Linear(256, output_dim)
            )
        else:
            self.pool = None
            self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256, bias=True), nn.ReLU(inplace=True),
                nn.Linear(256, output_dim)
            )

        self.reset_parameters()

    def forward(self, data_batch):
        image = data_batch
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        logits = self.fc(x)

        return logits

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, key, query, value):
        b, d, n = key.size()
        _, _, m = query.size()
        _, do, _ = value.size()
        key = key.reshape(b * self.num_heads, d // self.num_heads, n)
        query = query.reshape(b * self.num_heads, d // self.num_heads, m)
        value = value.reshape(b * self.num_heads, do // self.num_heads, m)
        affinity = torch.bmm(key.transpose(1, 2), query)  # (b * nh, n, m)
        weight = torch.softmax(affinity / math.sqrt(d), dim=1)  # (b * nh, n, m)
        output = torch.bmm(value, weight)  # (b * nh, do // nh, m)
        output = output.reshape(b, -1, m)  # (b, nh * do, m)
        weight = weight.reshape(b, self.num_heads, n, m)  # (b, nh, n, m)
        return output, weight


class RelationNet(BaseCNN):
    def __init__(self,
                 in_channels=3,
                 output_dim=3,
                 max_pooling=True,
                 ):
        super().__init__()

        self.output_dim = output_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 62, 3, padding=1, bias=True), nn.ReLU(inplace=True),
        )

        identity = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=torch.float32)
        # (1, h, w, 2)
        grid = F.affine_grid(identity, [1, 1, 8, 8])
        grid = grid.permute(0, 3, 1, 2).contiguous()
        self.register_buffer('grid', grid)

        fs = (64, 8, 8)

        self.key = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.LayerNorm(fs), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, bias=True),
        )

        self.query = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.LayerNorm(fs), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, bias=True),
        )

        self.value = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.LayerNorm(fs), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, bias=False), nn.LayerNorm(fs), nn.ReLU(inplace=True),
        )
        self.attention = MultiheadAttention(num_heads=4)

        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=True), nn.ReLU(inplace=True),
        )

        self.cnn2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(64, 128, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Linear(128, 256, bias=True), nn.ReLU(inplace=True),
                nn.Linear(256, output_dim)
            )
        else:
            self.pool = None
            self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256, bias=True), nn.ReLU(inplace=True),
                nn.Linear(256, output_dim)
            )

        self.reset_parameters()

    def forward(self, data_batch):
        image = data_batch
        x = self.cnn(image)

        b, c1, h1, w1 = x.size()
        x = torch.cat([x, self.grid.expand(b, -1, -1, -1)], dim=1)
        # (b, dk, h1 * w1)
        key = self.key(x).flatten(2)
        query = self.query(x).flatten(2)
        value = self.value(x).flatten(2)
        attn_output, attn_weight = self.attention(key, query, value)
        attn_output = attn_output.reshape(b, -1, h1, w1)
        x = torch.cat([x, attn_output], dim=1)
        x = self.conv(x)
        x = self.cnn2(x)

        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        logits = self.fc(x)

        return logits