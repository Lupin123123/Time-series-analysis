import torch
import torch.nn as nn

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., pred=True):
        super(Mlp, self).__init__()
        #out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.pred = pred
        if pred==True:
            self.fc2 = nn.Linear(hidden_features, 1)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        q = self.q(x)
        # print(q.shape)
        k = self.k(x)
        # print(k.shape)
        v = self.v(x)
        # print(v.shape)
        attn = (q @ k.transpose(-2, -1))
        # print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(-1)
        # print(x.size())
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.pred==False:
            x += x1
        return x


class TF(nn.Module):
    def __init__(self, in_features, drop=0.):
        super().__init__()
        self.Block1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.Block2 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.in_features = in_features
    def forward(self, x):
        x0 = self.Block1(x)
        # print(x0.shape)
        x1 = self.Block2(x0)
        # print(x1.shape)
        x2 = torch.flatten(x1, start_dim=1)
        # print(x2.shape)
        x3 = nn.Linear(in_features=x2.size(-1), out_features=8)(x2)
        return x3.reshape(-1, 1, self.in_features)
