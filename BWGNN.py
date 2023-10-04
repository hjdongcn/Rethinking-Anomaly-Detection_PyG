import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from torch_geometric.nn import GCNConv
import copy
from torch_geometric.utils import get_laplacian


def calculate_theta2(d):
    thetas = [] # 创建一个空列表
    x = sympy.symbols('x') #创建一个符号
    for i in range(d+1): # 循环d+1次
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i))) # 构建了一个多项式
        coeff = f.all_coeffs() # 获取f的系数
        inv_coeff = [] # 空列表
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i])) # 添加到inv中
        thetas.append(inv_coeff) # 添加到thetas中
    return thetas

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, h, edge_index):
        def unnLaplacian(feat, edge_index):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            lap_, lap_w = get_laplacian(edge_index, normalization='sym')

            lap = torch.sparse_coo_tensor(lap_, lap_w, size=(feat.size()[0], feat.size()[0]))
            res = torch.sparse.mm(lap, feat)
            # print(res.size())
            # exit()
            return res
            
        feat = h.clone()
        h = self._theta[0] * feat
        for k in range(1, self._k):
            feat = unnLaplacian(feat, edge_index)
            h += self._theta[k] * feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, d=2, batch=False):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
            # if not batch:
            #     self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
            # else:
            #     self.conv.append(PolyConvBatch(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat, edge_index):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(h, edge_index)
            h_final = torch.cat([h_final, h0], -1)
            # print(f"{h_final.shape=}")
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h