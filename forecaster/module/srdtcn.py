# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import division

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.nn import init


# 计算两个矩阵之间的 Frobenius 范数距离，并归一化该距离
def gdistance_fro(A1, A2):
    return torch.norm(A1 - A2, p="fro") / A1.numel()


def gdistance_kl(A1, A2):
    A1 = torch.distributions.bernoulli.Bernoulli(A1)
    A2 = torch.distributions.bernoulli.Bernoulli(A2)
    return kl_divergence(A1, A2).mean() + kl_divergence(A2, A1).mean()


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncwl,vw->ncvl", (x, A))
        return x.contiguous()


class dyna_nconv(nn.Module):
    def __init__(self):
        super(dyna_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncwl,nvw->ncvl", (x, A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,nvwl->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class dyna_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dyna_mixprop, self).__init__()
        self.nconv = dyna_nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(1)).to(x.device)
        d = torch.sum(adj, dim=2, keepdim=True)
        h = x
        out = [h]
        a = adj / d
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class dy_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        # b, c, n ,t
        # adj = adj + torch.eye(adj.size(0)).to(x.device)
        # d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception_ablation(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2, padding=False):
        super(dilated_inception_ablation, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            if not padding:
                self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
            else:
                self.tconv.append(
                    nn.Conv2d(
                        cin, cout, (1, kern), dilation=(1, dilation_factor), padding=(0, (kern - 1) * dilation_factor)
                    ),
                )

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3) :]
        x = torch.cat(x, dim=1)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3) :]
        x = torch.cat(x, dim=1)
        return x


class dyna_graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3):
        super(dyna_graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha

    def forward(self, idx, emb):
        nodevec2 = nodevec1 = emb

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(1, 2)) - torch.matmul(nodevec2, nodevec1.transpose(1, 2))
        adj = F.relu(torch.tanh(self.alpha * a))  # B, N, N
        mask = torch.zeros(emb.size(0), idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 2)
        mask.scatter_(2, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx, emb):
        nodevec2 = nodevec1 = emb

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(1, 2)) - torch.matmul(nodevec2, nodevec1.transpose(1, 2))
        adj = F.relu(torch.tanh(self.alpha * a))  # B, N, N
        return adj


class dynamic_graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, static_gc, alpha=3):
        super(dynamic_graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.gate_static = nn.Linear(dim, dim)
        self.gate_dynamic = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_gc = static_gc

    def forward(self, idx, emb):
        static_nodevec1 = self.static_gc.emb1(idx)  # N, H
        static_nodevec2 = self.static_gc.emb2(idx)
        nodevec2 = nodevec1 = emb
        nodevec1_gate = torch.sigmoid(self.gate_static(static_nodevec1) + self.gate_dynamic(nodevec1))
        nodevec1 = (1 - nodevec1_gate) * nodevec1 + nodevec1_gate * static_nodevec1
        nodevec2_gate = torch.sigmoid(self.gate_static(static_nodevec2) + self.gate_dynamic(nodevec2))
        nodevec2 = (1 - nodevec2_gate) * nodevec2 + nodevec2_gate * static_nodevec2

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(1, 2)) - torch.matmul(nodevec2, nodevec1.transpose(1, 2))
        adj = F.relu(torch.tanh(self.alpha * a))  # B, N, N
        mask = torch.zeros(emb.size(0), idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 2)
        mask.scatter_(2, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx, emb):
        static_nodevec1 = self.static_gc.emb1(idx)  # N, H
        static_nodevec2 = self.static_gc.emb2(idx)
        nodevec2 = nodevec1 = emb
        nodevec1_gate = torch.sigmoid(self.gate_static(static_nodevec1) + self.gate_dynamic(nodevec1))
        nodevec1 = (1 - nodevec1_gate) * nodevec1 + nodevec1_gate * static_nodevec1
        nodevec2_gate = torch.sigmoid(self.gate_static(static_nodevec2) + self.gate_dynamic(nodevec2))
        nodevec2 = (1 - nodevec2_gate) * nodevec2 + nodevec2_gate * static_nodevec2

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(1, 2)) - torch.matmul(nodevec2, nodevec1.transpose(1, 2))
        adj = F.relu(torch.tanh(self.alpha * a))  # B, N, N
        return adj


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        # 如果不存在静态特征，从嵌入层获取节点向量 nodevec1 和 nodevec2
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        # 如果存在静态特征，直接从静态特征中获取节点向量 nodevec1 和 nodevec2
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        # 使用线性层 lin1 和 lin2 对节点向量进行变换，并通过 tanh 激活函数缩放
        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        # 计算节点向量之间的相似度矩阵 a，并通过 tanh 和 relu 函数进行非线性变换
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        # 创建掩码矩阵 mask
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        # 使用 topk 函数选择每行中前 k 个最大的值及其索引
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        # 使用 scatter_ 函数将掩码矩阵对应位置设置为 1
        mask.scatter_(1, t1, s1.fill_(1))
        # 将邻接矩阵与掩码矩阵相乘，保留前 k 个最大值的位置，其他位置为 0
        adj = adj * mask
        return adj

    # 生成完整的邻接矩阵
    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj

    # 返回节点相似度矩阵
    def get_a(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        return torch.mm(nodevec1, nodevec2.transpose(1, 0)), torch.mm(nodevec2, nodevec1.transpose(1, 0))


def sparse_graph(adj, idx, k):
    # 创建一个大小为 [N, N] 的零矩阵 mask，其中 N 是节点数，adj 是邻接矩阵
    mask = torch.zeros(idx.size(0), idx.size(0)).to(adj.device)
    # 使用 mask.fill_(float("0")) 将掩码矩阵填充为 0
    mask.fill_(float("0"))
    # 给邻接矩阵 adj 添加一个很小的随机扰动 torch.rand_like(adj) * 0.01，以打破相同值的平衡
    # 使用 topk(k, 1) 函数选择每行中前 k 个最大的值及其索引。s1 是前 k 个最大值，t1 是对应的索引
    s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(k, 1)
    # 使用 scatter_ 函数将掩码矩阵 mask 的相应位置设置为 1
    mask.scatter_(1, t1, s1.fill_(1))
    # 将邻接矩阵 adj 与掩码矩阵 mask 相乘，保留前 k 个最大值的位置，其他位置为 0
    adj = adj * mask
    return adj


def dyna_sparse_graph(adj, idx, k):
    # 创建一个大小为 [B, N, N] 的零矩阵 mask，其中 B 是批次大小，N 是节点数
    mask = torch.zeros(adj.size(0), idx.size(0), idx.size(0)).to(adj.device)
    # 使用 mask.fill_(float("0")) 将掩码矩阵填充为 0
    mask.fill_(float("0"))
    # 给邻接矩阵 adj 添加一个很小的随机扰动 torch.rand_like(adj) * 0.01，以打破相同值的平衡
    # 使用 topk(k, 2) 函数选择每个矩阵中第 2 维度（列）中前 k 个最大的值及其索引。s1 是前 k 个最大值，t1 是对应的索引
    s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(k, 2)
    # 使用 scatter_ 函数将掩码矩阵 mask 的相应位置设置为 1
    mask.scatter_(2, t1, s1.fill_(1))
    # 将邻接矩阵 adj 与掩码矩阵 mask 相乘，保留前 k 个最大值的位置，其他位置为 0
    adj = adj * mask
    return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.parameter.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.parameter.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
