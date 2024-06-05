import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
import os
from sklearn.decomposition import FastICA


def get_model(p, num_nodes, num_rels):
    if p.score_func.lower() == 'conve':
        model = ConvE(num_nodes, num_rels, params=p)
    else:
        raise NotImplementedError
    model.to(p.device)
    return model


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


# 线性变换模型模型
class LTEModel(nn.Module):
    def __init__(self, num_ents, num_rels, params=None):
        super(LTEModel, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params
        self.init_embed = get_param((num_ents, self.p.init_dim))
        self.init_rel = get_param((num_rels * 2, self.p.init_dim))
        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.p.x_ops
        self.diff_ht = False

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


    def exop(self, x, x_ops=None, diff_ht=False):
        x_head = x_tail = x

        if len(x_ops) > 0:
            for x_op in x_ops.split("."):

                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head) + x_head
                    x_tail = self.t_ops_dict[x_op](x_tail) + x_tail
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head) + x_head

        return x_head, x_tail


class ConvE(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)

        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.p.gamma + 2) / self.p.init_dim]),
            requires_grad=False
        )
        self.pi = 3.14159262358979323846


    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1,
                                 self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed],
                              dim=1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):
        x = self.init_embed
        r = self.init_rel

        x_h, x_t = self.exop(x, self.x_ops)

        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        phase_head, mod_head = torch.chunk(sub_emb, 2, dim=-1)
        phase_relation, mod_relation = torch.chunk(rel_emb, 2, dim=-1)
        phase_tail, mod_tail = torch.chunk(x, 2, dim=-1)
        
        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)
        mod_relation = torch.abs(mod_relation)
        all_ent = torch.cat([phase_tail, mod_tail],dim=-1)
        a = torch.cat([phase_head , phase_relation],dim=-1)
        b = torch.cat([phase_relation , mod_relation],dim=-1)
        stk_inp = self.concat(a, b)

        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score
