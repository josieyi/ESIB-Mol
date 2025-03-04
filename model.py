#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_min
from torch.nn import ParameterList, Parameter, ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from base_gnn import GNN_node, GNN


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

def MI_Est(discriminator, gnn_emb, sca_emb, gro_emb):

    batch_size = gnn_emb.shape[0]
    shuffle_embeddings = gnn_emb[torch.randperm(batch_size)]
    sca_joint = discriminator(gnn_emb, sca_emb)
    sca_margin = discriminator(shuffle_embeddings, sca_emb)
    valid_sca = torch.isfinite(torch.exp(sca_margin))
    mi_est_sca = torch.mean(sca_joint[valid_sca]) - torch.clamp(torch.log(torch.mean(torch.exp(sca_margin[valid_sca]))),
                                                                -100000, 100000)

    gro_joint = discriminator(gnn_emb, gro_emb)
    gro_margin = discriminator(shuffle_embeddings, gro_emb)
    valid_gro = torch.isfinite(torch.exp(gro_margin))
    mi_est_gro = torch.mean(gro_joint[valid_gro]) - torch.clamp(torch.log(torch.mean(torch.exp(gro_margin[valid_gro]))),
                                                                -100000, 100000)
    mi_est = 0.5 * (mi_est_sca + mi_est_gro)

    return mi_est

class gate(torch.nn.Module):
    def __init__(self, emb_dim, gate_dim=300):
        super(gate, self).__init__()
        self.linear1 = nn.Linear(emb_dim, gate_dim)
        self.batchnorm = nn.BatchNorm1d(gate_dim)
        self.linear2 = nn.Linear(gate_dim, gate_dim)

    def forward(self, x):
        x = self.linear1(x)
        try:
            x = self.batchnorm(x)
        except:
            pass
        x = F.relu(x)
        gate_emb = self.linear2(x)
        return gate_emb


class GNN_topexpert(torch.nn.Module):

    def __init__(self, args, criterion):
        super(GNN_topexpert, self).__init__()

        self.args = args
        self.num_layer = args.num_layer
        self.drop_ratio = args.dropout_ratio
        self.JK = args.JK
        self.emb_dim = args.emb_dim
        self.num_tasks = args.num_tasks
        self.num_experts = args.num_experts
        self.graph_pooling = args.graph_pooling
        self.gnn_type = args.gnn_type
        self.tau = args.tau
        self.all_expert_num = self.num_experts * self.num_tasks
        self.T = args.T

        self.sca_gate = gate(args.emb_dim, args.gate_dim)
        self.gro_gate = gate(args.emb_dim, args.gate_dim)

        self.sca_cluster = nn.Parameter(torch.Tensor(args.num_experts, args.gate_dim))
        torch.nn.init.xavier_normal_(self.sca_cluster.data)
        self.gro_cluster = nn.Parameter(torch.Tensor(args.num_experts, args.gate_dim))
        torch.nn.init.xavier_normal_(self.gro_cluster.data)

        self.criterion = criterion

        self.sca_experts_w = nn.Parameter(torch.empty(self.emb_dim, self.all_expert_num))
        self.sca_experts_b = nn.Parameter(torch.empty(self.all_expert_num))
        self.gro_experts_w = nn.Parameter(torch.empty(self.emb_dim, self.all_expert_num))
        self.gro_experts_b = nn.Parameter(torch.empty(self.all_expert_num))
        self.reset_experts()

        self.gate_pool = global_add_pool

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if self.args.class_type == "cls":
            self.gnn = GNN(self.num_layer, self.emb_dim, self.JK, self.drop_ratio, gnn_type=self.gnn_type)
        else:
            self.gnn = GNN_node(self.num_layer, self.emb_dim, JK=self.JK, drop_ratio=self.drop_ratio,
                                residual=False, gnn_type=self.gnn_type)

        self.sca_mlp = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.gro_mlp = torch.nn.Linear(self.emb_dim, self.num_tasks)

        self.sca_pred_list = []
        self.gro_pred_list = []
        for _ in range(self.num_experts):
            self.sca_pred_list.append(nn.Linear(self.emb_dim, self.num_tasks).to(args.device))
            self.gro_pred_list.append(nn.Linear(self.emb_dim, self.num_tasks).to(args.device))

        # Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling[:-1] == "set2set":
            set2set_iter = int(self.graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def reset_experts(self):
        bound = 1 / math.sqrt(self.emb_dim)
        torch.nn.init.kaiming_uniform_(self.sca_experts_w, a=math.sqrt(5))
        torch.nn.init.uniform_(self.sca_experts_b, -bound, bound)

        torch.nn.init.kaiming_uniform_(self.gro_experts_w, a=math.sqrt(5))
        torch.nn.init.uniform_(self.gro_experts_b, -bound, bound)

    def from_pretrained(self, model_file, map):
        self.gnn.load_state_dict(torch.load(model_file, map_location=map), strict=False)


    def forward(self, data):

        batch, node_batch = data.batch, data.node_batch

        node_index, _ = scatter_min(node_batch, batch)
        if self.args.class_type == "cls":
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            node_rep = self.gnn(x, edge_index, edge_attr)
        else:
            node_rep = self.gnn(data)
        gnn_out = self.pool(node_rep, batch)

        scaffold_indices = torch.nonzero(torch.eq(node_batch.unsqueeze(1), node_index), as_tuple=False)
        sca_gro_mask = torch.zeros(len(node_batch)).bool()
        sca_gro_mask[scaffold_indices[:, 0]] = True
        sca_node_rep, sca_batch = node_rep[sca_gro_mask], batch[sca_gro_mask]
        gro_node_rep, gro_batch = node_rep[~sca_gro_mask], batch[~sca_gro_mask]
        sca_rep = self.pool(sca_node_rep, sca_batch)
        try:
            gro_rep = self.pool(gro_node_rep, gro_batch)
        except:
            gro_rep = torch.empty((0, self.emb_dim), device=node_rep.device)

        sca_gate_input = self.gate_pool(sca_node_rep, sca_batch)
        try:
            gro_gate_input = self.gate_pool(gro_node_rep, gro_batch)
        except:
            gro_gate_input = torch.empty((0, self.emb_dim), device=node_rep.device)

        nosca_index = torch.nonzero(data.scaffold_flag == 0).squeeze(1)
        if len(sca_rep) == len(gro_rep) and len(nosca_index) == 0:
            pass
        else:
            s_g_num = len(sca_rep) - len(gro_rep)
            if len(nosca_index) == 0 or s_g_num != 0:
                for idx in range(s_g_num, 0, -1):
                    gro_rep = torch.cat((gro_rep, sca_rep[-idx].unsqueeze(0)))
                    gro_gate_input = torch.cat((gro_gate_input, sca_gate_input[-idx].unsqueeze(0)))
            else:
                for nosca_i in nosca_index:
                    if len(sca_rep) - nosca_i <= s_g_num:
                        gro_rep = torch.cat((gro_rep, sca_rep[nosca_i].unsqueeze(0)))
                        gro_gate_input = torch.cat((gro_gate_input, sca_gate_input[nosca_i].unsqueeze(0)))
                    else:
                        gro_rep[nosca_i] = sca_rep[nosca_i]
                        gro_gate_input[nosca_i] = sca_gate_input[nosca_i]

            gro_rep[gro_rep.sum(1) == 0] = sca_rep[gro_rep.sum(1) == 0]

            # multi-head mlps
            sca_rep_emb = torch.unsqueeze(sca_rep, -1)
            sca_gnn_emb = sca_rep_emb.repeat(1, 1, self.all_expert_num)
            gro_rep_emb = torch.unsqueeze(gro_rep, -1)
            gro_gnn_emb = gro_rep_emb.repeat(1, 1, self.all_expert_num)

            sca_e = sca_rep
            gro_e = gro_rep

            if self.args.class_type == "cls":
                sca_emb = sca_gnn_emb * self.sca_experts_w
                gro_emb = gro_gnn_emb * self.gro_experts_w
                if self.training:
                    sca_logit = torch.sum(sca_emb, dim=1) + self.sca_experts_b
                    gro_logit = torch.sum(gro_emb, dim=1) + self.gro_experts_b
                else:
                    sca_logit = torch.sum(sca_emb, dim=1)
                    gro_logit = torch.sum(gro_emb, dim=1)
                sca_logit = sca_logit.view(-1, self.num_tasks, self.num_experts)
                gro_logit = gro_logit.view(-1, self.num_tasks, self.num_experts)
                clf_logit = torch.cat((sca_logit, gro_logit), dim=2)
            else:
                sca_logits = []
                gro_logits = []
                for e_i in range(self.num_experts):
                    sca_logits.append(self.sca_pred_list[e_i](sca_e))
                    gro_logits.append(self.gro_pred_list[e_i](gro_e))
                sca_logit = torch.hstack(sca_logits)
                gro_logit = torch.hstack(gro_logits)
                clf_logit = torch.cat((sca_logit, gro_logit), dim=1)

            sca_z = self.sca_gate(sca_gate_input)
            gro_z = self.gro_gate(gro_gate_input)

            sca_q = self.sca_get_q(sca_z)
            gro_q = self.gro_get_q(gro_z)

            return clf_logit, sca_e, gro_e, gnn_out, sca_z, gro_z, sca_q, gro_q

    def assign_head(self, q):

        q_idx = torch.argmax(q, dim=-1)  # N x 1
        if self.training:
            g = F.gumbel_softmax((q + 1e-10).log(), tau=10, hard=False, dim=-1)
            g = torch.unsqueeze(g, 1)
            g = g.repeat(1, self.num_tasks, 1)  # N x tasks x heads
            return g, q_idx
        else:
            q = torch.unsqueeze(q, 1)
            q = q.repeat(1, self.num_tasks, 1)  # N x tasks x heads
            return q, q_idx

    def sca_get_q(self, z):

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.sca_cluster, 2), 2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def gro_get_q(self, z):

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.gro_cluster, 2), 2))
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def clf_loss(self, discriminator, clf_outs, labels, sca_emb, gro_emb, gnn_emb, assign):

        mi_loss = MI_Est(discriminator, gnn_emb.detach(), sca_emb.detach(), gro_emb.detach())

        if self.args.class_type == "cls":
            is_valid = labels ** 2 > 0
            is_valid_tensor = torch.unsqueeze(is_valid, -1)
            is_valid_tensor = is_valid_tensor.repeat(1, 1, clf_outs.shape[-1])
            labels_clf = torch.unsqueeze(labels, -1).repeat(1, 1, clf_outs.shape[-1])
            loss_tensor = self.criterion(clf_outs, (labels_clf + 1) / 2)
            loss_tensor_valid = torch.where(is_valid_tensor, loss_tensor,
                                            torch.zeros(loss_tensor.shape).to(loss_tensor.device).to(loss_tensor.dtype))
            loss_mat = torch.sum(assign * loss_tensor_valid, dim=0)
            num_valid_mat = torch.sum(assign * is_valid_tensor.long(), dim=0)
        else:
            is_valid = labels ** 2 > 0
            is_valid_tensor = is_valid.repeat(1, clf_outs.shape[-1])
            labels_clf = labels.repeat(1, clf_outs.shape[-1])
            loss_tensor = self.criterion(clf_outs, labels_clf)
            loss_tensor_valid = torch.where(is_valid_tensor, loss_tensor,
                                            torch.zeros(loss_tensor.shape).to(loss_tensor.device).to(loss_tensor.dtype))
            loss_mat = torch.sum(assign.squeeze() * loss_tensor_valid, dim=0)
            num_valid_mat = torch.sum(assign.squeeze() * is_valid_tensor.long(), dim=0)


        return mi_loss, loss_mat, (num_valid_mat + 1e-10)

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p

