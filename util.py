#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
from typing import Any, List
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans


class Scf_index:
    def __init__(self, dataset, args):
        self.device = args.device

        self.max_scf_idx = None
        self.scfIdx_to_label = None
        self.num_scf = None

        self.get_scf_idx(dataset)

    def get_scf_idx(self, dataset):

        scf = defaultdict(int)
        max_scf_idx = 0
        for data in dataset:
            idx = data.scf_idx.item()
            scf[idx] += 1
            if max_scf_idx < idx:
                max_scf_idx = idx
        self.max_scf_idx = max_scf_idx
        scf = sorted(scf.items(), key=lambda x: x[1], reverse=True)

        self.scfIdx_to_label = torch.ones(max_scf_idx + 1).to(torch.long).to(torch.long) * -1
        self.scfIdx_to_label = self.scfIdx_to_label.to(self.device)

        for i, k in enumerate(scf):
            self.scfIdx_to_label[k[0]] = i

        self.num_scf = len(scf)


def load_models(args, model):
    if not args.ckpt_all == "":

        load = torch.load(args.ckpt_all)
        mis_keys, unexp_keys = model.load_state_dict(load, strict=False)
        print('missing_keys:', mis_keys)
        print('unexpected_keys:', unexp_keys)

    elif not args.input_model_file == "":
        model.from_pretrained(args.input_model_file, map=args.device)


# utils for eval
def cal_roc(y_true, y_scores):
    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    return sum(roc_list) / (len(roc_list) + 1e-10) * 100


def cal_rmse(y_true, y_scores):
    '''
        compute RMSE score averaged across tasks
    '''
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(((y_true[is_labeled, i] - y_scores[is_labeled, i]) ** 2).mean()))

    return sum(rmse_list) / len(rmse_list)


def init_centroid(model, sca_zs_init, gro_zs_init, num_experts):

    sca_zs = sca_zs_init.detach().cpu().numpy()
    num_data = sca_zs.shape[0]
    if num_data > 35000:
        mask_idx = list(range(num_data))
        random.shuffle(mask_idx)
        sca_zs = sca_zs[mask_idx[:35000]]
    kmeans_sca = KMeans(n_clusters=num_experts, random_state=0).fit(sca_zs)
    centroids_sca = kmeans_sca.cluster_centers_
    model.sca_cluster.data = torch.tensor(centroids_sca).to(model.sca_cluster.device)

    gro_zs = gro_zs_init.detach().cpu().numpy()
    num_data = gro_zs.shape[0]
    if num_data > 35000:
        mask_idx = list(range(num_data))
        random.shuffle(mask_idx)
        gro_zs = gro_zs[mask_idx[:35000]]
    kmeans_gro = KMeans(n_clusters=num_experts, random_state=0).fit(gro_zs)
    centroids_gro = kmeans_gro.cluster_centers_
    model.gro_cluster.data = torch.tensor(centroids_gro).to(model.gro_cluster.device)


def init_centroid1(model, zs_init, num_experts):

    zs = zs_init.detach().cpu().numpy()
    num_data = zs.shape[0]
    if num_data > 35000:
        mask_idx = list(range(num_data))
        random.shuffle(mask_idx)
        zs = zs[mask_idx[:35000]]
    kmeans = KMeans(n_clusters=num_experts, random_state=0).fit(zs)
    centroids = kmeans.cluster_centers_
    model.cluster.data = torch.tensor(centroids).to(model.cluster.device)

def get_z(model, loader, device):
    model.train()

    sca_z_s = []
    gro_z_s = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            _, _, _, _, sca_z, gro_z, _, _ = model(batch)
        sca_z_s.append(sca_z)
        gro_z_s.append(gro_z)

    sca_z_s = torch.cat(sca_z_s, dim=0)
    gro_z_s = torch.cat(gro_z_s, dim=0)
    return sca_z_s, gro_z_s


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
