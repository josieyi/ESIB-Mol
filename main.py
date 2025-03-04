#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
from tqdm import tqdm
from util import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from splitters import data_split
from configs import load_args
from loader import MoleculeDataset
from model import GNN_topexpert, MI_Est
from base_gnn import Discriminator
import warnings
warnings.filterwarnings('ignore')

data_path = './'

def train(args, model, discriminator, loader, optimizer_dis, optimizer):
    model.train()

    for batch in loader:
        model.T = max(torch.tensor(args.min_temp), model.T * args.temp_alpha)
        batch = batch.to(args.device)
        num_graph = batch.id.shape[0]

        if args.class_type == "cls":
            labels = batch.y.view(num_graph, -1).to(torch.float64)
        else:
            labels = batch.y.view(num_graph, -1)

        _, _, _, _, _, _, sca_q, gro_q = model(batch)
        sca_q = sca_q.data
        gro_q = gro_q.data

        sca_p = model.target_distribution(sca_q)
        gro_p = model.target_distribution(gro_q)

        clf_logit, sca_rep, gro_rep, gnn_emb, sca_z, gro_z, sca_q, gro_q = model(batch)
        sca_g, sca_q_idx = model.assign_head(sca_q)  # g--> N x tasks x head
        gro_g, gro_q_idx = model.assign_head(gro_q)

        cnt = 0
        inner_best_loss = None
        for j in range(0, args.inner_loop):
            optimizer_dis.zero_grad()
            inner_mi_loss = -MI_Est(discriminator, gnn_emb.detach(), sca_rep.detach(), gro_rep.detach())
            inner_mi_loss.backward()

            if inner_best_loss is None:
                inner_best_loss = inner_mi_loss
                cnt = 0
            elif inner_mi_loss < inner_best_loss:
                inner_best_loss = inner_mi_loss
                cnt = 0
            else:
                cnt += 1
            optimizer_dis.step()
            if cnt >= 5:
                break

        assign = F.normalize(torch.cat((sca_g, gro_g), dim=-1), p=1, dim=-1)
        mi_loss, clf_loss_mat, num_valid_mat = model.clf_loss(discriminator, clf_logit, labels, sca_rep, gro_rep,
                                                              gnn_emb, assign)
        if args.class_type == "cls":
            classification_loss = torch.sum(clf_loss_mat / num_valid_mat) / args.num_tasks
        else:
            classification_loss = torch.sum(clf_loss_mat / num_valid_mat) / args.num_tasks

        cluster_loss = F.kl_div(sca_q.log(), sca_p, reduction='sum') + F.kl_div(gro_q.log(), gro_p, reduction='sum')
        loss_total = args.alpha * mi_loss + args.beta * (cluster_loss / (num_graph * 2)) + classification_loss

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return loss_total.detach().item(), inner_best_loss.detach().item()


def eval(args, model, loader):
    model.eval()

    y_true, y_scores = [], []
    for batch in loader:
        batch = batch.to(args.device)
        with torch.no_grad():
            clf_logit, _, _, _, _, _, sca_q, gro_q = model(batch)
            sca_q, sca_q_idx = model.assign_head(sca_q)  # N x tasks x head
            gro_q, gro_q_idx = model.assign_head(gro_q)
            assign = F.normalize(torch.cat((sca_q, gro_q), dim=-1), p=1, dim=-1)
            if args.class_type == "cls":
                scores = torch.sum(torch.sigmoid(clf_logit) * assign, dim=-1)
            else:
                scores = torch.sum(clf_logit * assign.squeeze(1), dim=-1)

        y_true.append(batch.y.view(batch.id.shape[0], -1))
        y_scores.append(scores.view(batch.id.shape[0], -1))

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    if args.class_type == "cls":
        avg_result = cal_roc(y_true, y_scores)
    else:
        avg_result = cal_rmse(y_true, y_scores)

    return avg_result

def main(args):
    set_seed(args.seed)
    model_path = "{}/Experiments/{}/{}_{}_{}/".format(data_path, args.dataset,
                                                      args.gnn_type, args.num_experts, args.seed)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_file = "{}/{}_{}_{}_{}_model.pt".format(model_path, args.lr, args.alpha, args.beta, args.decay)

    # dataset split & data loader
    dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
    if args.class_type == "cls":
        train_dataset, valid_dataset, test_dataset = data_split(args, dataset)
    else:
        train_dataset = dataset[dataset.id_split['train'].to(args.device)]
        valid_dataset = dataset[dataset.id_split['valid'].to(args.device)]
        test_dataset = dataset[dataset.id_split['test'].to(args.device)]

    # criterion
    if args.class_type == "cls":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = nn.MSELoss(reduction="none")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_iter = math.ceil(len(train_dataset) / args.batch_size)
    args.temp_alpha = np.exp(np.log(args.min_temp / 10 + 1e-10) / (args.epochs * num_iter))


    model = GNN_topexpert(args, criterion)
    load_models(args, model)
    model = model.to(args.device)
    discriminator = Discriminator(args, args.emb_dim)
    discriminator.to(args.device)
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)

    # init centroid using randomly initialized gnn
    sca_zs_init, gro_zs_init = get_z(model, train_loader, args.device)
    init_centroid(model, sca_zs_init, gro_zs_init, args.num_experts)

    valid_curve = []
    test_curve = []

    best_valid = None
    pbar = tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        train_loss, mi_loss = train(args, model, discriminator, train_loader, optimizer_dis, optimizer)
        val_acc = eval(args, model, val_loader)
        te_acc = eval(args, model, test_loader)

        valid_curve.append(val_acc)
        test_curve.append(te_acc)
        if best_valid is None:
            best_valid = val_acc
            torch.save(model, model_file)
        elif args.class_type == "cls" and val_acc >= best_valid:
            best_valid = val_acc
            torch.save(model, model_file)
        elif args.class_type == "reg" and val_acc <= best_valid:
            best_valid = val_acc
            torch.save(model, model_file)

        if args.class_type == "cls":
            pbar.set_description(
                f'{epoch}epoch, inner loss:{mi_loss:.4f}, train loss:{train_loss:.4f},'
                  f' val acc:{val_acc:.1f}, test acc:{te_acc:.1f} ')
        else:
            pbar.set_description(
                f'{epoch}epoch, inner loss:{mi_loss:.4f}, train loss:{train_loss:.4f},'
                  f' val acc:{val_acc:.3f}, test acc:{te_acc:.3f} ')

    if args.class_type == "cls":
        best_test = max(test_curve)
        best_val_epoch = np.argmax(np.array(valid_curve))
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_curve[best_val_epoch]))
        print('Best Test score: {}'.format(best_test))
    else:
        best_test = min(test_curve)
        best_val_epoch = np.argmin(np.array(valid_curve))
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_curve[best_val_epoch]))
        print('Best Test score: {}'.format(best_test))



if __name__ == "__main__":
    args = load_args()
    main(args)
