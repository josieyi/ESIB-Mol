#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse

def load_args():
    parser = argparse.ArgumentParser()

    # seed & device
    parser.add_argument('--device_no', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
    parser.add_argument('--device', type=str)

    # dataset
    parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
    parser.add_argument('--dataset', type=str, default='bace', help='root directory of dataset')
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=2)

    # model
    parser.add_argument('-i', '--input_model_file', type=str, default='', help='filename to read the model')
    parser.add_argument('-c', '--ckpt_all', type=str, default='', help='filename to read the model ')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max, concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--class_type', type=str, default="cls", help="cls, reg")

    # train
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')

    ## loss balance
    parser.add_argument('--alpha', type=float, default=0.5, help="balance parameter for MI")
    parser.add_argument('--beta', type=float, default=0.1, help="balance parameter for clustering")
    parser.add_argument('--temp_alpha', type=float)

    ## clustering
    parser.add_argument('--min_temp', type=float, default=1, help=" temperature for gumble softmax, annealing")
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--gate_dim', type=int, default=300, help="gate embedding space dimension")

    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--T', type=float, default=10)
    parser.add_argument('--inner_loop', type=int, default=30)


    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")

    # Bunch of classification tasks
    if args.class_type == "cls":
        if args.dataset == "tox21":
            args.num_tasks = 12
            args.num_classes = 2
        elif args.dataset == "hiv":
            args.num_tasks = 1
            args.num_classes = 2
        elif args.dataset == "pcba":
            args.num_tasks = 128
            args.num_classes = 2
        elif args.dataset == "muv":
            args.num_tasks = 17
            args.num_classes = 2
        elif args.dataset == "bace":
            args.num_tasks = 1
            args.num_classes = 2
        elif args.dataset == "bbbp":
            args.num_tasks = 1
            args.num_classes = 2
        elif args.dataset == "toxcast":
            args.num_tasks = 617
            args.num_classes = 2
        elif args.dataset == "sider":
            args.num_tasks = 27
            args.num_classes = 2
        elif args.dataset == "clintox":
            args.num_tasks = 2
            args.num_classes = 2
        else:
            raise ValueError("Invalid dataset name.")
    else:
        args.num_tasks = 1
        args.num_classes = 1

    return args

