#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import pandas as pd
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)

    return scaffold



def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:

        scaffold = generate_scaffold(smiles, include_chirality=True)

        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)


def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):


    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset


def random_split(dataset, task_idx=None, null_value=0,
                 frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
                 smiles_list=None, return_idx=False):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:

        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value
        idx_array = np.where(non_null)[0]
        dataset = dataset[torch.tensor(idx_array)]
    else:
        pass

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols) + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if return_idx:
        return train_idx, valid_idx, test_idx

    else:
        if not smiles_list:
            return train_dataset, valid_dataset, test_dataset
        else:
            train_smiles = [smiles_list[i] for i in train_idx]
            valid_smiles = [smiles_list[i] for i in valid_idx]
            test_smiles = [smiles_list[i] for i in test_idx]
            return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                                valid_smiles,
                                                                test_smiles)


def data_split(args, dataset):
    if args.split == "scaffold":
        smiles_list = pd.read_csv(args.dataset_dir + "/" + args.dataset + '/processed/smiles.csv', header=None)[
            0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,
                                                                    frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
                                                                  frac_test=0.1, seed=args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(args.dataset_dir + "/" + args.dataset + '/processed/smiles.csv', header=None)[
            0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
                                                                           frac_train=0.8, frac_valid=0.1,
                                                                           frac_test=0.1, seed=args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    return train_dataset, valid_dataset, test_dataset

