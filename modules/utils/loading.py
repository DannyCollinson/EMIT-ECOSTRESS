import os
import sys
import importlib
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

base_data_path = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/data'

def join_path(relative_path: str) -> str:
    return os.path.join(base_data_path, relative_path)

def pickle_load(relative_path: str):  # -> pickled_file_contents
        return pickle.load(open(join_path(relative_path), 'rb'))

def merge_emit_pickle(preprocessing, splits, dim):
    full_dataset = []

    if preprocessing == "AE":
        full_dataset = torch.tensor(pickle_load(f'latent_embeddings_dim_{dim}.pkl'), dtype=torch.float32)
        train_data_subset, val_data_subset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
        train_data = full_dataset[train_data_subset.indices]
        val_data = full_dataset[val_data_subset.indices]
        full_dataset = [train_data, val_data]
    elif preprocessing == "PCA":
        for split in splits:
            full_dataset.append(
                    pickle_load(
                        f'reflectance_{split}_pca244.pkl'
                    )
                )
    else:
        for split_num in range(splits):
            num = "0"*(2-len(str(split_num))) + str(split_num)

            full_dataset.append(
                pickle_load(
                    f'reflectance_{num}_list.pkl'
                )
            )
    full_dataset = np.concatenate(full_dataset, axis=0)

    return full_dataset

def load_emit(train_splits, val_splits, preprocessing, dim):
    if preprocessing == "AE":
            temp = merge_emit_pickle(preprocessing, train_splits, dim)
            emit_train = temp[0]
            emit_val = temp[1]
    else:
        emit_train_list = merge_emit_pickle(preprocessing, train_splits, dim)
        emit_train = np.concatenate(emit_train_list, axis=0)

        emit_val_list = merge_emit_pickle(preprocessing, val_splits, dim)
        emit_val = np.concatenate(emit_val_list, axis=0)
    
    return emit_train, emit_val


def load_ecostress(base_data_path, train_splits, val_splits):
    eco_train_list = []
    eco_val_list = []
    
    for split in train_splits:
        eco_train_list.append(
            pickle_load(
                os.path.join(base_data_path, f'LSTE_{split}.pkl')
            )
        )
    eco_train = np.concatenate(eco_train_list, axis=0)

    for split in val_splits:
        eco_val_list.append(
            pickle_load(
                os.path.join(base_data_path, f'LSTE_{split}.pkl')
            )
        )
    eco_val = np.concatenate(eco_val_list, axis=0)
    return eco_train, eco_val

def load_elevation(base_data_path, train_splits, val_splits, need_elevation):
    if need_elevation:
        elev_train_list = []
        
        for split in train_splits:
            elev_train_list.append(
                pickle_load(
                    os.path.join(base_data_path, f'elevation_{split}.pkl')
                )
            )
        elev_train = np.concatenate(elev_train_list, axis=0)

        elev_val_list = []
        for split in val_splits:
            elev_val_list.append(
                pickle_load(
                    os.path.join(base_data_path, f'Elevation\\elevation_{split}.pkl')
                )
            )
        elev_val = np.concatenate(elev_val_list, axis=0)
        return elev_train, elev_val
    else:
        return None, None

# FOR TESTING

train_splits = ['00', '02', '04']
val_splits = ['01', '05', '09']

def run_script(proj_path, train_splits, val_splits, preprocessing, dim, need_elevation):
    em_tr, em_val = load_emit(train_splits, val_splits, preprocessing, dim)
    print("done emit")
    eco_tr, eco_val = load_ecostress(proj_path, train_splits, val_splits)
    print("done eco")
    ele_tr, ele_val = load_elevation(proj_path, train_splits, val_splits, need_elevation)
    print("done ele")

    return (em_tr, em_val, eco_tr, eco_val, ele_tr, ele_val)

em_tr_test, em_val_test, eco_tr_test, eco_val_test, ele_tr_test, ele_val_test = run_script(base_data_path, train_splits, val_splits, "AE", 16, False)
print("done")