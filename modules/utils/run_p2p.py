import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.pickling import join_path, pickle_save, pickle_load
import datasets.Datasets
import models.PatchToPixel
import utils.train
import utils.eval


def train_patch_to_pixel(
    project_path: str,
    base_data_path: str,
    input_type: str,
    n_dimensions: int,
    radius: int,
    model_type: str,
    train_batch_size: int = 256,
    val_batch_size: int = 1024,
    n_epochs: int = 5,
    dropout_rate: float = 0.0,
    learning_rate: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    '''
    Runs the full training pipeline for a patch-to-pixel model
    '''

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running using {device}\n')
    
    
    # load data
    
    if input_type == 'raw':
        ref_str = os.path.join('Raw', 'reflectance_***.pkl')
        omit_components = 244 - n_dimensions
    elif input_type == 'PCA':
        ref_str = os.path.join('PCA', 'reflectance_***_pca244.pkl')
        omit_components = 244 - n_dimensions
    elif input_type == 'AE':
        ref_str = os.path.join('AE', f'dim_{n_dimensions}_***.pkl')
        omit_components = 0
        
    emit_train = pickle_load(
        project_path,
        os.path.join(base_data_path, ref_str.replace('***', 'train'))
    )
    emit_val = pickle_load(
        project_path,
        os.path.join(base_data_path, ref_str.replace('***', 'val'))
    )

    elev_train = pickle_load(
        project_path,
        os.path.join(base_data_path, 'Non-Ref', 'elevation_train.pkl')
    )
    elev_val = pickle_load(
        project_path,
        os.path.join(base_data_path, 'Non-Ref', 'elevation_val.pkl')
    )

    elev_train = (
        (
            elev_train - np.mean(np.concatenate([elev_train, elev_val], axis=1))
        ) / 
        np.std(np.concatenate([elev_train, elev_val], axis=1))
    )
    elev_val = (
        (
            elev_val - np.mean(np.concatenate([elev_train, elev_val], axis=1))
        ) / 
        np.std(np.concatenate([elev_train, elev_val], axis=1))
    )
    eco_train = pickle_load(
        project_path,
        os.path.join(base_data_path, 'Non-Ref', 'temp_train.pkl')
    )
    eco_val = pickle_load(
        project_path,
        os.path.join(base_data_path, 'Non-Ref', 'temp_val.pkl')
    )
    
    
    # create datasets and dataloaders

    train_dataset = datasets.Datasets.PatchToPixelDataset(
        emit_data=emit_train,
        omit_components=omit_components,
        ecostress_data=eco_train,
        ecostress_center=None,
        ecostress_scale=None,
        additional_data=(elev_train,),
        radius=radius,
        boundary_width=radius,
    )

    val_dataset = datasets.Datasets.PatchToPixelDataset(
        emit_data=emit_val,
        omit_components=omit_components,
        ecostress_data=eco_val,
        ecostress_center=None,
        ecostress_scale=None,
        additional_data=(elev_val,),
        radius=radius,
        boundary_width=radius,
    )

    if train_batch_size is not None:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            drop_last=False,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            drop_last=False,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=None, shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=None, shuffle=False,
        )
    
    
    # define model and other training configurations

    if model_type == 'linear':
        model = models.PatchToPixel.LinearModel(
            input_dim=train_dataset.input_dim,
            radius=radius,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'mini':
        model = models.PatchToPixel.MiniDenseNN(
            input_dim=train_dataset.input_dim,
            radius=radius,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'small':
        model = models.PatchToPixel.SmallDenseNN(
            input_dim=train_dataset.input_dim,
            radius=radius,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'large':
        model = models.PatchToPixel.LargeDenseNN(
            input_dim=train_dataset.input_dim,
            radius=radius,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'attention':
        model = models.PatchToPixel.SelfAttentionModel(
            input_dim=train_dataset.input_dim,
            radius=radius,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'transformer':
        raise NotImplementedError(
            'Transformer training has not yet been implemented in this notebook'
        )
    
    model = model.to(torch.device(device))

    optimizer = optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=0, fused=True
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.2, patience=2
    )

    loss_fn = nn.MSELoss(reduction='sum')

    print(f'radius={radius}, n_dimensions={n_dimensions}\n{model}')
    
    
    # run training!
    
    train_loss, val_loss, eval_stats, train_loss_array, val_loss_array = (
        utils.train.train(
            model,
            optimizer,
            scheduler,
            loss_fn,
            train_loader,
            val_loader,
            n_epochs=n_epochs,
            loss_interval=1,
            device=device,
        )
    )
    
    print('\nRunning performance evaluations')
    
    eval_train_loader = DataLoader(
        dataset=train_dataset, batch_size=2048, shuffle=False,
    )
    train_loss_array = utils.eval.train_loss_map(
        model, eval_train_loader, device
    )
    
    if eval_stats is not None:
        eval_stats = np.concatenate(
            [
                np.array((radius, n_dimensions))[:, np.newaxis],
                eval_stats[:, np.newaxis],
            ],
            axis=0,
        )
        
        stats_columns = utils.eval.initialize_eval_results().columns.to_list()
        stats = pd.DataFrame({column: stat for column, stat in zip(stats_columns, eval_stats)})
        stats['radius'] = stats['radius'].astype(int)
        stats['n_dimensions'] = stats['n_dimensions'].astype(int)
    else:
        stats = None
    
    # print(stats)
    
    utils.eval.plot_loss_patch_to_pixel(
        train_loss, val_loss, radius, n_dimensions, model_type, input_type
    )
    
    utils.eval.plot_loss_on_map_patch_to_pixel(
        train_loss_array, val_loss_array, radius, n_dimensions
    )

    return train_loss, val_loss, stats, train_loss_array, val_loss_array