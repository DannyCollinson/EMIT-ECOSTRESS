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
import models.DannyCNN
import utils.train
import utils.eval


def train_cnn(
    project_path: str,
    base_data_path: str,
    input_type: str,
    n_dimensions: int,
    x_size: int,
    y_size: int,
    model_type: str,
    train_batch_size: int = 256,
    val_batch_size: int = 1024,
    n_epochs: int = 5,
    dropout_rate: float = 0.0,
    learning_rate: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    '''
    Runs the full training pipeline for a CNN model
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

    train_dataset = datasets.Datasets.CNNDataset(
        emit_data=emit_train,
        omit_components=omit_components,
        ecostress_data=eco_train,
        ecostress_center=None,
        ecostress_scale=None,
        additional_data=(elev_train,),
        y_size=y_size,
        x_size=x_size,
    )

    val_dataset = datasets.Datasets.CNNDataset(
        emit_data=emit_val,
        omit_components=omit_components,
        ecostress_data=eco_val,
        ecostress_center=None,
        ecostress_scale=None,
        additional_data=(elev_val,),
        y_size=y_size,
        x_size=x_size,
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

    if model_type == 'U-Net':
        model = models.DannyCNN.UNet(
            y_size=y_size,
            x_size=x_size,
            input_dim=train_dataset.input_dim,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'SS':
        model = models.DannyCNN.SemanticSegmenter(
            y_size=y_size,
            x_size=x_size,
            input_dim=train_dataset.input_dim,
            dropout_rate=dropout_rate,
        )

    model = model.to(torch.device(device))

    optimizer = optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=0, fused=True
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.2, patience=2
    )

    loss_fn = nn.MSELoss(reduction='sum')

    print(
        f'x_size={x_size}, y_size={y_size}, '
        f'n_dimensions={n_dimensions}\n{model}'
    )
    
    
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
                np.array((y_size, x_size, n_dimensions))[:, np.newaxis],
                eval_stats[:, np.newaxis],
            ],
            axis=0,
        )
        
        stats_columns = utils.eval.initialize_eval_results().columns.to_list()
        stats_columns.extend(['y_size', 'x_size'])
        stats_columns.remove('radius')
        stats = pd.DataFrame({column: stat for column, stat in zip(stats_columns, eval_stats)})
        stats['y_size'] = stats['y_size'].astype(int)
        stats['x_size'] = stats['x_size'].astype(int)
        stats['n_dimensions'] = stats['n_dimensions'].astype(int)
    else:
        stats = None
    
    # print(stats)
    
    utils.eval.plot_loss_cnn(
        train_loss,
        val_loss,
        x_size,
        y_size,
        n_dimensions,
        model_type,
        input_type,
    )
    
    utils.eval.plot_loss_on_map_cnn(
        train_loss_array, val_loss_array, x_size, y_size, n_dimensions
    )

    return train_loss, val_loss, stats, train_loss_array, val_loss_array