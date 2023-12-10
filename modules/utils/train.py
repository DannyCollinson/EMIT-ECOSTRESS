import time
from typing import Union, Any

import numpy as np
import pandas as pd

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

import utils.eval
import datasets.Datasets


def train(
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader: DataLoader,
    val_loader: Union[DataLoader, None] = None,
    n_epochs: int = 10,
    loss_interval: Union[int, None] = 5,
    preexisting_losses: Union[list[np.ndarray, np.ndarray], None] = None,
    device: str = 'cpu',
) -> tuple[np.ndarray[np.float64, Any], np.ndarray[np.float64, Any]]:
    begin = time.time()
    t = begin
    start_epoch = 0
    
    train_std = train_loader.dataset.ecostress_scale
    if val_loader is not None:
        val_std = val_loader.dataset.ecostress_scale

    if preexisting_losses is not None:
        start_epoch = len(preexisting_losses[0])
    current_epoch = start_epoch
    train_loss = np.zeros(shape=n_epochs + int(start_epoch == 0))
    val_loss = np.zeros(shape=n_epochs + int(start_epoch == 0))
    
    try:
        for epoch in range(n_epochs + int(start_epoch == 0)):
            current_epoch = epoch
            if epoch == start_epoch + n_epochs:
                train_loss_eval_list = []
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                x = x.to(dtype=torch.float, device=device)
                y = y.to(dtype=torch.float, device=device)
                x = model(x)
                loss = loss_fn(x, y.squeeze())
                if epoch > 0 or start_epoch != 0:
                    loss.backward()
                    optimizer.step()
                train_loss[epoch] += (
                    loss.item() / (
                        train_loader.dataset.ecostress_data.detach().numpy().size
                    )
                )
                if epoch == start_epoch + n_epochs:
                    train_loss_eval_list.append(
                        mse_loss(
                            x, y.squeeze(), reduction='none'
                        ).cpu().detach().numpy()
                    )
            train_loss[epoch] = np.sqrt(train_loss[epoch])

            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    if epoch == start_epoch + n_epochs:
                        val_loss_eval_list = []
                    for x, y in val_loader:
                        x = x.to(dtype=torch.float, device=device)
                        y = y.to(dtype=torch.float, device=device)
                        x = model(x)
                        
                        val_loss[epoch] += (
                            loss_fn(x, y.squeeze()).item() / (
                                val_loader.dataset.ecostress_data.detach().numpy().size
                            )
                        )
                        if epoch == start_epoch + n_epochs:
                            val_loss_eval_list.append(
                                mse_loss(
                                    x, y.squeeze(), reduction='none'
                                ).cpu().detach().numpy()
                            )
                    val_loss[epoch] = np.sqrt(val_loss[epoch])
                    
                    if epoch == start_epoch + n_epochs:
                        eval_stats, train_eval_losses, val_eval_losses = (
                            utils.eval.evaluate_model_performance(
                                (
                                    np.array(
                                        train_loader.dataset.ecostress_data.shape
                                    ) -
                                    2 * train_loader.dataset.boundary_width
                                ),
                                train_loader.dataset.ecostress_scale,
                                (
                                    np.array(
                                        val_loader.dataset.ecostress_data.shape
                                    ) -
                                    2 * val_loader.dataset.boundary_width
                                ),
                                val_loader.dataset.ecostress_scale,
                                train_loss_eval_list,
                                val_loss_eval_list,
                            )
                        )

            if (
                loss_interval is not None
                and (epoch % loss_interval == 0 or epoch == n_epochs)
            ):
                print_epoch = ("0" * (3 - len(str(epoch + start_epoch))) + 
                               str(epoch + start_epoch)
                )
                print(
                    f'Epoch {print_epoch}:    ',
                    'Train (RMSE, K):  '
                    f'{train_loss[epoch]:6.5}, ',
                    f'{train_std * train_loss[epoch]:6.5}   \t',
                    'Val (RMSE, K):  '
                    f'{val_loss[epoch]:6.5}, ',
                    f'{val_std * val_loss[epoch]:6.5}   \t',
                    end='',
                )
                if scheduler is not None:
                    print(
                        f'LR: {optimizer.param_groups[0]["lr"]:6.5}\t', end=''
                    )
                else:
                    print('', end='')
                print(f'Time: {time.time() - t:3.3}')
            
            if scheduler is not None:
                scheduler.step(val_loss[epoch])
                
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        train_loss = train_loss[:current_epoch]
        val_loss = val_loss[:current_epoch]
        eval_stats = None
        train_eval_losses = None
        val_eval_losses = None
    
    if preexisting_losses is not None:
        train_loss = np.concatenate([preexisting_losses[0], train_loss])
        val_loss = np.concatenate([preexisting_losses[1], val_loss])

    return train_loss, val_loss, eval_stats, train_eval_losses, val_eval_losses