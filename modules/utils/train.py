import time
from typing import Union, Any

import numpy as np

import torch
from torch.utils.data import DataLoader

def train(
    model,
    optimizer,
    scheduler,
    loss_fn,
    std_dev: float,
    train_loader: DataLoader,
    val_loader: Union[DataLoader, None] = None,
    n_epochs: int = 10,
    loss_interval: Union[int, None] = 5,
    preexisting_losses: Union[list[np.ndarray, np.ndarray], None] = None,
    device: str = 'cpu',
) -> tuple[np.ndarray[np.float64, Any], np.ndarray[np.float64, Any]]:
    begin: float = time.time()
    t: float = begin
    start_epoch: int = 0
    if preexisting_losses is not None:
        start_epoch = len(preexisting_losses[0])
    current_epoch = start_epoch
    train_loss = np.zeros(shape=n_epochs + int(start_epoch == 0))
    val_loss = np.zeros(shape=n_epochs + int(start_epoch == 0))
    try:
        for epoch in range(n_epochs + int(start_epoch == 0)):
            current_epoch = epoch
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
                train_loss[epoch] += loss.item() / len(train_loader)

            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(dtype=torch.float, device=device)
                        y = y.to(dtype=torch.float, device=device)
                        x = model(x)
                        val_loss[epoch] += (
                            loss_fn(x, y.squeeze()).item() / len(val_loader)
                        )

            if (
                loss_interval is not None
                and (epoch % loss_interval == 0 or epoch == n_epochs - 1)
            ):
                avg_error = std_dev * np.sqrt(val_loss[epoch])
                print_epoch = ("0" * (3 - len(str(epoch + start_epoch))) + 
                               str(epoch + start_epoch)
                )
                print(
                    f'Epoch {print_epoch}\t',
                    f'Train Loss: {train_loss[epoch]:.5}\t',
                    f'Val Loss: {val_loss[epoch]:.5} \t',
                    f'Avg Error: {avg_error:.5}\t',
                    end='',
                )
                if scheduler is not None:
                    print(
                        f'LR: {optimizer.param_groups[0]["lr"]:.6}\t', end=''
                    )
                else:
                    print('', end='')
                print(f'Time: {time.time() - t:.2}')
                t = time.time()
            
            if (epoch > 0 or start_epoch != 0) and scheduler is not None:
                scheduler.step(val_loss[epoch])
                
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        train_loss = train_loss[:current_epoch]
        val_loss = val_loss[:current_epoch]
    
    if preexisting_losses is not None:
        train_loss = np.concatenate([preexisting_losses[0], train_loss])
        val_loss = np.concatenate([preexisting_losses[1], val_loss])

    return train_loss, val_loss