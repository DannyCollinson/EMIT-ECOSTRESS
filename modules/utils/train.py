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
    train_loader: DataLoader,
    val_loader: Union[DataLoader, None] = None,
    n_epochs: int = 10,
    loss_interval: Union[int, None] = 5,
    device: str = 'cpu',
) -> tuple[np.ndarray[np.float64, Any], np.ndarray[np.float64, Any]]:
    begin: float = time.time()
    t: float = begin
    current_epoch: int = 0
    train_loss: np.ndarray[np.float64, Any] = np.zeros(shape=n_epochs+1)
    val_loss: np.ndarray[np.float64, Any] = np.zeros(shape=n_epochs+1)
    try:
        for epoch in range(n_epochs+1):
            current_epoch = epoch
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                x = x.to(dtype=torch.float, device=device)
                y = y.to(dtype=torch.float, device=device)
                x = model(x)
                loss = loss_fn(x, y.squeeze())
                if epoch != 0:
                    loss.backward()
                    optimizer.step()
                train_loss[epoch] += (
                    loss.item() /
                    len(train_loader.dataset) # type: ignore
                )

            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(dtype=torch.float, device=device)
                        y = y.to(dtype=torch.float, device=device)
                        x = model(x)
                        val_loss[epoch] += (
                            loss_fn(x, y.squeeze()).item() /
                            len(val_loader.dataset) # type: ignore
                        )

            if epoch != 0 and scheduler is not None:
                scheduler.step(val_loss[epoch])

            if (
                loss_interval is not None
                and (epoch % loss_interval == 0 or epoch == n_epochs - 1)
            ):
                print(
                    f'Epoch {"0" * (3 - len(str(object=epoch))) + str(object=epoch)}\t\t'
                    f'Train Loss: {train_loss[epoch]:.5}\t\t'
                    f'Val Loss: {val_loss[epoch]:.5} \t\t',
                    end='',
                )
                if scheduler is not None:
                    print(
                        f'LR: {optimizer.param_groups[0]["lr"]:.6}\t\t', end=''
                    )
                else:
                    print('', end='')
                print(f'Time: {time.time() - t}')
                t = time.time()
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
        train_loss = train_loss[:current_epoch]
        val_loss = val_loss[:current_epoch]

    return train_loss, val_loss