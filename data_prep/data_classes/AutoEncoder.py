import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
import numpy as np
import seaborn as sns
from datetime import datetime
import os

class AutoEncoderWrapper:
    """
    Wrapper class for managing a PyTorch Lightning AutoEncoder model.

    Attributes:
    - model (AutoEncoder): Instance of the AutoEncoder model.

    Methods:
    - create_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        Creates a DataLoader for the given dataset.

    - create_trainer() -> pl.Trainer:
        Creates a PyTorch Lightning Trainer for training the model.

    - fit(train_data: torch.utils.data.Dataset, val_data: torch.utils.data.Dataset = None):
        Fits the model to the training data. If validation data is provided, performs validation during training.

    - predict():
        Placeholder method for making predictions with the trained model.

    - save():
        Placeholder method for saving the trained model.

    """
    def __init__(self, input_dim: int, encoding_dim: int):
        self.model = AutoEncoder(input_dim, encoding_dim)

    def create_dataloader(self, dataset: torch.utils.data.Dataset):
        """
        Creates a DataLoader for the given dataset.

        Parameters:
        - dataset (torch.utils.data.Dataset): The dataset to be loaded.

        Returns:
        - torch.utils.data.DataLoader: DataLoader for the provided dataset.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=2048,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def create_trainer(self) -> pl.Trainer:
        """
        Creates a PyTorch Lightning Trainer for training the model.

        Returns:
        - pl.Trainer: PyTorch Lightning Trainer instance.
        """
        trainer = pl.Trainer(
            max_epochs=25,
            num_sanity_val_steps=0,
        )
        return trainer

    def fit(self, train_data: torch.utils.data.Dataset, val_data: torch.utils.data.Dataset = None):
        """
        Fits the model to the training data. If validation data is provided, performs validation during training.

        Parameters:
        - train_data (torch.utils.data.Dataset): Training dataset.
        - val_data (torch.utils.data.Dataset, optional): Validation dataset.

        """
        if val_data is None:
            train_data = self.create_dataloader(train_data)
            trainer = self.create_trainer()
            trainer.fit(self.model, train_data)
        else:
            train_dataloader = self.create_dataloader(train_data)
            val_dataloader = self.create_dataloader(val_data)
            trainer = self.create_trainer()
            trainer.fit(self.model, train_dataloader, val_dataloader)


    def predict(self):
        """
        Placeholder method for making predictions with the trained model.
        """
        pass

    def save(self):
        """
        Placeholder method for saving the trained model.
        """
        pass


class AutoEncoder(pl.LightningModule):
    """
    PyTorch Lightning Module implementing an AutoEncoder model.

    Attributes:
    - encoder (nn.Sequential): Encoder layers.
    - decoder (nn.Sequential): Decoder layers.
    - train_losses (list): List to store training losses during training.
    - val_losses (list): List to store validation losses during training.
    - train_epoch_loss (list): List to store average training loss per epoch.
    - val_epoch_loss (list): List to store average validation loss per epoch.
    - x (List[torch.Tensor]): List to store input data during validation.
    - x_recon (List[torch.Tensor]): List to store reconstructed data during validation.
    - metric (R2Score): R2Score metric for evaluation.
    - r2_values (list): List to store R2 values during validation.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass of the model.

    - training_step(batch: torch.Tensor) -> torch.Tensor:
        Training step of the model.

    - on_train_epoch_end() -> None:
        Actions to be performed at the end of each training epoch.

    - validation_step(batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        Validation step of the model.

    - on_validation_epoch_end() -> None:
        Actions to be performed at the end of each validation epoch.

    - configure_optimizers() -> torch.optim.Optimizer:
        Configures the optimizer for the model.

    - plot_losses(en_dim: int):
        Plots and saves the training and validation loss over epochs.

    - plot_x_xrecon():
        Plots and displays heatmaps of ground truth, predictions, and their absolute differences.

    """
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.train_losses = []
        self.val_losses = []
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.x = []
        self.x_recon = []
        self.metric = R2Score()
        self.r2_values = []

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        self.z = self.encoder(x)
        x_recon = self.decoder(self.z)
        return x_recon

    def training_step(self, batch):
        """
        Training step of the model.

        Parameters:
        - batch (Tuple[torch.Tensor, torch.Tensor]): Input-output pair batch.

        Returns:
        - torch.Tensor: Loss value.
        """
        x = batch
        x_recon = self(x)
        loss = nn.MSELoss()(x_recon, x)
        self.log('train_loss', loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
        )
        self.train_losses.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Tracking training loss.
        """
        self.train_epoch_loss.append(torch.mean(torch.tensor(self.train_losses)))
        self.train_losses = []

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Parameters:
        - batch (Tuple[torch.Tensor, torch.Tensor]): Input-output pair batch.
        - batch_idx (int): Batch index.

        Returns:
        - torch.Tensor: Loss value.
        """
        x = batch
        self.x.append(x)
        x_recon = self(x)
        self.x_recon.append(x_recon)
        loss = nn.MSELoss()(x_recon, x)
        self.log('val_loss', loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Tracking validation loss.
        """
        self.val_epoch_loss.append(torch.mean(torch.tensor(self.val_losses)))
        self.val_losses = []

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
        - torch.optim.Optimizer: Optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def plotLosses(self, en_dim):
        """
        Plots and saves the training and validation loss over epochs.

        Parameters:
        - en_dim (int): Dimension of the latent space.

        """
        # Add your own path
        save_dir = '/Users/gabriellatwombly/Desktop/CS 101/EMIT-ECOSTRESS/loss plots/'

        os.makedirs(save_dir, exist_ok=True)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        file_name = f"updated_loss_plot_dim_{en_dim}_{current_time}.png"

        save_path = os.path.join(save_dir, file_name)

        epochs = np.arange(1, len(self.train_epoch_loss) + 1)

        plt.plot(epochs, self.train_epoch_loss, label='Training Loss')
        plt.plot(epochs, self.val_epoch_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.yscale('log')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_x_xrecon(self):
        """
        Plots and displays heatmaps of ground truth, predictions, and their absolute differences.

        Not currenly called in script but available to use.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(torch.concat(self.x)[0:100], cmap='viridis', annot=False, cbar_kws={'label': 'Intensity'})
        plt.title('Ground Truth')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.heatmap(torch.concat(self.x_recon)[0:100], cmap='viridis', annot=False, cbar_kws={'label': 'Intensity'})
        plt.title('Predictions')
        plt.show()

        differences = torch.abs(torch.concat(self.x)[0:100] - torch.concat(self.x_recon)[0:100])

        plt.figure(figsize=(10, 6))
        sns.heatmap(differences, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Absolute Difference'})
        plt.title('Difference between Ground Truth and Predictions')
        plt.show()

