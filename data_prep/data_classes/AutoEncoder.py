import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from datetime import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AutoEncoderWrapper:
    def __init__(self, input_dim, encoding_dim):
        self.model = AutoEncoder(input_dim, encoding_dim)

    def create_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=2048,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def create_trainer(self) -> pl.Trainer:
        trainer = pl.Trainer(
            max_epochs=25,
            num_sanity_val_steps=0,
            # logger=None,
            # check_val_every_n_epoch=1,
        )
        return trainer

    def fit(self, train_data, val_data = None):
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
        pass

    def save(self):
        pass


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim: int = 64):
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

    def forward(self, x):
        self.z = self.encoder(x)
        x_recon = self.decoder(self.z)
        return x_recon

    def training_step(self, batch, batch_idx):
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
        self.train_epoch_loss.append(torch.mean(torch.tensor(self.train_losses)))
        self.train_losses = []

    def validation_step(self, batch, batch_idx):
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
        self.val_epoch_loss.append(torch.mean(torch.tensor(self.val_losses)))
        self.val_losses = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss'  # Make sure this metric is logged in validation_step
        #     }
        # }

    def plotLosses(self, en_dim):
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

    def generate_r2(self):
        # choosing to do by sample / pixel because we want to compare recon samples
        self.r2_values = []
        preds = torch.cat(self.x_recon)
        gt = torch.cat(self.x)
        self.r2_values = [r2_score(preds[idx], gt[idx]) for idx in range(preds.shape[0])]

        # plt.figure()
        # sns.histplot(self.r2_values)
        # plt.show()
        # print(np.mean(self.r2_values))
        return np.mean(self.r2_values)

        # plt.scatter(np.arange(len(self.r2_values)), self.r2_values, label='R2 values')
        # plt.xlabel('Epoch')
        # plt.ylabel('R2')
        # plt.title('R2 Values Over Epochs')
        # plt.legend()
        # plt.show()
        # print("Done")

    def plot_x_xrecon(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(torch.concat(self.x)[0:100], cmap='viridis', annot=False, cbar_kws={'label': 'Intensity'})
        plt.title('Ground Truth')
        # plt.xlabel('')
        # plt.ylabel('')
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

