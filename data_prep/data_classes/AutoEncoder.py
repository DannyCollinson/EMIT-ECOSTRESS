import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class AutoEncoderWrapper:
    def __init__(self, input_dim, encoding_dim):
        self.model = AutoEncoder(input_dim, encoding_dim)

    def create_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def create_trainer(self) -> pl.Trainer:
        trainer = pl.Trainer(
            max_epochs=10,
            num_sanity_val_steps=0,
            # logger=None,
            # check_val_every_n_epoch=1,
        )
        return trainer

    def fit(self, train_data, val_data):
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


    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
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
        x_recon = self(x)
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
    
    def plotLosses(self):
        epochs = range(1, len(self.train_epoch_loss) + 1)

        plt.plot(epochs, self.train_epoch_loss, label='Training Loss')
        plt.plot(epochs, self.val_epoch_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

    # make residual plot function 