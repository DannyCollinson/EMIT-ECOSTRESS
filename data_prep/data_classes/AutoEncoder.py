import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
from sklearn.metrics import r2_score
import numpy as np

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

    
        # def fit(self, train_data, val_data):
        # train_dataloader = self.create_dataloader(train_data)
        # val_dataloader = self.create_dataloader(val_data)
        # trainer = self.create_trainer()
        # trainer.fit(self.model, train_dataloader, val_dataloader)

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
        self.z = []


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
        x_recon = self(x)
        self.x.append(x)
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
    
    def plotLosses(self):
        epochs = range(1, len(self.train_epoch_loss) + 1)

        plt.plot(epochs, self.train_epoch_loss, label='Training Loss')
        plt.plot(epochs, self.val_epoch_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.yscale('log')
        plt.legend()
        plt.show()

    # make residual plot function 
    # def plot_weight_differences(self):
        # Get the weights of the encoder and decoder layers
        # encoder_weights = self.encoder[0].weight.data
        # decoder_weights = self.decoder[-1].weight.data

        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(encoder_weights, cmap='viridis')
        # plt.title('Encoder Weights')
        # plt.colorbar()

        # plt.subplot(1, 2, 2)
        # plt.imshow(decoder_weights, cmap='viridis')
        # plt.title('Decoder Weights')
        # plt.colorbar()

        # plt.tight_layout()
        # plt.show()

    def generate_r2(self):
        # choosing to do by sample / pixel because we want to compare recon samples
        self.r2_values = []
        preds = torch.cat(self.x_recon)
        gt = torch.cat(self.x)
        for idx in range(preds.shape[0]):
            self.r2_values.append(r2_score(preds[idx], gt[idx]))

        print(np.mean(self.r2_values))

        # plt.scatter(np.arange(len(self.r2_values)), self.r2_values, label='R2 values')
        # plt.xlabel('Epoch')
        # plt.ylabel('R2')
        # plt.title('R2 Values Over Epochs')
        # plt.legend()
        # plt.show()
        # print("Done")

    def plot_x_xrecon(self):
        epochs = range(1, len(self.train_epoch_loss) + 1)

        plt.plot(epochs, self.x, label='ground truth')
        plt.plot(epochs, self.x_recon, label='predictions')
        plt.title('Ground Truth vs Predictions Over Epochs')
        plt.legend()
        plt.show()
