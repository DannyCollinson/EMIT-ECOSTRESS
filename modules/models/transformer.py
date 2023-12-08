import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TransformerWrapper:
    def __init__(self, input_dim, output_dim):
        self.model = Transformer(input_dim, output_dim)

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

class Transformer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4, num_heads=4):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets)
        self.val_losses.append(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss'
        #     }
        # }