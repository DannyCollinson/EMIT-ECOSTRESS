import os
import sys
import importlib
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data.Datasets
import models.Feedforward
import models.Attention
import models.cnn
import utils.train
import utils.loading


# import utils.loading dataset stuff
class runModel():

    def __init__(self, model_type, elevation_boolean, data_type) -> None:
        self.model_type = model_type
        self.elevation_boolean =  elevation_boolean
        self.data_type = data_type
        self.project_path = os.getcwd()

        self.emit_train = None
        self.eco_train = None
        self.elev_train = None


        self.emit_val = None
        self.eco_val = None
        self.elev_val = None

        self.train_dataset = None
        self.val_dataset = None

        self.train_loader = None
        self.val_loader = None

        self.train_loss = None
        self.val_loss = None


        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.omit_components = 244 - 244

        self.batch_size = 256

        self.base_data_path = (self.join_path('data'))

        self.train_splits = ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18']
        self.val_splits = ['01', '05', '09', '13', '17']
        self.test_splits = ['03', '07', '11', '15', '19']


    def join_path(self, relative_path: str) -> str:
        return os.path.join(self.project_path, relative_path)

    def pickle_load(self, relative_path: str):  # -> pickled_file_contents
        return pickle.load(open(self.join_path(relative_path), 'rb'))

    def pickle_save(self, obj: object, relative_path: str) -> None:
        pickle.dump(obj, open(self.join_path(relative_path), 'wb'))

    def get_data_arrays(self):
        self.emit_train, self.emit_val, self.eco_train, self.eco_val, self.elev_train, self.elev_val = utils.loading.run_script(
            self.base_data_path,self.train_splits, self.val_splits, self.data_type, self.batch_size, self.elevation_boolean)

        
    def create_dataloader(self):
        self.train_dataset = data.Datasets.EmitEcostressDataset(
            emit_data=self.emit_train,
            omit_components=self.omit_components,
            ecostress_data=self.eco_train,
            ecostress_center=None,
            ecostress_scale=None,
            additional_data=(self.elev_train,),
            device=self.device,
        )

        self.val_dataset = data.Datasets.EmitEcostressDataset(
            emit_data=self.emit_val,
            omit_components=self.omit_components,
            ecostress_data=self.eco_val,
            ecostress_center=None,
            ecostress_scale=None,
            additional_data=(self.elev_val,),
            device=self.device,
        )

        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, drop_last=True
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, drop_last=True
        )

    def model(self):
        '''
        Edit this function for customized running of the model parameters
        
        '''
        n_epochs = 200
        if self.model_type == "cnn":
            model = models.cnn.SimpleCNN(
                ...
            )
        elif self.model_type == 'attention':
            model = models.Attention.TransformerModel(
                ...
            )
        elif self.model_type == "feedforward":
            model = models.Feedforward.SimpleFeedforwardModel(
                input_dim=self.train_dataset.input_dim
            )

        if self.device == 'cuda':
            model = model.cuda()

        optimizer = optim.Adam(
            params=model.parameters(), lr=0.0001, weight_decay=0, fused=True
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=2
        )
        # scheduler = None

        loss_fn = nn.MSELoss(reduction='sum')

        self.train_loss, self.val_loss = utils.train.train(
            model,
            optimizer,
            scheduler,
            loss_fn,
            self.train_loader,
            self.val_loader,
            n_epochs=n_epochs,
            loss_interval=1,
            preexisting_losses=[self.train_loss, self.val_loss],
            device=self.device,
        )


    def run_model(self):
        self.get_data_arrays()
        self.create_dataloader()
        self.model()
        return self.train_loss, self.val_loss


if __name__ == "__main__":
    # TODO "cnn" || "attention" || "feedforward"
    model_type = "feedforward"
    # TODO "raw" || "PCA" || "AE"
    data_type = "raw"
    model_class = runModel("cnn", False, "raw")
    train_loss, val_loss = model_class.run_model()




