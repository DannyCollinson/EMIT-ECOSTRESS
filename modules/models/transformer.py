import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

class TransformerWrapper:
    """
    Wrapper class for managing a PyTorch Lightning Transformer model.

    Attributes:
    - model (Transformer): Instance of the Transformer model.

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
    def __init__(self, input_dim: int, output_dim: int):
        self.model = Transformer(input_dim, output_dim)

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

class Transformer(pl.LightningModule):
    """
    PyTorch Lightning Module implementing a Transformer model for a specific task.

    Attributes:
    - embedding (nn.Linear): Linear layer for input data embedding.
    - transformer_encoder (nn.TransformerEncoder): Transformer encoder layer.
    - fc (nn.Linear): Linear layer for final output.
    - train_losses (list): List to store training losses during training.
    - val_losses (list): List to store validation losses during training.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass of the model.

    - training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        Training step of the model.

    - validation_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        Validation step of the model.

    - configure_optimizers() -> torch.optim.Optimizer:
        Configures the optimizer for the model.

    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 4, num_heads: int = 4):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def training_step(self, batch):
        """
        Training step of the model.

        Parameters:
        - batch (Tuple[torch.Tensor, torch.Tensor]): Input-output pair batch.

        Returns:
        - torch.Tensor: Loss value.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, batch):
        """
        Validation step of the model.

        Parameters:
        - batch (Tuple[torch.Tensor, torch.Tensor]): Input-output pair batch.

        Returns:
        - torch.Tensor: Loss value.
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets)
        self.val_losses.append(loss)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
        - torch.optim.Optimizer: Optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)