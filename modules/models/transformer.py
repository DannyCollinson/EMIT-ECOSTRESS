import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4, num_heads=4):
        super(TransformerModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Fully connected layer for output
        x = self.fc(x)
        
        return x

# Define the input and output dimensions
input_dim = 244
output_dim = 1

# Create an instance of the Transformer model
model = TransformerModel(input_dim, output_dim)

# Print the model architecture
print(model)