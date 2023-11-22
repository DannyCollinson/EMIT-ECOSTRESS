import torch
import torch.nn as nn
from torch import Tensor


class TestCNN(nn.Module):
    def __init__(self, y_size: int, x_size: int, input_dim: int) -> None:
        super(TestCNN, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.input_dim = input_dim
        
        
        
        self.linear = nn.Linear(1,1)
        
    def forward(self, x: Tensor) -> Tensor:
        x0 = x[:,:,0]
        x0 = x0 + self.linear(x0[0:1,0:1])
        return x0[:,:]