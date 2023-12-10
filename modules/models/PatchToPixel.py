import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, concatenate, zeros



class LargeDenseNN(nn.Module):
    '''
    Defines a large densely-connected feedforward pytorch neural network model
    made up of linear layers of sizes 512, 256, 128, 64, 32, and 16,
    followed by ReLU activation, layer normalization, and optional dropout,
    and ending with a linear output layer that returns a float
    '''
    def __init__(
            self, input_dim: int, radius: int, dropout_rate: float
        ) -> None:
        super(LargeDenseNN, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.linear1 = nn.Linear(
            in_features=(((2 * radius) + 1)**2) * input_dim,
            out_features=512
        )
        
        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=256
        )
        
        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=128
        )
        
        self.linear4 = nn.Linear(
            in_features=self.linear3.out_features, out_features=64
        )
        
        self.linear5 = nn.Linear(
            in_features=self.linear4.out_features, out_features=32
        )
        
        self.linear6 = nn.Linear(
            in_features=self.linear5.out_features, out_features=16
        )
        
        self.linear_output = nn.Linear(
            in_features=self.linear6.out_features, out_features=1
        )

        self.layernorm1 = nn.LayerNorm(self.linear1.out_features)
        self.layernorm2 = nn.LayerNorm(self.linear2.out_features)
        self.layernorm3 = nn.LayerNorm(self.linear3.out_features)
        self.layernorm4 = nn.LayerNorm(self.linear4.out_features)
        self.layernorm5 = nn.LayerNorm(self.linear5.out_features)
        self.layernorm6 = nn.LayerNorm(self.linear6.out_features)


    def forward(self, x: Tensor) -> Tensor:
        x = F.dropout(
            self.layernorm1(F.relu(input=self.linear1(x))), self.dropout_rate
        )
        x = F.dropout(
            self.layernorm2(F.relu(input=self.linear2(x))), self.dropout_rate
        )
        x = F.dropout(
            self.layernorm3(F.relu(input=self.linear3(x))), self.dropout_rate
        )
        x = F.dropout(
            self.layernorm4(F.relu(input=self.linear4(x))), self.dropout_rate
        )
        x = F.dropout(
            self.layernorm5(F.relu(input=self.linear5(x))), self.dropout_rate
        )
        x = F.dropout(
            self.layernorm6(F.relu(input=self.linear6(x))), self.dropout_rate
        )
        x = self.linear_output(x)
        return x.squeeze()
    


class SmallDenseNN(nn.Module):
    '''
    Defines a small densely-connected feedforward pytorch neural network model
    made up of linear layers of sizes 32, 16, and 8,
    followed by ReLU activation, layer normalization, and optional dropout,
    and ending with a linear output layer that returns a float
    '''
    def __init__(
            self, input_dim: int, radius: int, dropout_rate: float
        ) -> None:
        super(SmallDenseNN, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.linear1 = nn.Linear(
            in_features=(((2 * radius) + 1)**2) * input_dim,
            out_features=32
        )
        
        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=16
        )
        
        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=8
        )
        
        self.linear_output = nn.Linear(
            in_features=self.linear3.out_features, out_features=1
        )

        self.layernorm1 = nn.LayerNorm(self.linear1.out_features)
        self.layernorm2 = nn.LayerNorm(self.linear2.out_features)
        self.layernorm3 = nn.LayerNorm(self.linear3.out_features)


    def forward(self, x: Tensor) -> Tensor:
        x = F.dropout(
            self.layernorm1(F.relu(input=self.linear1(x))), self.dropout_rate
        )
        x = F.dropout(
            self.layernorm2(F.relu(input=self.linear2(x))), self.dropout_rate
        )
        x = F.dropout(
            self.layernorm3(F.relu(input=self.linear3(x))), self.dropout_rate
        )
        x = self.linear_output(x)
        return x.squeeze()
    


class MiniDenseNN(nn.Module):
    '''
    Defines a mini densely-connected feedforward pytorch neural network model
    made up of one linear layer of size 4,
    followed by ReLU activation, layer normalization, and optional dropout,
    and ending with a linear output layer that returns a float
    '''
    def __init__(
        self, input_dim: int, radius: int, dropout_rate: float = 0
    ) -> None:
        super(MiniDenseNN, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.linear1 = nn.Linear(
            in_features=(((2 * radius) + 1)**2) * input_dim,
            out_features=4,
        )

        self.linear_output = nn.Linear(
            in_features=self.linear1.out_features, out_features=1
        )
        
        self.layernorm1 = nn.LayerNorm(self.linear1.out_features)


    def forward(self, x: Tensor) -> Tensor:
        x = F.dropout(
            self.layernorm1(
                F.relu(input=self.linear1(x))
            ),
            self.dropout_rate,
        )

        x = self.linear_output(x)
        return x.squeeze()



class LinearModel(nn.Module):
    '''
    Defines pytorch neural network model
    that only has a linear output layer that returns a float
    '''
    def __init__(
            self, input_dim: int, radius: int, dropout_rate: float = 0
    ) -> None:
        super(LinearModel, self).__init__()
        self.linear_output = nn.Linear(
                    in_features=(((2 * radius) + 1)**2) * input_dim,
                    out_features=1,
        )
        
        self.dropout_rate = dropout_rate


    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(self.linear_output(x), self.dropout_rate).squeeze()
    
    

class SelfAttentionModel(nn.Module):
    '''
    Defines a pytorch model using self-attention on the spectra
    '''
    def __init__(
            self, input_dim: int, radius: int, dropout_rate: float
        ) -> None:
        '''
        input_dim, dropout_rate, and radius
        are only here for interface compatability
        '''
        super(SelfAttentionModel, self).__init__()
        
        self.attention = nn.MultiheadAttention(256, 4, batch_first=True)
        
        self.linear_output = nn.Linear(
            in_features=256, out_features=1
        )


    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] < 256:
            x = concatenate(
                [
                    x, zeros(x.shape[0], 256 - x.shape[1], device=x.device)
                ], dim=-1
            )
        x, _ = self.attention(x, x, x)
        x = self.linear_output(x)
        return x.squeeze()