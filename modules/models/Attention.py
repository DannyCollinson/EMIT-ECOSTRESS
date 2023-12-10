import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, concatenate, zeros


class SelfAttentionModel(nn.Module):
    '''
    Defines a pytorch model using self-attention on the spectra
    '''
    def __init__(self, input_dim: int) -> None:
        super(SelfAttentionModel, self).__init__()
        
        self.attention = nn.MultiheadAttention(256, 16, batch_first=True)

        self.linear1 = nn.Linear(
            in_features=256, out_features=64
        )
        
        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=32
        )
        
        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=16
        )
        
        self.linear_output = nn.Linear(
            in_features=self.linear3.out_features, out_features=1
        )

        self.layernorm1 = nn.LayerNorm(self.linear1.out_features)
        self.layernorm2 = nn.LayerNorm(self.linear2.out_features)
        self.layernorm3 = nn.LayerNorm(self.linear3.out_features)


    def forward(self, x: Tensor) -> Tensor:
        if x.shape[2] < 256:
            x = concatenate(
                [x, zeros((x.shape[0], x.shape[1], 256-x.shape[2]))], dim=-1
            )
        x, _ = self.attention(x, x, x)
        x = self.layernorm1(F.relu(input=self.linear1(x)))
        x = self.layernorm2(F.relu(input=self.linear2(x)))
        x = self.layernorm3(F.relu(input=self.linear3(x)))
        x = self.linear_output(x)
        return x.squeeze()