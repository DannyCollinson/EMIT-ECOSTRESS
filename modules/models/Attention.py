import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, concatenate, zeros


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