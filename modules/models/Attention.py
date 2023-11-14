import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerModel(nn.Module):
    '''
    Defines a pytorch model using self-attention on the spectra
    '''
    def __init__(self, input_dim: int) -> None:
        super(TransformerModel, self).__init__()
        
        self.tranformer = nn.Transformer(
            8, 4, 2, 2, 128, batch_first=True, norm_first=True
        )

        self.linear1 = nn.Linear(
            in_features=128, out_features=64
        )
        
        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=32
        )
        
        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=16
        )
        
        # self.linear4 = nn.Linear(
            # in_features=self.linear3.out_features, out_features=16
        # )
        
        self.linear_output = nn.Linear(
            in_features=self.linear3.out_features, out_features=1
        )

        self.layernorm1 = nn.LayerNorm(self.linear1.out_features)
        self.layernorm2 = nn.LayerNorm(self.linear2.out_features)
        self.layernorm3 = nn.LayerNorm(self.linear3.out_features)


    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.tranformer(x, x, x)
        x = self.layernorm1(F.relu(input=self.linear1(x)))
        x = self.layernorm2(F.relu(input=self.linear2(x)))
        x = self.layernorm3(F.relu(input=self.linear3(x)))
        x = self.linear_output(x)
        return x.squeeze()