import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionModel(nn.Module):
    '''
    Defines a pytorch model using self-attention on the spectra
    '''
    def __init__(self, input_dim: int) -> None:
        super(AttentionModel, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            in_features=input_dim, out_features=64
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

        self.batchnorm1 = nn.BatchNorm1d(num_features=self.linear1.out_features)
        self.batchnorm2 = nn.BatchNorm1d(num_features=self.linear2.out_features)
        self.batchnorm3 = nn.BatchNorm1d(num_features=self.linear3.out_features)
        # self.batchnorm4 = nn.BatchNorm1d(num_features=self.linear2.out_features)



    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.self_attention(x, x, x)
        x = self.batchnorm1(F.relu(input=self.linear1(x)))
        x = self.batchnorm2(F.relu(input=self.linear2(x)))
        x = self.batchnorm3(F.relu(input=self.linear3(x)))
        # x = self.batchnorm4(F.relu(input=self.linear4(x)) + x)
        x = self.linear_output(x)
        return x.squeeze()