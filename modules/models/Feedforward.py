import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleFeedforwardModel(nn.Module):
    '''
    Defines a pytorch model made up of dense linear layers and ReLU activation
    '''
    def __init__(self, input_dim: int) -> None:
        super(SimpleFeedforwardModel, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=24)
        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=24
        )
        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=16
        )
        self.linear4 = nn.Linear(
            in_features=self.linear3.out_features, out_features=16
        )
        self.linear5 = nn.Linear(
            in_features=self.linear3.out_features, out_features=8
        )
        self.linear_output = nn.Linear(
            in_features=self.linear5.out_features, out_features=1
        )

        self.batchnorm1 = nn.BatchNorm1d(num_features=self.linear1.out_features)
        self.batchnorm2 = nn.BatchNorm1d(num_features=self.linear2.out_features)
        self.batchnorm3 = nn.BatchNorm1d(num_features=self.linear3.out_features)
        self.batchnorm4 = nn.BatchNorm1d(num_features=self.linear4.out_features)
        self.batchnorm5 = nn.BatchNorm1d(num_features=self.linear5.out_features)


    def forward(self, x: Tensor) -> Tensor:
        x = self.batchnorm1(F.relu(input=self.linear1(x)))
        x = self.batchnorm2(F.relu(input=self.linear2(x)))
        x = self.batchnorm3(F.relu(input=self.linear3(x)))
        x = self.batchnorm4(F.relu(input=self.linear4(x)))
        x = self.batchnorm5(F.relu(input=self.linear5(x)))
        x = self.linear_output(x)
        return x.squeeze()


class ToyModel(nn.Module):
    '''
    Defines a pytorch model made up of dense linear layers and ReLU activation
    '''
    def __init__(self, input_dim: int) -> None:
        super(ToyModel, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=128)

        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=64
        )

        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=32
        )

        # self.linear4 = nn.Linear(
            # in_features=self.linear3.out_features, out_features=16
        # )

        # self.linear5 = nn.Linear(
        #     in_features=self.linear3.out_features, out_features=8
        # )

        self.linear_output = nn.Linear(
            in_features=self.linear3.out_features, out_features=1
        )

        self.batchnorm1 = nn.BatchNorm1d(num_features=self.linear1.out_features)
        self.batchnorm2 = nn.BatchNorm1d(num_features=self.linear2.out_features)
        self.batchnorm3 = nn.BatchNorm1d(num_features=self.linear3.out_features)
        # self.batchnorm4 = nn.BatchNorm1d(num_features=self.linear4.out_features)
        # self.batchnorm5 = nn.BatchNorm1d(num_features=self.linear5.out_features)


    def forward(self, x: Tensor) -> Tensor:
        x = self.batchnorm1(F.relu(input=self.linear1(x)))
        x = self.batchnorm2(F.relu(input=self.linear2(x)))
        x = self.batchnorm3(F.relu(input=self.linear3(x)))
        # x = self.batchnorm4(F.relu(input=self.linear4(x)))
        # x = self.batchnorm5(F.relu(input=self.linear5(x)))
        x = self.linear_output(x)
        return x.squeeze()