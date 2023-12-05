import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleFeedforwardModel(nn.Module):
    '''
    Defines a pytorch model made up of dense linear layers and ReLU activation
    '''
    def __init__(self, input_dim: int) -> None:
        super(SimpleFeedforwardModel, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=512)
        
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
        x = self.layernorm1(F.relu(input=self.linear1(x)))
        x = self.layernorm2(F.relu(input=self.linear2(x)))
        x = self.layernorm3(F.relu(input=self.linear3(x)))
        x = self.layernorm4(F.relu(input=self.linear4(x)))
        x = self.layernorm5(F.relu(input=self.linear5(x)))
        x = self.layernorm6(F.relu(input=self.linear6(x)))
        x = self.linear_output(x)
        return x.squeeze()


class ToyModel(nn.Module):
    '''
    Defines a pytorch model made up of dense linear layers and ReLU activation
    '''
    def __init__(self, input_dim: int) -> None:
        super(ToyModel, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=32)

        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=16
        )

        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=8
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
        
        self.layernorm1 = nn.LayerNorm(self.linear1.out_features)
        self.layernorm2 = nn.LayerNorm(self.linear2.out_features)
        self.layernorm3 = nn.LayerNorm(self.linear3.out_features)

        # self.batchnorm1 = nn.BatchNorm1d(num_features=self.linear1.out_features)
        # self.batchnorm2 = nn.BatchNorm1d(num_features=self.linear2.out_features)
        # self.batchnorm3 = nn.BatchNorm1d(num_features=self.linear3.out_features)
        # self.batchnorm4 = nn.BatchNorm1d(num_features=self.linear4.out_features)
        # self.batchnorm5 = nn.BatchNorm1d(num_features=self.linear5.out_features)


    def forward(self, x: Tensor) -> Tensor:
        x = self.layernorm1(
            F.relu(input=self.linear1(x))
        )
        
        x = self.layernorm2(
            F.relu(input=self.linear2(x))  # + x[:, :self.linear2.out_features]
        )

        x = self.layernorm3(
            F.relu(input=self.linear3(x))  # + x[:, :self.linear3.out_features]
        )

        x = self.linear_output(x)
        return x.squeeze()



class LinearModel(nn.Module):
    '''
    Defines a pytorch model that does linear regression
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
    
    
    
class PatchToPixelModel(nn.Module):
    '''
    Defines a pytorch model made up of dense linear layers and ReLU activation
    that predicts the temperature at one pixel of interest from a patch of
    input data surrounding and including that pixel
    '''
    def __init__(
            self, input_dim: int, radius: int, dropout_rate: float=0.0
    ) -> None:
        super(PatchToPixelModel, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.linear1 = nn.Linear(
            in_features=input_dim * ((2 * radius) + 1)**2, out_features=32
        )
        
        self.layernorm1 = nn.LayerNorm(self.linear1.out_features)

        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features, out_features=16
        )
        
        self.layernorm2 = nn.LayerNorm(self.linear2.out_features)
        
        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features, out_features=8
        )
        
        self.layernorm3 = nn.LayerNorm(self.linear3.out_features)

        # self.linear4 = nn.Linear(
        #     in_features=self.linear3.out_features, out_features=1024
        # )

        # self.layernorm4 = nn.LayerNorm(self.linear4.out_features)

        # self.linear5 = nn.Linear(
        #     in_features=self.linear4.out_features, out_features=512
        # )
        
        # self.layernorm5 = nn.LayerNorm(self.linear5.out_features)

        # self.linear6 = nn.Linear(
        #     in_features=self.linear5.out_features, out_features=256
        # )

        # self.layernorm6 = nn.LayerNorm(self.linear6.out_features)

        # self.linear7 = nn.Linear(
        #     in_features=self.linear6.out_features, out_features=64
        # )
        
        # self.layernorm7 = nn.LayerNorm(self.linear7.out_features)

        self.linear_output = nn.Linear(
            in_features=self.linear3.out_features, out_features=1
        )


    def forward(self, x: Tensor) -> Tensor:
        x = F.dropout(
            self.layernorm1(F.relu(input=self.linear1(x))),
            p=self.dropout_rate
        )
        x = F.dropout(
            self.layernorm2(F.relu(input=self.linear2(x))),
            p=self.dropout_rate
        )
        x = F.dropout(
            self.layernorm3(F.relu(input=self.linear3(x))),
            p=self.dropout_rate
        )
        # x = F.dropout(
        #     self.layernorm4(F.relu(input=self.linear4(x))),
        #     p=self.dropout_rate
        # )
        # x = F.dropout(
        #     self.layernorm5(F.relu(input=self.linear5(x))),
        #     p=self.dropout_rate
        # )
        # x = F.dropout(
        #     self.layernorm6(F.relu(input=self.linear6(x))),
        #     p=self.dropout_rate
        # )
        # x = F.dropout(
        #     self.layernorm7(F.relu(input=self.linear7(x))),
        #     p=self.dropout_rate
        # )
        x = self.linear_output(x)
        return x.squeeze()