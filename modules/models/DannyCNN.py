import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DannyCNN(nn.Module):
    def __init__(self, y_size: int, x_size: int, input_dim: int) -> None:
        super(DannyCNN, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, out_channels=128, kernel_size=3
        ) # decreases height and width by 2

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3
        ) # decreases height and width by 2

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3
        ) # decreases height and width by 2
        
        self.pool1 = nn.AdaptiveMaxPool2d(output_size=(20, 20))
        
        self.layernorm1 = nn.LayerNorm(normalized_shape=[32, 19, 19])

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3
        ) # decreases height and width by 2

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3
        ) # decreases height and width by 2

        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=3
        ) # decreases height and width by 2

        self.pool2 = nn.AdaptiveMaxPool2d(output_size=(10, 10))
        
        self.layernorm2 = nn.LayerNorm(normalized_shape=[16, 13, 13])
        
        self.convT1 = nn.ConvTranspose2d(
            in_channels=16, out_channels=8, kernel_size=2, stride=2
        ) # takes height, width to 20, 20
                
        self.convT2 = nn.ConvTranspose2d(
            in_channels=8, out_channels=4, kernel_size=2, stride=2
        ) # takes height, weight to 40, 40
        
        self.layernorm3 = nn.LayerNorm(normalized_shape=[4, 52, 52])
                
        self.linear1 = nn.Linear(
            in_features=4 * 52 * 52, out_features=y_size * x_size * 8
        )
        
        self.layernorm4 = nn.LayerNorm(normalized_shape=[y_size * x_size * 8])
        
        self.linear2 = nn.Linear(
            in_features=self.linear1.out_features,
            out_features=y_size * x_size * 4
        )
        
        self.layernorm5 = nn.LayerNorm(normalized_shape=[y_size * x_size * 4])
        
        self.linear3 = nn.Linear(
            in_features=self.linear2.out_features,
            out_features=y_size * x_size * 2
        )

        self.layernorm6 = nn.LayerNorm(normalized_shape=[y_size * x_size * 2])
        
        self.linear4 = nn.Linear(
            in_features=self.linear3.out_features,
            out_features=y_size * x_size
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(*torch.arange(x.ndim - 1, -1, -1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.layernorm1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.layernorm2(x)
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.layernorm3(x)
        
        x = x.flatten()
        x = self.layernorm4(F.relu(self.linear1(x)))
        x = self.layernorm5(F.relu(self.linear2(x)))
        x = self.layernorm6(F.relu(self.linear3(x)))
        x = self.linear4(x)
        
        x = x.reshape((self.y_size, self.x_size))
        return x
    

class SemanticSegmenter(nn.Module):
    def __init__(self, y_size: int, x_size: int, input_dim: int) -> None:
        super(SemanticSegmenter, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.input_dim = input_dim
        
        self.down1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        self.down2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        self.down3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        self.down4 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        self.down5 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        self.midlinear1 = nn.Linear(
            in_features=32 * 4 * 4, out_features=64
        )
        
        self.midlinear2 = nn.Linear(in_features=64, out_features=32)
        
        self.midlinear3 = nn.Linear(in_features=32, out_features=16)
        
        self.up4 = nn.ConvTranspose2d(
            in_channels=33,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        
        self.up3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=48,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        
        self.up1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        
        self.up0 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        
        self.linear_out = nn.Linear(
            in_features=y_size*x_size, out_features=y_size*x_size,
        )
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(2, 0, 1)
        
        down1 = F.dropout(F.relu(self.down1(x), 0.25))
        down2 = F.dropout(F.relu(self.down2(down1), 0.25))
        down3 = F.dropout(F.relu(self.down3(down2), 0.25))
        down4 = F.dropout(F.relu(self.down4(down3), 0.25))
        down5 = F.dropout(F.relu(self.down5(down4), 0.25))
        
        mid1 = F.dropout(F.sigmoid(self.midlinear1(down5.flatten())), 0.25)
        mid2 = F.dropout(F.sigmoid(self.midlinear2(mid1)), 0.25)
        mid3 = F.sigmoid(self.midlinear3(mid2))
        
        up5 = torch.cat([down5, mid3.reshape((1, 4, 4))], axis=0)
        
        up4 = F.relu(self.up4(up5))
        up4 = torch.cat([down4, up4], axis=0)
        up3 = F.relu(self.up3(up4))
        up3 = torch.cat([down3, up3], axis=0)
        up2 = F.relu(self.up2(up3))
        up2 = torch.cat([down2, up2], axis=0)
        up1 = F.relu(self.up1(up2))
        up1 = torch.cat([down1, up1], axis=0)
        
        up0 = F.sigmoid(self.up0(up1))
        
        out = self.linear_out(up0.flatten())
        
        return out.reshape((self.y_size, self.x_size))