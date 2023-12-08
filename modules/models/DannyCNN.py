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
            padding_mode='zeros',
        )
        
        self.down2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down4 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down5 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
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
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.up3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=48,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.up1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.up0 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
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
    
    
    
class UNet_Temp(nn.Module):
    def __init__(self, y_size: int, x_size: int, input_dim: int) -> None:
        super(UNet, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.input_dim = input_dim
        
        self.down1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down4 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down5 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.up4 = nn.ConvTranspose2d(
            in_channels=33,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.up3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=48,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.up1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.up0 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
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
    


class UNetLarge(nn.Module):
    def __init__(self, y_size: int, x_size: int, input_dim: int) -> None:
        super(UNetLarge, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.input_dim = input_dim
        
        self.down1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, return_indices=True,
        )
        
        self.down2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, return_indices=True,
        )
        
        self.down3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, return_indices=True,
        )
        
        self.down4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool4 = nn.MaxPool2d(
            kernel_size=5, return_indices=True,
        )
        
        self.down5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool5 = nn.MaxPool2d(
            kernel_size=5, return_indices=True,
        )
        
        self.down6 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool6 = nn.MaxPool2d(
            kernel_size=5, return_indices=True,
        )
        
        self.down7 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool7 = nn.MaxPool2d(
            kernel_size=7, return_indices=True,
        )
        
        self.down8 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool8 = nn.MaxPool2d(
            kernel_size=7, return_indices=True,
        )
        
        self.down9 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
        )
        
        self.up9 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=7,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool9 = nn.MaxUnpool2d(
            kernel_size=7,
        )
        
        self.up8 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=256,
            kernel_size=7,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool8 = nn.MaxUnpool2d(
            kernel_size=7,
        )
        
        self.up7 = nn.ConvTranspose2d(
            in_channels=384,
            out_channels=256,
            kernel_size=7,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool7 = nn.MaxUnpool2d(
            kernel_size=7,
        )
        
        self.up6 = nn.ConvTranspose2d(
            in_channels=384,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool6 = nn.MaxUnpool2d(
            kernel_size=5,
        )
        
        self.up5 = nn.ConvTranspose2d(
            in_channels=192,
            out_channels=128,
            kernel_size=5,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool5 = nn.MaxUnpool2d(
            kernel_size=5,
        )
        
        self.up4 = nn.ConvTranspose2d(
            in_channels=192,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool4 = nn.MaxUnpool2d(
            kernel_size=5,
        )
        
        self.up3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool3 = nn.MaxUnpool2d(
            kernel_size=3,
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=96,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.maxunpool2 = nn.MaxUnpool2d(
            kernel_size=3,
        )
        
        self.up1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros',
            output_padding=1,
        )
        
        self.linear_out = nn.Linear(
            in_features=y_size * x_size * 16,
            out_features=y_size * x_size,
        )
        
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(2, 0, 1)
        
        down1 = F.relu(self.down1(x))
        down2, down2_ind = self.maxpool1(down1)
        down2 = F.relu(self.down2(down2))
        down3, down3_ind = self.maxpool2(down2)
        down3 = F.relu(self.down3(down3))
        down4, down4_ind = self.maxpool3(down3)
        down4 = F.relu(self.down4(down4))
        down5, down5_ind = self.maxpool4(down4)
        down5 = F.relu(self.down5(down5))
        down6, down6_ind = self.maxpool5(down5)
        down6 = F.relu(self.down6(down6))
        down7, down7_ind = self.maxpool6(down6)
        down7 = F.relu(self.down7(down7))
        down8, down8_ind = self.maxpool7(down7)
        down8 = F.relu(self.down8(down8))
        down9, down9_ind = self.maxpool8(down8)
        down9 = F.relu(self.down9(down9))
        
        up9 = F.relu(self.up9(down9))
        up8 = self.maxunpool9(up9, down9_ind)
        up8 = torch.cat([up8, down8])
        up8 = F.relu(self.up8(up8))
        up7 = self.maxunpool8(up8, down8_ind)
        up7 = torch.cat([up7, down7])
        up7 = F.relu(self.up7(up7))
        up6 = self.maxunpool7(up7, down7_ind)
        up6 = torch.cat([up6, down6])
        up6 = F.relu(self.up6(up6))
        up5 = self.maxunpool6(up6, down6_ind)
        up5 = torch.cat([up5, down5])
        up5 = F.relu(self.up5(up5))
        up4 = self.maxunpool5(up5, down5_ind)
        up4 = torch.cat([up4, down4])
        up4 = F.relu(self.up4(up4))
        up3 = self.maxunpool4(up4, down4_ind)
        up3 = torch.cat([up3, down3])
        up3 = F.relu(self.up3(up3))
        up2 = self.maxunpool3(up3, down3_ind)
        up2 = torch.cat([up2, down2])
        up2 = F.relu(self.up2(up2))
        up1 = self.maxunpool2(up2, down2_ind)
        up1 = torch.cat([up1, down1])
        up1 = F.relu(self.up1(up1))
        
        out = self.linear_out(up1.flatten())
        
        return out.reshape(self.y_size, self.x_size)
        
        # down1 = F.dropout(F.relu(self.down1(x), 0.25))
        # down2 = F.dropout(F.relu(self.down2(down1), 0.25))
        # down3 = F.dropout(F.relu(self.down3(down2), 0.25))
        # down4 = F.dropout(F.relu(self.down4(down3), 0.25))
        # down5 = F.dropout(F.relu(self.down5(down4), 0.25))
        
        # up5 = torch.cat([down5, ((1, 4, 4))], axis=0)
        
        # up4 = F.relu(self.up4(up5))
        # up4 = torch.cat([down4, up4], axis=0)
        # up3 = F.relu(self.up3(up4))
        # up3 = torch.cat([down3, up3], axis=0)
        # up2 = F.relu(self.up2(up3))
        # up2 = torch.cat([down2, up2], axis=0)
        # up1 = F.relu(self.up1(up2))
        # up1 = torch.cat([down1, up1], axis=0)
        
        # up0 = F.sigmoid(self.up0(up1))
        
        # out = self.linear_out(up0.flatten())
        
        # return out.reshape((self.y_size, self.x_size))
        
        
        
class UNet(nn.Module):
    def __init__(
            self, y_size: int,
            x_size: int,
            input_dim: int,
            dropout_rate: float = 0.0
    ) -> None:
        super(UNet, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        self.down1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2, return_indices=True,
        )
        
        self.down2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, return_indices=True,
        )
        
        self.down3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=2, return_indices=True,
        )
        
        self.down4 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool4 = nn.MaxPool2d(
            kernel_size=2, return_indices=True,
        )
        
        self.down5 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.up5 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            output_padding=0,
        )
        
        self.maxunpool5 = nn.MaxUnpool2d(
            kernel_size=2,
        )
        
        self.up4 = nn.ConvTranspose2d(
            in_channels=32 + self.down4.out_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            output_padding=0,
        )
        
        self.maxunpool4 = nn.MaxUnpool2d(
            kernel_size=2,
        )
        
        self.up3 = nn.ConvTranspose2d(
            in_channels=32 + self.down3.out_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            output_padding=0,
        )
        
        self.maxunpool3 = nn.MaxUnpool2d(
            kernel_size=2,
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=16 + self.down2.out_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            output_padding=0,
        )
        
        self.maxunpool2 = nn.MaxUnpool2d(
            kernel_size=2,
        )
        
        self.up1 = nn.ConvTranspose2d(
            in_channels=16 + self.down1.out_channels,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            output_padding=0,
        )
        
        self.linear_out1 = nn.Linear(
            in_features=y_size * x_size * 4,
            out_features=y_size * x_size,
        )
        
        self.linear_out2 = nn.Linear(
            in_features=y_size * x_size,
            out_features=y_size * x_size,
        )
        
    
    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1)
            concat_axis = 0
        else:
            x = x.permute(0, 3, 1, 2)
            concat_axis = 1
                    
        down1 = F.dropout(F.relu(self.down1(x)), self.dropout_rate)
        down2, down2_ind = self.maxpool1(down1)
        down2 = F.dropout(F.relu(self.down2(down2)), self.dropout_rate)
        down3, down3_ind = self.maxpool2(down2)
        down3 = F.dropout(F.relu(self.down3(down3)), self.dropout_rate)
        down4, down4_ind = self.maxpool3(down3)
        down4 = F.dropout(F.relu(self.down4(down4)), self.dropout_rate)
        down5, down5_ind = self.maxpool4(down4)
        down5 = F.dropout(F.relu(self.down5(down5)), self.dropout_rate)

        up5 = F.dropout(F.relu(self.up5(down5)), self.dropout_rate)
        up4 = self.maxunpool5(up5, down5_ind)
        up4 = torch.cat([up4, down4], axis=concat_axis)
        up4 = F.dropout(F.relu(self.up4(up4)), self.dropout_rate)
        up3 = self.maxunpool4(up4, down4_ind)
        up3 = torch.cat([up3, down3], axis=concat_axis)
        up3 = F.dropout(F.relu(self.up3(up3)), self.dropout_rate)
        up2 = self.maxunpool3(up3, down3_ind)
        up2 = torch.cat([up2, down2], axis=concat_axis)
        up2 = F.dropout(F.relu(self.up2(up2)), self.dropout_rate)
        up1 = self.maxunpool2(up2, down2_ind)
        up1 = torch.cat([up1, down1], axis=concat_axis)
        up1 = F.dropout(F.relu(self.up1(up1)), self.dropout_rate)
        
        if len(up1.shape) == 3:
            out = up1.flatten()
        else:
            out = up1.reshape(up1.shape[0], len(up1.flatten()) // up1.shape[0])
        
        out = F.relu(self.linear_out1(out))
        out = self.linear_out2(out)

        if len(out.shape) == 1:
            out = out.reshape(self.y_size, self.x_size)
        else:
            out = out.reshape(out.shape[0], self.y_size, self.x_size)
        
        return out
    


class PaperCNN(nn.Module):
    def __init__(
            self, y_size: int,
            x_size: int,
            input_dim: int,
            dropout_rate: float = 0.0
    ) -> None:
        super(PaperCNN, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        self.down1 = nn.Conv3d(
            in_channels=input_dim,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        # self.maxpool1 = nn.MaxPool2d(
        #     kernel_size=2, return_indices=True,
        # )
        
        self.down2 = nn.Conv3d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        # self.maxpool2 = nn.MaxPool2d(
        #     kernel_size=2, return_indices=True,
        # )
        
        self.down3 = nn.Conv3d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        # self.maxpool3 = nn.MaxPool2d(
        #     kernel_size=2, return_indices=True,
        # )
        
        self.down4 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        # self.maxpool4 = nn.MaxPool2d(
        #     kernel_size=2, return_indices=True,
        # )
        
        self.down5 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down6 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.down7 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
        )
        
        self.maxpool = nn.MaxPool3d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        self.linear_out1 = nn.Linear(
            in_features=y_size * x_size * 4,
            out_features=200,
        )
        
        self.linear_out2 = nn.Linear(
            in_features=self.linear_out1.out_features,
            out_features=16,
        )
        
    
    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1)
            concat_axis = 0
        else:
            x = x.permute(0, 3, 1, 2)
            concat_axis = 1
                    
        down1 = F.dropout(F.relu(self.down1(x)), self.dropout_rate)
        # down2, down2_ind = self.maxpool1(down1)
        down2 = F.dropout(F.relu(self.down2(down1)), self.dropout_rate)
        # down3, down3_ind = self.maxpool2(down2)
        down3 = F.dropout(F.relu(self.down3(down2)), self.dropout_rate)
        # down4, down4_ind = self.maxpool3(down3)
        down4 = F.dropout(F.relu(self.down4(down3)), self.dropout_rate)
        # down5, down5_ind = self.maxpool4(down4)
        down5 = F.dropout(F.relu(self.down5(down4)), self.dropout_rate)
        down6 = F.dropout(F.relu(self.down5(down5)), self.dropout_rate)
        down7 = F.dropout(F.relu(self.down5(down6)), self.dropout_rate)
        pool = self.maxpool(down7)

        # up5 = F.dropout(F.relu(self.up5(down5)), self.dropout_rate)
        # up4 = self.maxunpool5(up5, down5_ind)
        # up4 = torch.cat([up4, down4], axis=concat_axis)
        # up4 = F.dropout(F.relu(self.up4(up4)), self.dropout_rate)
        # up3 = self.maxunpool4(up4, down4_ind)
        # up3 = torch.cat([up3, down3], axis=concat_axis)
        # up3 = F.dropout(F.relu(self.up3(up3)), self.dropout_rate)
        # up2 = self.maxunpool3(up3, down3_ind)
        # up2 = torch.cat([up2, down2], axis=concat_axis)
        # up2 = F.dropout(F.relu(self.up2(up2)), self.dropout_rate)
        # up1 = self.maxunpool2(up2, down2_ind)
        # up1 = torch.cat([up1, down1], axis=concat_axis)
        # up1 = F.dropout(F.relu(self.up1(up1)), self.dropout_rate)
        
        # if len(up1.shape) == 3:
        #     out = up1.flatten()
        # else:
        #     out = up1.reshape(up1.shape[0], len(up1.flatten()) // up1.shape[0])
        
        out = F.relu(self.linear_out1(pool))
        out = self.linear_out2(out)

        if len(out.shape) == 1:
            out = out.reshape(self.y_size, self.x_size)
        else:
            out = out.reshape(out.shape[0], self.y_size, self.x_size)
        
        return out