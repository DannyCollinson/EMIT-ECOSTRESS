import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

    

class SemanticSegmenter(nn.Module):
    '''
    Implements the semantic segmentation model from the following article:
    https://towardsdatascience.com/practical-guide-to-semantic-segmentation-7c55b540489c
    with modification to get regression output instead of classification
    '''
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
        
        
        
class UNet(nn.Module):
    '''
    Implements the U-Net outlined in Yao et al.'s
    "Pixel-wise regression using U-Net and its application on pansharpening"
    https://doi.org/10.1016/j.neucom.2018.05.103
    with modification to get single-channel output
    '''
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
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            output_padding=0,
        )
        
        self.linear_out1 = nn.Linear(
            in_features=y_size * x_size,
            out_features=y_size * x_size // 2,
        )
        
        self.linear_out2 = nn.Linear(
            in_features=self.linear_out1.out_features,
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
    
    
    
class SmallUNet(nn.Module):
    '''
    Implements the U-Net outlined in Yao et al.'s
    "Pixel-wise regression using U-Net and its application on pansharpening"
    https://doi.org/10.1016/j.neucom.2018.05.103
    with modification to get single-channel output
    '''
    def __init__(
            self, y_size: int,
            x_size: int,
            input_dim: int,
            dropout_rate: float = 0.0
    ) -> None:
        super(SmallUNet, self).__init__()
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

        self.up2 = nn.ConvTranspose2d(
            in_channels=self.down2.out_channels,
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
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            output_padding=0,
        )
        
        self.linear_out1 = nn.Linear(
            in_features=y_size * x_size,
            out_features=y_size * x_size // 2,
        )
        
        self.linear_out2 = nn.Linear(
            in_features=self.linear_out1.out_features,
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

        up2 = F.dropout(F.relu(self.up2(down2)), self.dropout_rate)
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