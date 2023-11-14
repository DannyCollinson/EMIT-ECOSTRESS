import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (N//4) * (N//4), 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * (N//4) * (N//4))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the size of the input grid (NxN)
N = 10

# Set the number of input channels (spectra + elevations)
input_channels = 2

# Set the number of output channels (1 for predicting temperature)
output_size = 1

# Instantiate the model
model = SimpleCNN(input_channels, output_size)

# Print the model architecture
print(model)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
    