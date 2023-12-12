import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    """
    Defines a very simple pytorch CNN model. 
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, 1) 

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional layers with max pooling and ReLU activation
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.pool(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0),-1)


        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        x = x.view(-1, 1, 1)

        return x

    
