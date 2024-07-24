import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.name = 'covnet'
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Define activation function
        self.relu = nn.ReLU()
        
        # Define max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Define fully connected layer
        self.fc = nn.Linear(128 * 50, 128)  # 128 channels * 50 length
        
    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Reshape for fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        
        # Forward pass through fully connected layer
        x = self.fc(x)
        
        return x