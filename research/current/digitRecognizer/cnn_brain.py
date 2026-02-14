import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1. Convolutional Layer 1
        # Input: 1 channel (grayscale), Output: 32 channels (32 different filters)
        # Kernel: 3x3 (The size of the sliding window)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 2. Convolutional Layer 2
        # Input: 32 channels, Output: 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 3. Max Pooling (The "Shrinker")
        # Reduces size by half (2x2 window)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 4. Fully Connected Layers (The Classifier)
        # After 2 pools, 28x28 becomes 7x7.
        # 64 channels * 7 * 7 = 3136 inputs
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout (The "Tough Love" Regularizer)
        # Randomly kills 50% of neurons to prevent memorization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Layer 1: Conv -> ReLU -> Pool
        # 28x28 -> 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        
        # Layer 2: Conv -> ReLU -> Pool
        # 14x14 -> 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten: Turn 3D cube (64x7x7) into 1D line
        x = x.view(-1, 64 * 7 * 7)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout before final layer
        x = self.fc2(x)
        return x