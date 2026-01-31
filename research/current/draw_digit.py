import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print('Loading brain...')
net = Net()
net.load_state_dict(torch.load('./mnist_net.pth', map_location=torch.device('cpu')))
net.eval()

