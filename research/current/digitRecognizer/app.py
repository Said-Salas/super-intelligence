from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 == nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print('Loading model...')
device = torch.device('cpu')
net = Net()
model_path = os.path.join(os.path.dirname(__file__), 'mnist_net.pth')
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()



        