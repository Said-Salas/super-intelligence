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

class DigitPainter:
    def __init__(self):
        self.window =tk.Tk()
        self.window.title('Draw a Digit (0-9)')

        self.canvas = tk.Canvas(self.window, width=280, height=280, bg='black')
        self.canvas.pack()

        self.image = Image.new('L', (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.paint)

        btn_frame = tk.Frame(self.window)
        btn_frame.pack()

        tk.Button(btn_frame, text='PREDICT', command=self.predict, bg='green').pack(side=tk.LEFT)
        tk.Button(btn_frame, text='CLEAR', command=self.clear).pack(side=tk.LEFT)

        self.label = tk.Label(self.window, text='Draw a number...', font=('Helvetica', 24))
        self.label.pack()

        self.window.mainloop()

        