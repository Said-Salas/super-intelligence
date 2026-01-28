import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading dataset...")
train_set = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Script Running! Training images: {len(train_set)}, Test images: {len(test_set)}")

# data_iter = iter(train_loader)
# images, labels = next(data_iter)
# print(f"Image shape: {images[0].shape}")

class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) #Extracts raw features(lines/edges)
        self.fc2 = nn.Linear(128, 64) #Combines features into concepts (loops, crossings)
        self.fc3 = nn.Linear(64, 10) #Takes the concepts and makes the final vote (0-9)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer =  optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
