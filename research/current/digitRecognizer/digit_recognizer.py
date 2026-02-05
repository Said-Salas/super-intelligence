from typing import Any


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

print("\nStarting training")

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 500 == 0:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 500:.3f}')

print('Finished training')
torch.save(net.state_dict(), './mnist_net.pth')
print('Model saved to mnist_net.pth')
