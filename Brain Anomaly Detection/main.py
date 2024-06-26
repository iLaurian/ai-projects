import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage import io


class CTImages(Dataset):
    def __init__(self, filename, root_dir, transform=None):
        self.annotations = pd.read_csv(filename, skipinitialspace=True, delimiter=',', lineterminator='\n',
                                       dtype={'id': 'string', 'class': 'int8'}).to_numpy()
        # self.annotations = self.annotations.reshape(len(self.annotations) // 2, 2)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, f'{self.annotations[index][0]}.png')
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations[index][1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label


train_set = CTImages(filename='./data/train_labels.txt', root_dir='data/data', transform=transforms.ToTensor())
test_set = CTImages(filename='./data/validation_labels.txt', root_dir='data/data', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_set, batch_size=50, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=50, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 64, 5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


model = CNN()

# loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 5], dtype=torch.float32))
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

tp, fp, fn = 0, 0, 0


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 150 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t{loss.item():.06f}')


def test():
    model.eval()

    test_loss = 0
    correct = 0
    true_positives, false_positives, false_negatives = 0, 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            true_positives += ((pred == 1) & (target == 1)).sum().item()
            false_positives += ((pred == 1) & (target != 1)).sum().item()
            false_negatives += ((pred != 1) & (target == 1)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return true_positives, false_positives, false_negatives


if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
        tp, fp, fn = test()
        print(f'{2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn)))}\n')
