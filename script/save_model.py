#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# library import
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd

import numpy as np

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(42, 100)

        self.fc2 = nn.Linear(100, 100)

        self.fc3 = nn.Linear(100, 100)

        self.fc4 = nn.Linear(100, 100)

        self.fc5 = nn.Linear(100, 3)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)


        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)

        return x

mlp = MLP()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mlp.to(device)

optimizer = optim.SGD(mlp.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

df = pd.read_csv('../data/dataset.csv', header = None)
arr = df.values

# df.head(5)

x = arr[:,:-1]
t = arr[:,-1] 

# print(x)
# print(t)

x_torch = torch.tensor(x, dtype=torch.float32)
t_torch = torch.tensor(t, dtype=torch.int64)

dataset = torch.utils.data.TensorDataset(x_torch, t_torch)

# print(f'dataseat_sample = {dataset[0]}')

n_train = int(len(dataset) * 0.8) 
n_test = len(dataset) - n_train 

# print(f'n_train={n_train}')
# print(f'n_test={n_test}')

torch.manual_seed(0)

train_data, test_data = torch.utils.data.random_split(dataset, [n_train, n_test])

# print(f'train_data = {dataset[0]}')

batchsize = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=False)

# print(len(train_loader))
# print(len(test_loader))

epochs = 90

for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = mlp(inputs)

        optimizer.zero_grad()

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print("[{}, {}] train loss: {}".format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    test_loss = 0.0
    correct = 0

    for i, data in enumerate(test_loader, 0):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = mlp(inputs)

        test_loss += criterion(outputs, labels).item()

        pred = outputs.argmax(dim=1, keepdim=True)

        correct += pred.eq(labels.view_as(pred)).sum().item()
        acc = correct / len(test_data)
        
    if epoch == epochs-1:
      print("epoch: {}".format(epoch + 1))
      print("test accuracy: {}".format(acc))
      print("test loss:{}".format((test_loss / len(test_loader))))


# Save Model
model_path = '../model/model.pth'
torch.save(mlp.state_dict(), model_path)
torch.save(mlp.state_dict(), model_path)


