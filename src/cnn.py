import torch
import data_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import numpy as np
import data_loader_upper
from collections import defaultdict

DEVICE = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 500


def get_dataloader(x, y, batch_size):
    dataset = [(x[i].T, y[i]) for i in range(y.shape[0])]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def acc_loss(data_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = net(x.float())
            _, predicted = torch.max(outputs.data, 1)

            w = torch.sum((predicted - y) != 0).item()
            r = len(y) - w
            correct += r
            total += len(y)

            total_loss += criterion(outputs, y.long()).item() * len(x)
    return correct / total, total_loss / total


def get_net(checkpoint_path):
    net = Net()
    net.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return net


def get_prob(net, input):
    net = net.to(DEVICE)
    with torch.no_grad():
        logit = net(input.float())
        prob = F.log_softmax(logit, dim=-1)
    return prob


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        # 16 channel, num_feature
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # 16 channel, num_feature

        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        # 32 channel, num_feature
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)
        # 32 channel, num_feature

        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(100 * 64, 3200)
        self.fc2 = nn.Linear(3200, 500)
        self.out = nn.Linear(500, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 100 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


if __name__ == "__main__":
    print(f'torch.cuda.is_available()={torch.cuda.is_available()}')

    trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_all_classic_random_split(
        flatten=False)
    trainx, trainy = data_loader.augment_train_set(
        trainx, trainy, augment_prop=4, is_flattened=False)

    trainloader = get_dataloader(trainx, trainy, BATCH_SIZE)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)

    _, num_feature, num_channel = trainx.shape
    num_feature, num_channel
    net = Net().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), weight_decay=0.01)
    hist = defaultdict(list)
    for epoch in range(40):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

        trainacc, trainloss = acc_loss(trainloader, criterion)
        devacc, devloss = acc_loss(devloader, criterion)
        hist['trainacc'].append(trainacc)
        hist['trainloss'].append(trainloss)
        hist['devacc'].append(devacc)
        hist['devloss'].append(devloss)

        print(f'Epoch {epoch} trainacc={trainacc} devacc={devacc}')
        print(f'        trainloss={trainloss} devloss={devloss}')

        torch.save(net.state_dict(), "../saved_model/cnn/cnn_1.pth")

    print('Finished Training')

    acc_loss(testloader, nn.CrossEntropyLoss())
    testacc, testloss = acc_loss(testloader, nn.CrossEntropyLoss())
    hist['testacc'] = testacc
    hist['testloss'] = testloss
