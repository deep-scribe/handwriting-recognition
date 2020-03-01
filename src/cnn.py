# cell 0
import torch
import data_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
from collections import defaultdict
# import autoencoder
import numpy as np
import data_loader_upper

# cell 1
torch.cuda.is_available()

# cell 2
trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_all_classic_random_split(
    flatten=False)
# trainx, devx, testx, trainy, devy, testy = data_loader.load_all_subject_split(flatten=False)

# cell 3
trainx, trainy = data_loader.augment_train_set(
    trainx, trainy, augment_prop=4, is_flattened=False)
trainx.shape, devx.shape, testx.shape, trainy.shape, devy.shape, testy.shape

# cell 4
# def split_ypr(x):
#     return x[:,:,0],x[:,:,1], x[:,:,2]

# cell 5
# _,_,_,encoder = autoencoder.ae_denoise(*split_ypr(trainx))

# cell 6
# def encode(x, encoder):
#     y,p,r = autoencoder.ae_predict(*split_ypr(x), encoder)
#     return np.stack((y,p,r), axis=2)

# trainx = encode(trainx, encoder)
# devx = encode(devx, encoder)
# testx = encode(testx, encoder)

# cell 7
trainx.shape, devx.shape, testx.shape, trainy.shape, devy.shape, testy.shape

# cell 8
BATCH_SIZE = 500


def get_dataloader(x, y, batch_size):
    dataset = [(x[i].T, y[i]) for i in range(y.shape[0])]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return dataloader


trainloader = get_dataloader(trainx, trainy, BATCH_SIZE)
devloader = get_dataloader(devx, devy, BATCH_SIZE)
testloader = get_dataloader(testx, testy, BATCH_SIZE)

# cell 9
_, num_feature, num_channel = trainx.shape
num_feature, num_channel

# cell 10


def acc_loss(data_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            outputs = net(x.float())
            _, predicted = torch.max(outputs.data, 1)

            w = torch.sum((predicted - y) != 0).item()
            r = len(y) - w
            correct += r
            total += len(y)

            total_loss += criterion(outputs, y.long()).item() * len(x)
    return correct / total, total_loss / total

# cell 11


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

        self.fc1 = nn.Linear(num_feature * 64, 3200)
        self.fc2 = nn.Linear(3200, 1600)
        self.fc3 = nn.Linear(1600, 500)
        self.out = nn.Linear(500, 26)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, num_feature * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


net = Net()
if torch.cuda.is_available():
    net.cuda()
net

# cell 12
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.AdamW(net.parameters(), weight_decay=0.01)

hist = defaultdict(list)
for epoch in range(40):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
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

print('Finished Training')

# cell 13
acc_loss(testloader, nn.CrossEntropyLoss())

# cell 14
testacc, testloss = acc_loss(testloader, nn.CrossEntropyLoss())
testacc, testloss
hist['testacc'] = testacc
hist['testloss'] = testloss

# cell 16

# cell 17


# cell 18


# cell 19


# cell 20


# cell 21


# cell 22


# cell 23
