import torch
import data_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
from collections import defaultdict
import numpy as np
import sys
from torch.nn.utils.rnn import pad_sequence


BATCH_SIZE = 500


def split_ypr(x):
    return x[:, :, 0], x[:, :, 1], x[:, :, 2]


def get_dataloader(x, y, batch_size):
    dataset = [(x[i].T, y[i]) for i in range(y.shape[0])]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def pad_x(input):
    max_length = max(len(input[i]) for i in range(input.shape[0]))
    result = np.zeros((input.shape[0], max_length, 3))
    for i in range(len(input)):
        result[i, :len(input[i]), :] = input[i]
    return result


def pad_all_x(trainx, devx, testx):
    return pad_x(trainx), pad_x(devx), pad_x(testx)


def acc(net, data_loader):
    correct = 0
    total = 0
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
    return correct / total


def acc_loss(net, data_loader, criterion):
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


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, bidirectional=True)
        # self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim*2, 800, bias=True)
        self.fc2 = nn.Linear(800, 500, bias=True)
        self.fc3 = nn.Linear(500, 26, bias=True)

    def forward(self, x):
        init_h = torch.randn(self.n_layers*2, x.shape[0], self.hidden_dim)
        init_c = torch.randn(self.n_layers*2, x.shape[0], self.hidden_dim)
        if torch.cuda.is_available():
            init_h = init_h.cuda()
            init_c = init_c.cuda()
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, (init_h, init_c))
        # out = self.dropout(out)
        # print("inter: ", out.shape)
        out = self.fc(out[:, -1, :])
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        out = torch.nn.functional.relu(out)
        out = self.fc3(out)
        # print("out: ", out.shape)
        return out


def get_net(checkpoint_path):
    net = Net(3, 100, 5)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(checkpoint_path))
    else:
        net.load_state_dict(torch.load(
            checkpoint_path, map_location=torch.device('cpu')))
    return net


def get_prob(net, input):
    if torch.cuda.is_available():
        input = input.cuda()
    else:
        net.cpu()
    net.eval()
    with torch.no_grad():
        logit = net(input.float())
        prob = F.log_softmax(logit, dim=-1)
    return prob


def main():
    print(torch.cuda.is_available())
    print(sys.argv[1:])
    _, experiment_type, resampled, trial = sys.argv
    filename = experiment_type + '_' + resampled + '_' + trial

    if experiment_type == "subject":
        if resampled == "resampled":
            trainx, devx, testx, trainy, devy, testy = data_loader.load_all_subject_split(
                resampled=True, flatten=False)
        else:
            trainx, devx, testx, trainy, devy, testy = data_loader.load_all_subject_split(
                resampled=False, flatten=False)
    else:
        if resampled == "resampled":
            trainx, devx, testx, trainy, devy, testy = data_loader.load_all_classic_random_split(
                resampled=True, flatten=False)
        else:
            trainx, devx, testx, trainy, devy, testy = data_loader.load_all_classic_random_split(
                resampled=False, flatten=False)

    print(trainx.shape, devx.shape, testx.shape,
          trainy.shape, devy.shape, testy.shape)
    if resampled == "resampled":
        trainx, trainy = data_loader.augment_train_set(
            trainx, trainy, augment_prop=3, is_flattened=False, resampled=True)
        trainx, devx, testx = pad_all_x(trainx, devx, testx)
    else:
        trainx, trainy = data_loader.augment_train_set(
            trainx, trainy, augment_prop=3, is_flattened=False, resampled=False)
        trainx, devx, testx = pad_all_x(trainx, devx, testx)
    print(trainx.shape, devx.shape, testx.shape,
          trainy.shape, devy.shape, testy.shape)

    trainloader = get_dataloader(trainx, trainy, BATCH_SIZE)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)
    sample_size, num_feature, num_channel = trainx.shape
    print(sample_size, num_feature, num_channel)

    net = Net(num_channel, 100, 5)
    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.AdamW(net.parameters(), weight_decay=0.01)

    hist = defaultdict(list)
    for epoch in range(250):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            print(f'{i if i%20==0 else ""}.', end='')

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

        trainacc, trainloss = acc_loss(net, trainloader, criterion)
        devacc, devloss = acc_loss(net, devloader, criterion)
        hist['trainacc'].append(trainacc)
        hist['trainloss'].append(trainloss)
        hist['devacc'].append(devacc)
        hist['devloss'].append(devloss)

        print(f'Epoch {epoch} trainacc={trainacc} devacc={devacc}')
        print(f'        trainloss={trainloss} devloss={devloss}')

    print('Finished Training')
    torch.save(net.state_dict(), "../saved_model/rnn_bilstm/" +
               "rnn_bilstm_" + filename + ".pth")

    testacc, testloss = acc_loss(net, testloader, nn.CrossEntropyLoss())
    testacc, testloss
    hist['testacc'] = testacc
    hist['testloss'] = testloss

    with open('../output/rnn_bilstm/rnn_' + filename + '.json', 'w') as f:
        json.dump(hist, f)


if __name__ == '__main__':
    main()
