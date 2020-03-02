import torch
import data_loader_upper
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
from collections import defaultdict
import numpy as np
import sys
from torch.nn.utils.rnn import pad_sequence
import data_augmentation
import data_flatten
from torch.nn.utils.rnn import pack_sequence


def get_dataloader(x, y, batch_size):
    dataset = [(x[i].T, y[i]) for i in range(y.shape[0])]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return dataloader


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


def get_net(checkpoint_path):
    net = LSTMencdec(n_channels=3)
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
    return logit


class LSTMencdec(nn.Module):
    def __init__(self, n_channels, hidden_dim=200, n_layers=3, bidirectional=True):
        super(LSTMencdec, self).__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_dir = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            n_channels, hidden_dim, n_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim*n_layers*self.num_dir, 200, bias=True)
        self.fc2 = nn.Linear(200, 26, bias=True)

    def forward(self, x):
        init_h = torch.randn(
            self.n_layers * self.num_dir,
            x.shape[0],
            self.hidden_dim)
        init_c = torch.randn(
            self.n_layers * self.num_dir,
            x.shape[0],
            self.hidden_dim)
        if torch.cuda.is_available():
            init_h = init_h.cuda()
            init_c = init_c.cuda()

        x = x.permute(0, 2, 1)

        output, (h_n, c_n) = self.lstm(x, (init_h, init_c))
        # c_n (num_layers * num_directions, batch, hidden_size)
        c_n = c_n.permute(1, 0, 2)
        # c_n (batch, num_layers * num_directions, hidden_size)
        c_n_flat = c_n.reshape(-1, self.n_layers*2*self.hidden_dim)
        # c_n (batch, num_layers * num_directions * hidden_size)
        out = self.fc(c_n_flat)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out


def main():
    print(torch.cuda.is_available())
    print(sys.argv[1:])
    _, trial = sys.argv
    filename = '_' + trial

    trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_all_classic_random_split(
        resampled=False, flatten=False, keep_idx_and_td=True)

    def aug_head_tail(x, y):
        x, y = data_augmentation.augment_head_tail_noise(
            x, y, augment_prop=5)
        x = data_flatten.resample_dataset_list(x)
        x = np.array(x)
        return x, y

    trainx, trainy = aug_head_tail(trainx, trainy)
    devx, devy = aug_head_tail(devx, devy)
    testx, testy = aug_head_tail(testx, testy)

    trainx, trainy = data_loader_upper.augment_train_set(
        trainx, trainy, augment_prop=3,
        is_flattened=False, resampled=True)

    BATCH_SIZE = 500
    trainloader = get_dataloader(trainx, trainy, BATCH_SIZE)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)

    net = LSTMencdec(n_channels=3)
    if torch.cuda.is_available():
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), weight_decay=0.005)

    hist = defaultdict(list)
    best_loss = 1000
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            print(f'{i if i%20==0 else ""}.', end='', flush=True)

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
        if best_loss > devloss:
            best_loss = devloss
            torch.save(net.state_dict(), "../saved_model/lstm_encdec/" +
                       "lstm_encdec_" + filename + ".pth")

    print('Finished Training', 'Best Dev Loss', best_loss)

    net.load_state_dict(torch.load("../saved_model/lstm_encdec/" +
                                   "lstm_encdec_" + filename + ".pth"))

    testacc, testloss = acc_loss(net, testloader, nn.CrossEntropyLoss())
    testacc, testloss
    hist['testacc'] = testacc
    hist['testloss'] = testloss
    print(f'test loss={testloss} test acc={testacc}')

    with open('../output/lstm_encdec/lstm_encdec_' + filename + '.json', 'w') as f:
        json.dump(hist, f)


if __name__ == '__main__':
    main()
