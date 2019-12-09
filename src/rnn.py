#cell 0
import torch
import data_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#cell 1
print(torch.cuda.is_available())

#cell 2
trainx, devx, testx, trainy, devy, testy = data_loader.load_all_classic_random_split(flatten=False)

#cell 3
trainx, trainy = data_loader.augment_train_set(trainx, trainy, augment_prop=3, is_flattened=False)
print(trainx.shape, devx.shape, testx.shape, trainy.shape, devy.shape, testy.shape)

#cell 4
BATCH_SIZE = 500

def get_dataloader(x, y, batch_size):
    dataset = [(x[i].T, y[i]) for i in range(y.shape[0])]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

trainloader = get_dataloader(trainx, trainy, BATCH_SIZE)
devloader = get_dataloader(devx, devy, BATCH_SIZE)
testloader = get_dataloader(testx, testy, BATCH_SIZE)

#cell 5
sample_size, num_feature, num_channel = trainx.shape
print(sample_size, num_feature, num_channel)

#cell 6
def acc(data_loader):
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

#cell 7
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 26, bias = True)

    def forward(self, x):
        init_h = torch.randn(self.n_layers, x.shape[0], self.hidden_dim).cuda()
        init_c = torch.randn(self.n_layers, x.shape[0], self.hidden_dim).cuda()
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, (init_h, init_c))
        # print("inter: ", out.shape)
        out = self.fc(out[:,-1,:])
        # print("out: ", out.shape)
        return out

net = Net(num_channel, 100, 5)
if torch.cuda.is_available():
    net.cuda()

#cell 8
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(200):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        print(f'{i if i%20==0 else ""}.', end='')

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

    trainacc = acc(trainloader)
    devacc = acc(devloader)

    print('')
    print(f'Epoch {epoch} trainacc={trainacc} devacc={devacc}')

print('Finished Training')

#cell 9
print(acc(testloader))

#cell 10


#cell 11


#cell 12
