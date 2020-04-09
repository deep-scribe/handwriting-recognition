import rnn_final
import rnn_bilstm
import torch
import data_loader_upper
import data_loader
from sklearn.metrics import confusion_matrix
import pandas as pd
import string

# trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_all_classic_random_split(resampled = True, flatten=False)
#
# BATCH_SIZE = 1000
#
# sample_size, num_feature, num_channel = trainx.shape
# print(sample_size, num_feature, num_channel)
#
# trainloader = rnn_final.get_dataloader(trainx, trainy, BATCH_SIZE)
# net = rnn_final.get_net("../saved_model/rnn_final/rnn_final_random_resampled_4.pth")
#
# # for i, data in enumerate(trainloader):
# input, label = next(iter(trainloader))
# logit = rnn_final.get_prob(net, input)
# # print(logit)
# _, predicted = torch.max(logit.data, 1)
# # print(predicted)
# cm = confusion_matrix(label, predicted)
# index = list(string.ascii_uppercase)
# df = pd.DataFrame(data=cm, index=index, columns=index)
# print(df)

trainx, devx, testx, trainy, devy, testy = data_loader.load_all_classic_random_split(
    resampled=True, flatten=False)
print(len(testy))

BATCH_SIZE = 1050

testloader = rnn_bilstm.get_dataloader(testx, testy, BATCH_SIZE)
net = rnn_bilstm.get_net(
    "../saved_model/rnn_bilstm/rnn_bilstm_random_resampled_0.pth")
print(rnn_bilstm.acc(net, testloader))

# for i, data in enumerate(trainloader):
input, label = next(iter(testloader))
logit = rnn_bilstm.get_prob(net, input)
# print(logit)
_, predicted = torch.max(logit.data, 1)
print(torch.sum((predicted - label) == 0).item()/len(label))
# print(predicted)
cm = confusion_matrix(label, predicted)
index = list(string.ascii_uppercase)
df = pd.DataFrame(data=cm, index=index, columns=index)
print(df)
