import rnn_bilstm
import torch
import data_loader

trainx, devx, testx, trainy, devy, testy = data_loader.load_all_subject_split(resampled = True, flatten=False)

BATCH_SIZE = 250

sample_size, num_feature, num_channel = trainx.shape
print(sample_size, num_feature, num_channel)

testloader = rnn_bilstm.get_dataloader(testx, testy, BATCH_SIZE)

input, _ = next(iter(testloader))

net = rnn_bilstm.get_net("../saved_model/rnn_bilstm/rnn_bilstm_subject_resampled_0.pth")

logit = rnn_bilstm.get_logit(net, input)
print(logit)
_, predicted = torch.max(logit.data, 1)
print(predicted)
