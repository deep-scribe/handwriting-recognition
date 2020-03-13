import lstm_siamese
import torch
import data_loader_upper
import torch.nn as nn
import torch.optim as optim
import data_augmentation
import os
import json
import numpy as np
import data_flatten
from collections import defaultdict
from datetime import datetime
import time

# training hyperparameter
# pertaining anything that does not modify the model structure
# modify this before running training script
BATCH_SIZE = 1500
CONCAT_TRIM_AUGMENT_PROP = 1
NOISE_AUGMENT_PROP = 3
DEV_PROP = 0.1
TEST_PROP = 0.1
NUM_EPOCH = 500
USE_NONCLASS = True

# should not change
MODEL_WEIGHT_PATH = '../saved_model/da'
MODEL_HIST_PATH = '../output/da'
WEIGHT_DIR = '../saved_model/'


class DANet(torch.nn.Module):
    def __init__(self, input_dim):
        super(DANet, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim/2), bias = True)
        self.fc2 = nn.Linear(int(input_dim/2), 1, bias = True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)

def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def main():
    start_time = time.time()
    model_class = lstm_siamese.LSTM_char_classifier
    print('Training model class [{}]'.format(model_class.__name__))

    print('Number of epoches:')
    NUM_EPOCH = int(input())

    # pick config as defined
    print('Select model config to train')
    for idx, c in enumerate(lstm_siamese.config):
        assert len(c) == len(lstm_siamese.config_keys)
        print('[{}] '.format(idx, end=''))
        for i, item in enumerate(lstm_siamese.config_keys):
            print('{}={} '.format(item, c[i], end=''))
        print()
    selected_config = None
    while not selected_config:
        try:
            # n = int(input('type a number: '))
            n = 0
            selected_config = lstm_siamese.config[n]
        except KeyboardInterrupt:
            quit()
        except:
            pass
    print()

    # define filename
    description = 'rus_kev_upper'
    while description == '':
        description = input('input a model description (part of filename): ')
    config_strs = [str(int(c)) for c in selected_config]
    s = '-'.join(config_strs)
    now = datetime.now()
    time_str = now.strftime("%m-%d-%H-%M")
    file_prefix = '{}.{}.{}-{}-{}.{}.{}'.format(
    model_class.__name__, s, BATCH_SIZE, CONCAT_TRIM_AUGMENT_PROP, NOISE_AUGMENT_PROP, time_str, description
    )
    weight_filename = file_prefix+'.pth'
    hist_filename = file_prefix+'.json'
    print('Model weights will be saved to [{}]'.format(MODEL_WEIGHT_PATH))
    print('Model weights will be saved as [{}]'.format(weight_filename))
    print('Training history will be saved to [{}]'.format(MODEL_HIST_PATH))
    print('Training history will be saved as [{}]'.format(hist_filename))
    print()

    model = model_class(*selected_config)
    print('torch.cuda.is_available()={}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        model = model.cuda()

    print('Select weight files to load')
    pth_files_paths = get_all_pth_files()
    for idx, path in enumerate(pth_files_paths):
        print('[{}] {}'.format(idx, path[0]))
    selected_file_path = None
    while not selected_file_path:
        try:
            # n = int(input('type a number: '))
            n = 1
            selected_file_path = pth_files_paths[n]
        except KeyboardInterrupt:
            quit()
        except:
            pass
    print()

    model_class, model_param, train_param, train_time, _, _ = \
        selected_file_path[0].split('.')
    model_param_list = model_param.split('-')
    for i in range(len(model_param_list)):
        model_param_list[i] = int(model_param_list[i])
    model_param_list[-1] = bool(model_param_list[-1])
    model_param_list[-2] = bool(model_param_list[-2])
    train_param_list = train_param.split('-')

    print('[SELECTED MODEL]')
    print('  {}'.format(model_class))
    print('[MODEL PARAMS]')
    assert len(model_param_list) == len(lstm_siamese.config_keys)
    for i, c in enumerate(lstm_siamese.config_keys):
        print('  {}: {}'.format(c, model_param_list[i]))
    print('[TRAIN PARAMS]')
    print('  batchsize {}'.format(train_param_list[0]))
    print('  concat_trim_aug_prop {}'.format(train_param_list[1]))
    print('  noise_aug_prop {}'.format(train_param_list[2]))
    print()

    # get the class, instantiate model, load weight
    # model = globals()[model_class](*model_param_list)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(selected_file_path[1]))
    else:
        model.load_state_dict(torch.load(
            selected_file_path[1], map_location=torch.device('cpu')))

    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     if "fc" in name:
    #         param.requires_grad = True

    da_model = DANet(200)
    if torch.cuda.is_available():
        da_model.cuda()

    # load raw data
    da_trainx, devx, testx, da_trainy, devy, testy = data_loader_upper.load_subject_classic_random_split(
        DEV_PROP, TEST_PROP,
        resampled=False, flatten=False, keep_idx_and_td=True, subjects = ["Kelly_new"])
    print('da_trainx', len(da_trainx), 'da_trainy', len(da_trainy), 'devx', len(devx), 'testx', len(testx))
    print()

    or_trainx, _, _, or_trainy, _, _ = data_loader_upper.load_all_classic_random_split(
        0.45, 0.45, resampled=False, flatten=False, keep_idx_and_td=True)
    print('or_trainx', len(or_trainx), 'or_trainy', len(or_trainy))
    print()

    # augment dev set, keeping raw sequences in
    devx, devy = aug_concat_trim(devx, devy)
    devy = np.stack((devy, np.ones_like(devy)), axis = 1)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)

    # dont augment test set
    testx = data_flatten.resample_dataset_list(testx)
    testy = np.stack((testy, np.ones_like(testy)), axis = 1)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    da_criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.005)
    hist = defaultdict(list)
    best_acc = 0

    # try:
    for epoch in range(NUM_EPOCH):
        running_loss = 0.0
        print('Epoch [{}]'.format(epoch))

        # augment train set differently every epoch
        # do not keep raw sequence
        # model should only overfit to true handwriting char part
        # but not any other unnecesary signal
        print('  augment')
        a_trainx_or, a_trainy_or = aug_concat_trim(
            or_trainx, or_trainy, keep_orig=False)
        a_trainx_or, a_trainy_or = data_augmentation.noise_stretch_rotate_augment(
            a_trainx_or, a_trainy_or, augment_prop=NOISE_AUGMENT_PROP,
            is_already_flattened=False, resampled=True)

        a_trainx_da, a_trainy_da = aug_concat_trim(
            da_trainx, da_trainy, keep_orig=False)
        a_trainx_da, a_trainy_da = data_augmentation.noise_stretch_rotate_augment(
            a_trainx_da, a_trainy_da, augment_prop=NOISE_AUGMENT_PROP,
            is_already_flattened=False, resampled=True)

        a_trainy_da = np.stack((a_trainy_da, np.ones_like(a_trainy_da)), axis = 1)
        a_trainy_or = np.stack((a_trainy_or, np.zeros_like(a_trainy_or)), axis = 1)

        # print(a_trainx_or.shape, a_trainy_or.shape, a_trainx_da.shape, a_trainy_da.shape)

        trainx = np.concatenate((a_trainx_da, a_trainx_or), axis = 0)
        trainy = np.concatenate((a_trainy_da, a_trainy_or), axis = 0)

        print('  train')
        trainloader = get_dataloader(trainx, trainy, BATCH_SIZE)
        print('  '.format(end=''))
        lambda_epoch = get_lambda(epoch, NUM_EPOCH)
        for i, data in enumerate(trainloader):
            # print('{}'.format([i//10] if i%10==0 else "", end='', flush=True))
            # print('{}'.format(i % 10, end='', flush=True))
            # print(data)
            if i % 1 == 0:
                print(" ", i, "/", len(trainloader), "Time:", time.time()-start_time)

            inputs, input_labels = data
            labels, da_labels = input_labels[:, 0], input_labels[:, 1]
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                da_labels = da_labels.cuda()
            optimizer.zero_grad()
            outputs, vectors = model(inputs.float())
            da_outputs = da_model(vectors)
            loss = criterion(outputs, labels.long()) - lambda_epoch*da_criterion(da_outputs, da_labels.float())
            loss.backward()
            optimizer.step()

        trainacc, trainloss = acc_loss(model, trainloader, criterion)
        trainacc_da, trainloss_da = acc_loss_da(model, da_model, trainloader, da_criterion)
        devacc, devloss = acc_loss(model, devloader, criterion)
        devacc_da, devloss_da = acc_loss_da(model, da_model, devloader, da_criterion)
        hist['trainacc'].append(trainacc)
        hist['trainloss'].append(trainloss)
        hist['trainacc_da'].append(trainacc_da)
        hist['trainloss_da'].append(trainloss_da)
        hist['devacc'].append(devacc)
        hist['devloss'].append(devloss)
        hist['devacc_da'].append(devacc_da)
        hist['devloss_da'].append(devloss_da)

        print('  trainacc={} devacc={} trainacc_da={} devacc_da={}'.format(trainacc, devacc, trainacc_da, devacc_da))
        print('  trainloss={} devloss={} trainloss_da={} devloss_da={}'.format(trainloss, devloss, trainloss_da, devloss_da))

        # save model if achieve lower dev loss
        # i.e. early stopping
        if best_acc < devacc:
            best_acc = devacc
            torch.save(model.state_dict(), os.path.join(
                MODEL_WEIGHT_PATH, weight_filename))
            print('  new best dev loss, weight saved')
    # except KeyboardInterrupt:
    #     pass

    print()
    print('Finished Training', 'best dev loss', best_loss)
    testacc, testloss = acc_loss(model, testloader, nn.CrossEntropyLoss())
    testacc_da, testloss_da = acc_loss_da(model, da_model, testloader, nn.BCELoss())
    hist['testacc'] = testacc
    hist['testloss'] = testloss
    hist['testacc_da'] = testacc_da
    hist['testloss_da'] = testloss_da
    print('test loss={} test acc={} test acc da={} test loss da={}'.format(testloss, testacc, testacc_da, testloss_da))

    with open(os.path.join(MODEL_HIST_PATH, hist_filename), 'w') as f:
        json.dump(hist, f)


def aug_concat_trim(x, y, keep_orig=True):
    aug_noise_x, aug_noise_y = data_augmentation.get_concat_augment(
        x, y, augment_prop=CONCAT_TRIM_AUGMENT_PROP)
    trimmed_x, trimmed_y = data_augmentation.get_trim_augment(
        x, y, augment_prop=CONCAT_TRIM_AUGMENT_PROP)
    if keep_orig:
        x = np.append(x, aug_noise_x)
        y = np.append(y, aug_noise_y)
        x = np.append(x, trimmed_x)
        y = np.append(y, trimmed_y)
    else:
        x = np.append(aug_noise_x, trimmed_x)
        y = np.append(aug_noise_y, trimmed_y)

    if USE_NONCLASS:
        num_nonclass = len(x) // 20
        nonclass_x, nonclass_y = data_augmentation.get_nonclass_samples(
            x, num_nonclass)
        x = np.append(x, nonclass_x)
        y = np.append(y, nonclass_y)

    x = np.array(data_flatten.resample_dataset_list(x))
    return x, y


def get_dataloader(x, y, batch_size):
    dataset = [(x[i].T, y[i]) for i in range(y.shape[0])]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def acc_loss(net, data_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            x, input_labels = data
            y = input_labels[:, 0]
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            outputs, _ = net(x.float())
            _, predicted = torch.max(outputs.data, 1)

            w = torch.sum((predicted - y) != 0).item()
            r = len(y) - w
            correct += r
            total += len(y)

            total_loss += criterion(outputs, y.long()).item() * len(x)
    return correct / total, total_loss / total


def acc_loss_da(net, da_model, data_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            x, input_labels = data
            y = input_labels[:, 1]
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            _, vectors = net(x.float())
            outputs = da_model(vectors)
            predicted = torch.round(outputs.data)

            w = torch.sum((predicted.float() - y.float()) != 0).item()
            r = len(y) - w
            correct += r
            total += len(y)

            total_loss += criterion(outputs, y.float()).item() * len(x)
    return correct / total, total_loss / total


def get_all_word_data_dirs():
    '''
    return ['dirname','full_path']
    '''
    l = sorted([
        (d, os.path.join(WORD_DATA_DIR, d))
        for d in os.listdir(WORD_DATA_DIR)
        if os.path.isdir(os.path.join(WORD_DATA_DIR, d))
    ])
    return l


def get_all_pth_files():
    '''
    return ['filename','full_path']
    '''

    l = []
    root, dirs, files = list(os.walk(WEIGHT_DIR))[0]
    for name in files:
        if name.split('.')[-1] == 'pth':
            l.append((name, os.path.join(WEIGHT_DIR, name)))
    return l


if __name__ == "__main__":
    main()
