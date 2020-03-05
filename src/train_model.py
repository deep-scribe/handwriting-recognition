import lstm
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

BATCH_SIZE = 1500
CONCAT_TRIM_AUGMENT_PROP = 1
NOISE_AUGMENT_PROP = 3
DEV_PROP = 0.1
TEST_PROP = 0.001
NUM_EPOCH = 100

MODEL_WEIGHT_PATH = '../saved_model/'
MODEL_HIST_PATH = '../output/'


def main():
    model_class = lstm.LSTM_char_classifier

    print(f'Training model class [{model_class.__name__}]')
    print('  confirm / modify config as defined on top of scipt:')
    print(f'  [BATCH_SIZE]               {BATCH_SIZE}')
    print(f'  [CONCAT_TRIM_AUGMENT_PROP] {CONCAT_TRIM_AUGMENT_PROP}')
    print(f'  [NOISE_AUGMENT_PROP]       {NOISE_AUGMENT_PROP}')
    print(f'  [DEV_PROP]                 {DEV_PROP}')
    print(f'  [TEST_PROP]                {TEST_PROP}')
    print()

    # pick config as defined
    print('Select model config to train')
    for idx, c in enumerate(lstm.config):
        assert len(c) == len(lstm.config_keys)
        print(f'[{idx}] ', end='')
        for i, item in enumerate(lstm.config_keys):
            print(f'{item}={c[i]} ', end='')
        print()
    selected_config = None
    while not selected_config:
        try:
            n = int(input('type a number: '))
            selected_config = lstm.config[n]
        except KeyboardInterrupt:
            quit()
        except:
            pass
    print()

    # define filename
    config_strs = [str(int(c)) for c in selected_config[1:]]
    s = '-'.join(config_strs)
    now = datetime.now()
    time_str = now.strftime("%m-%d-%H:%M")
    file_prefix = f'{model_class.__name__}.{s}.{BATCH_SIZE}-{CONCAT_TRIM_AUGMENT_PROP}-{NOISE_AUGMENT_PROP}.{time_str}'
    weight_filename = file_prefix+'.pth'
    hist_filename = file_prefix+'.json'
    print(f'Model weights will be saved to [{MODEL_WEIGHT_PATH}]')
    print(f'Model weights will be saved as [{weight_filename}]')
    print(f'Training history will be saved to [{MODEL_HIST_PATH}]')
    print(f'Training history will be saved as [{hist_filename}]')
    print()

    model = model_class(*selected_config)
    print(f'torch.cuda.is_available()={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        model = model.cuda()

    # load raw data
    trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_all_classic_random_split(
        DEV_PROP, TEST_PROP,
        resampled=False, flatten=False, keep_idx_and_td=True)
    print('trainx', len(trainx), 'devx', len(devx), 'testx', len(testx))
    print()

    devx, devy = aug_concat_trim(devx, devy)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.005)
    hist = defaultdict(list)
    best_loss = 1000

    try:
        for epoch in range(NUM_EPOCH):
            running_loss = 0.0
            print(f'Epoch [{epoch}]')
            print('  augment')
            a_trainx, a_trainy = aug_concat_trim(
                trainx, trainy, keep_orig=False)
            a_trainx, a_trainy = data_augmentation.noise_stretch_rotate_augment(
                a_trainx, a_trainy, augment_prop=NOISE_AUGMENT_PROP,
                is_already_flattened=False, resampled=True)

            print('  train')
            trainloader = get_dataloader(a_trainx, a_trainy, BATCH_SIZE)
            print('  ', end='')
            for i, data in enumerate(trainloader):
                print(f'{[i//10] if i%10==0 else ""}', end='', flush=True)
                print(i % 10, end='', flush=True)

                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
            print()

            trainacc, trainloss = acc_loss(model, trainloader, criterion)
            devacc, devloss = acc_loss(model, devloader, criterion)
            hist['trainacc'].append(trainacc)
            hist['trainloss'].append(trainloss)
            hist['devacc'].append(devacc)
            hist['devloss'].append(devloss)

            print(f'  trainacc={trainacc} devacc={devacc}')
            print(f'  trainloss={trainloss} devloss={devloss}')
            if best_loss > devloss:
                best_loss = devloss
                torch.save(model.state_dict(), os.path.join(
                    MODEL_WEIGHT_PATH, weight_filename))
                print(f'  new best dev loss, weight saved')
    except KeyboardInterrupt:
        pass

    print()
    print('Finished Training', 'best dev loss', best_loss)
    testacc, testloss = acc_loss(model, testloader, nn.CrossEntropyLoss())
    testacc, testloss
    hist['testacc'] = testacc
    hist['testloss'] = testloss
    print(f'test loss={testloss} test acc={testacc}')

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


if __name__ == "__main__":
    main()
