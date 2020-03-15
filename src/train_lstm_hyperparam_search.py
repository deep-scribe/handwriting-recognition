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
from pprint import pprint

# training hyperparameter
# pertaining anything that does not modify the model structure
# modify this before running training script
BATCH_SIZE = 1500
CONCAT_TRIM_AUGMENT_PROP = 3
NOISE_AUGMENT_PROP = 3
DEV_PROP = 0.1
TEST_PROP = 0.001
NUM_EPOCH = 15
USE_NONCLASS = True
FILENAME_DESCRIPTION = 'hyperparam_search_1'
NUM_RANDOM_CONFIG = 50
NUM_LSTM_LAYER_RANGE = [1, 8]
LSTM_HIDDEN_DIM_RANGE = [50, 300]
FC_HIDDEN_DIM_RANGE = [50, 400]

# should not change
MODEL_WEIGHT_PATH = '../saved_model/'
MODEL_HIST_PATH = '../output/'
MODEL_CLASS = lstm.LSTM_char_classifier


def main():
    print(f'Training model class [{MODEL_CLASS.__name__}]')
    print()

    # confirm hyperparam
    print('CONFIRM following training parameter as defined on top of train_lstm_hyperparam_search.py')
    print(f'  [BATCH_SIZE]               {BATCH_SIZE}')
    print(f'  [CONCAT_TRIM_AUGMENT_PROP] {CONCAT_TRIM_AUGMENT_PROP}')
    print(f'  [NOISE_AUGMENT_PROP]       {NOISE_AUGMENT_PROP}')
    print(f'  [DEV_PROP]                 {DEV_PROP}')
    print(f'  [TEST_PROP]                {TEST_PROP}')
    print(f'  [USE_NONCLASS]             {USE_NONCLASS}')
    print(f'  [FILENAME_DESCRIPTION]     {FILENAME_DESCRIPTION}')
    print(f'  [NUM_RANDOM_CONFIG]        {NUM_RANDOM_CONFIG}')
    print(f'  [NUM_LSTM_LAYER_RANGE]     {NUM_LSTM_LAYER_RANGE}')
    print(f'  [LSTM_HIDDEN_DIM_RANGE]    {LSTM_HIDDEN_DIM_RANGE}')
    print(f'  [FC_HIDDEN_DIM_RANGE]      {FC_HIDDEN_DIM_RANGE}')
    print()
    input()

    # confirm data subject
    print('CONFIRM following data subjects to be used as defined in data_laoder_upper.py')
    print(' ', data_loader_upper.VERIFIED_SUBJECTS)
    print()
    input()

    # generate some configs
    random_configs = set()
    while len(random_configs) < NUM_RANDOM_CONFIG:
        n_input_channels = 3
        lstm_hidden_dim_low, lstm_hidden_dim_high = LSTM_HIDDEN_DIM_RANGE
        lstm_hidden_dim = np.random.randint(
            lstm_hidden_dim_low, lstm_hidden_dim_high+1
        )
        lstm_n_layers_low, lstm_n_layers_high = NUM_LSTM_LAYER_RANGE
        lstm_n_layers = np.random.randint(
            lstm_n_layers_low, lstm_n_layers_high+1
        )
        fc_hidden_dim_low, fc_hidden_dim_high = FC_HIDDEN_DIM_RANGE
        fc_hidden_dim = np.random.randint(
            fc_hidden_dim_low, fc_hidden_dim_high+1
        )
        num_output = 27  # with nonclass
        bidirectional = False
        use_all_lstm_layer_state = True

        config = (n_input_channels, lstm_hidden_dim, lstm_n_layers,
                  fc_hidden_dim, num_output, bidirectional, use_all_lstm_layer_state)
        if config in random_configs:
            continue
        random_configs.add(config)
    print('[randomly generated configs]')
    pprint(random_configs)
    print()

    # load raw data
    trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_all_classic_random_split(
        DEV_PROP, TEST_PROP,
        resampled=False, flatten=False, keep_idx_and_td=True)
    print('trainx', len(trainx), 'devx', len(devx), 'testx', len(testx))
    print()
    # augment dev set, keeping raw sequences in
    devx, devy = aug_concat_trim(devx, devy)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)
    # dont augment test set
    testx = data_flatten.resample_dataset_list(testx)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)

    for selected_config in random_configs:
        train_model(selected_config, trainx, trainy, devloader, testloader)


def train_model(selected_config, trainx, trainy, devloader, testloader):
    print('\n[training model of config]', selected_config)
    # define filename
    config_strs = [str(int(c)) for c in selected_config]
    s = '-'.join(config_strs)
    now = datetime.now()
    time_str = now.strftime("%m-%d-%H-%M")
    file_prefix = f'{MODEL_CLASS.__name__}.{FILENAME_DESCRIPTION}.{s}.{BATCH_SIZE}-{CONCAT_TRIM_AUGMENT_PROP}-{NOISE_AUGMENT_PROP}.{time_str}'
    weight_filename = file_prefix+'.pth'
    hist_filename = file_prefix+'.json'
    print(f'Model weights will be saved as [{weight_filename}]')
    print(f'Training history will be saved as [{hist_filename}]')

    model = MODEL_CLASS(*selected_config)
    print(f'torch.cuda.is_available()={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.005)
    hist = defaultdict(list)
    best_loss = 1000

    try:
        for epoch in range(NUM_EPOCH):
            running_loss = 0.0
            print(f'Epoch [{epoch}]')

            # augment train set differently every epoch
            # do not keep raw sequence
            # model should only overfit to true handwriting char part
            # but not any other unnecesary signal
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

            # save model if achieve lower dev loss
            # i.e. early stopping
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
