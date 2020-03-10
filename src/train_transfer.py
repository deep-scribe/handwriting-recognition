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

# training hyperparameter
# pertaining anything that does not modify the model structure
# modify this before running training script
BATCH_SIZE = 1500
CONCAT_TRIM_AUGMENT_PROP = 1
NOISE_AUGMENT_PROP = 3
DEV_PROP = 0.1
TEST_PROP = 0.001
NUM_EPOCH = 100
USE_NONCLASS = True

# should not change
MODEL_WEIGHT_PATH = '../saved_model/'
MODEL_HIST_PATH = '../output/'
WEIGHT_DIR = '../saved_model/'


def main():
    model_class = lstm.LSTM_char_classifier
    print('Training model class [{}]'.format(model_class.__name__))
    print()

    # confirm hyperparam
    print('CONFIRM following training parameter as defined on top of train_model.py')
    print('  [BATCH_SIZE]               {}'.format(BATCH_SIZE))
    print('  [CONCAT_TRIM_AUGMENT_PROP] {}'.format(CONCAT_TRIM_AUGMENT_PROP))
    print('  [NOISE_AUGMENT_PROP]       {}'.format(NOISE_AUGMENT_PROP))
    print('  [DEV_PROP]                 {}'.format(DEV_PROP))
    print('  [TEST_PROP]                {}'.format(TEST_PROP))
    print('  [USE_NONCLASS]             {}'.format(USE_NONCLASS))
    print()
    input()

    # confirm data subject
    print('CONFIRM following data subjects to be used as defined in data_laoder_upper.py')
    print(' ', data_loader_upper.VERIFIED_SUBJECTS)
    print()
    input()

    # pick config as defined
    print('Select model config to train')
    for idx, c in enumerate(lstm.config):
        assert len(c) == len(lstm.config_keys)
        print('[{}] '.format(idx, end=''))
        for i, item in enumerate(lstm.config_keys):
            print('{}={} '.format(item, c[i], end=''))
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
    config_strs = [str(int(c)) for c in selected_config]
    s = '-'.join(config_strs)
    now = datetime.now()
    time_str = now.strftime("%m-%d-%H-%M")
    file_prefix = '{}.{}.{}-{}-{}.{}'.format(
    model_class.__name__, s, BATCH_SIZE, CONCAT_TRIM_AUGMENT_PROP, NOISE_AUGMENT_PROP, time_str
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
            n = int(input('type a number: '))
            selected_file_path = pth_files_paths[n]
        except KeyboardInterrupt:
            quit()
        except:
            pass
    print()

    model_class, model_param, train_param, train_time, _ = \
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
    assert len(model_param_list) == len(lstm.config_keys)
    for i, c in enumerate(lstm.config_keys):
        print('  {}: {}'.format(c, model_param_list[i]))
    print('[TRAIN PARAMS]')
    print('  batchsize {}'.format(train_param_list[0]))
    print('  concat_trim_aug_prop {}'.format(train_param_list[1]))
    print('  noise_aug_prop {}'.format(train_param_list[2]))
    print()

    # get the class, instantiate model, load weight
    model = globals()[model_class](*model_param_list)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(selected_file_path[1]))
    else:
        model.load_state_dict(torch.load(
            selected_file_path[1], map_location=torch.device('cpu')))

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "fc" in name:
            param.requires_grad = True

    # load raw data
    trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_subjects_classic_random_split(
        DEV_PROP, TEST_PROP,
        resampled=False, flatten=False, keep_idx_and_td=True, subjects = ["Kelly_new"])
    print('trainx', len(trainx), 'devx', len(devx), 'testx', len(testx))
    print()

    # augment dev set, keeping raw sequences in
    devx, devy = aug_concat_trim(devx, devy)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)

    # dont augment test set
    testx = data_flatten.resample_dataset_list(testx)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.005)
    hist = defaultdict(list)
    best_loss = 1000

    try:
        for epoch in range(NUM_EPOCH):
            running_loss = 0.0
            print('Epoch [{}]'.format(epoch))

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
            print('  '.format(end=''))
            for i, data in enumerate(trainloader):
                print('{}'.format([i//10] if i%10==0 else "", end='', flush=True))
                print('{}'.format(i % 10, end='', flush=True))

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

            print('  trainacc={} devacc={}'.format(trainacc, devacc))
            print('  trainloss={} devloss={}'.format(trainloss, devloss))

            # save model if achieve lower dev loss
            # i.e. early stopping
            if best_loss > devloss:
                best_loss = devloss
                torch.save(model.state_dict(), os.path.join(
                    MODEL_WEIGHT_PATH, weight_filename))
                print('  new best dev loss, weight saved')
    except KeyboardInterrupt:
        pass

    print()
    print('Finished Training', 'best dev loss', best_loss)
    testacc, testloss = acc_loss(model, testloader, nn.CrossEntropyLoss())
    testacc, testloss
    hist['testacc'] = testacc
    hist['testloss'] = testloss
    print('test loss={} test acc={}'.format(testloss, testacc))

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
