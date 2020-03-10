import lstm
import data_loader_upper
import random
import segmentation
import os
import word_search
import torch
import data_utils

from lstm import LSTM_char_classifier

from pprint import pprint

BATCH_SIZE = 150
CONCAT_TRIM_AUGMENT_PROP = 1
NOISE_AUGMENT_PROP = 3
DEV_PROP = 0.1
TEST_PROP = 0.001
NUM_EPOCH = 100
USE_NONCLASS = True

WEIGHT_DIR = '../saved_model/'
WORD_DATA_DIR = '../data_words/'
WEIGHT_SIAMESE_DIR = '../saved_model/siamese'

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label1, label2):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-(label1==label2)) * torch.pow(euclidean_distance, 2) +
                                      (label1==label2) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


def main():
    print('Select weight files to load')
    pth_files_paths = get_all_pth_files()
    for idx, path in enumerate(pth_files_paths):
        print(f'[{idx}] {path[0]}')
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

    print(f'[SELECTED MODEL]')
    print(f'  {model_class}')
    print(f'[MODEL PARAMS]')
    assert len(model_param_list) == len(lstm.config_keys)
    for i, c in enumerate(lstm.config_keys):
        print(f'  {c}: {model_param_list[i]}')
    print(f'[TRAIN PARAMS]')
    print(f'  batchsize {train_param_list[0]}')
    print(f'  concat_trim_aug_prop {train_param_list[1]}')
    print(f'  noise_aug_prop {train_param_list[2]}')
    print()

    # get the class, instantiate model, load weight
    model = globals()[model_class](*model_param_list)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(selected_file_path[1]))
    else:
        model.load_state_dict(torch.load(
            selected_file_path[1], map_location=torch.device('cpu')))

    trainx, devx, testx, trainy, devy, testy = data_loader_upper.load_all_classic_random_split(
        DEV_PROP, TEST_PROP,
        resampled=False, flatten=False, keep_idx_and_td=True)
    print('trainx', len(trainx), 'devx', len(devx), 'testx', len(testx))
    print()

    a_siamesex, a_siamesey = aug_concat_trim(
        siamese_x, siamese_y, keep_orig=False)
    a_siamesex, a_siamesey = data_augmentation.noise_stretch_rotate_augment(
        a_siamesex, a_siamesey, augment_prop=NOISE_AUGMENT_PROP,
        is_already_flattened=False, resampled=True)

    # augment dev set, keeping raw sequences in
    devx, devy = aug_concat_trim(devx, devy)
    devloader = get_dataloader(devx, devy, BATCH_SIZE)

    # dont augment test set
    testx = data_flatten.resample_dataset_list(testx)
    testloader = get_dataloader(testx, testy, BATCH_SIZE)

    siamese_df = data_utils.load_subject(selected_word_dir[1])
    siamese_x, siamese_y = data_utils.get_calibrated_yprs_samples(
        word_df, resampled=False, flatten=False,
        is_word_samples=True, keep_idx_and_td=True
    )

    siamese_dev_x = np.resize(a_siamesex, devx.shape)
    siamese_dev_y = np.resize(a_siamesey, devy.shape)
    s_devloader = get_dataloader(siamese_dev_x, siamese_dev_y, BATCH_SIZE)

    siamese_test_x = np.resize(a_siamesex, testx.shape)
    siamese_test_y = np.resize(a_siamesey, testy.shape)
    s_testloader = get_dataloader(siamese_test_x, siamese_test_y, BATCH_SIZE)


    clf_criterion = nn.CrossEntropyLoss()
    siamese_criterion = ContrastiveLoss()
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

            a_siamesex, a_siamesey = aug_concat_trim(
                siamese_x, siamese_y, keep_orig=False)
            a_siamesex, a_siamesey = data_augmentation.noise_stretch_rotate_augment(
                a_siamesex, a_siamesey, augment_prop=NOISE_AUGMENT_PROP,
                is_already_flattened=False, resampled=True)

            siamese_batch_x = np.resize(a_siamesex, a_trainx.shape)
            siamese_batch_y = np.resize(a_siamesey, a_trainy.shape)


            print('  train')
            trainloader = get_dataloader(a_trainx, a_trainy, BATCH_SIZE)
            s_trainloader = get_dataloader(siamese_batch_x, siamese_batch_y, BATCH_SIZE)
            print('  ', end='')
            trainloss = 0
            for i, data in enumerate(zip(trainloader, s_trainloader)):
                print(f'{[i//10] if i%10==0 else ""}', end='', flush=True)
                print(i % 10, end='', flush=True)

                (inputs, labels), (s_inputs, s_labels) = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    s_inputs = inputs.cuda()
                    s_labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs.float())
                s_outputs = model(ins_inputsputs.float())
                loss = siamese_criterion(outputs, labels.long(), s_outputs, labels.long(), s_labels.long())
                loss.backward()
                optimizer.step()
                trainloss += loss.item()
            print()

            trainacc = acc(model, s_trainloader)
            devacc, devloss = acc_loss(model, devloader, s_devloader, siamese_criterion)
            hist['trainacc'].append(trainacc)
            hist['trainloss'].append(trainloss/len(trainloader))
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
    testacc, testloss = acc_loss(model, testloader, s_testloader, siamese_criterion)
    print("Test ACC:", testacc, "Test Loss:", testloss)
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


def acc_loss(net, data_loader, s_data_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, s_data in zip(data_loader, s_data_loader):
            x, y = data
            s_x, s_y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                s_x = x.cuda()
                s_y = y.cuda()

            outputs = net(x.float())
            s_outputs = net(s_x.float())
            _, predicted = torch.max(s_outputs.data, 1)

            w = torch.sum((predicted - y) != 0).item()
            r = len(y) - w
            correct += r
            total += len(y)

            total_loss += criterion(outputs, s_outputs, y.long(), s_y.long()).item() * len(x)
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
