import lstm
import data_loader_upper
import random
import os
import word_search
import torch
import data_utils

from lstm import LSTM_char_classifier

from pprint import pprint

WEIGHT_DIR = '../saved_model/da/'
WORD_DATA_DIR = '../data_words/'

G = 7  # segment split granularity
K = 10  # word search top k choices


def main():
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

    model_class, model_param, train_param, train_time, description, extension = \
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
    model = globals()[model_class](*model_param_list)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(selected_file_path[1]))
    else:
        model.load_state_dict(torch.load(
            selected_file_path[1], map_location=torch.device('cpu')))

    print('Select uppercase word data to test with')
    word_data_dirs = get_all_word_data_dirs()
    for idx, path in enumerate(word_data_dirs):
        print('[{}] {}'.format(idx, path[0]))
    selected_word_dir = None
    while not selected_word_dir:
        try:
            n = int(input('type a number: '))
            selected_word_dir = word_data_dirs[n]
        except KeyboardInterrupt:
            quit()
        except:
            pass
    print()

    word_df = data_utils.load_subject(selected_word_dir[1])
    wordxs, wordys = data_utils.get_calibrated_yprs_samples(
        word_df, resampled=False, flatten=False,
        is_word_samples=True, keep_idx_and_td=True
    )

    for idx in range(len(wordys)):
        x = wordxs[idx]
        y = wordys[idx]

        trajs = word_search.word_search(x, G, K, model)
        print('Predicting [{}]'.format(y))

        for i, (likelihood, traj) in enumerate(trajs):
            word = ''
            for seg_begin, seg_end, pred, prob in traj:
                word += chr(pred+ord('A'))
            print('  ({}) likelihood {}'.format(i, likelihood), word)


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
