"""
*****************************************
** Usage for realtime word prediction: **
*****************************************
    from pipeline import Pipeline
    
    # you can choose model you want if set to False, otherwise use default model
    pipl = Pipeline(use_default_model = False) 

    word_df = data_utils.load_subject(path_to_realtime_foldername)

    predicted_word = pipl.predict_realtime(word_df, G = 7, K = 10)

    print("Predicted word is", predicted_word)

"""

import lstm
import data_loader_upper
import random
import os
import word_search
import torch
import data_utils
import sym_spell
import collections
import numpy as np

from lstm import LSTM_char_classifier
from pprint import pprint

WEIGHT_DIR = '../saved_model/'
WORD_DATA_DIR = '../data_words/'
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class Autocorrect_kernel:

    def identity(trajectory_score, frequency, edit_distance, beta=0.75):
        return 1

    def confidence_only(trajectory_score, frequency, edit_distance, beta=0.75):
        return trajectory_score

    def hard_freq_dist(trajectory_score, frequency, edit_distance, beta=0.75):
        ratio = np.log(frequency) * 1.0 / (edit_distance * 100.0 + 1)
        return ratio * trajectory_score

    def soft_freq_dist(trajectory_score, frequency, edit_distance, beta=0.75):
        ratio = np.power(np.log(frequency), beta / (edit_distance+1))
        return ratio * trajectory_score

    kernels = {
        "top_1": None,
        "identity": identity,
        "confidence_only": confidence_only,
        "hard_freq_dist": hard_freq_dist,
        "soft_freq_dist": soft_freq_dist,
    }


DEFAULT_PTH_FILE = ('LSTM_char_classifier.rus_kev_upper.3-200-3-200-27-0-1.1500-3-3.03-11-03-57.pth',
                    '../saved_model/LSTM_char_classifier.rus_kev_upper.3-200-3-200-27-0-1.1500-3-3.03-11-03-57.pth')
DEFAULT_WORD_FILE = ('russell_new_2', '../data_words/russell_new')
DEFAULT_KERNEL = Autocorrect_kernel.hard_freq_dist


class Pipeline():

    def __init__(self, pth_filepath=DEFAULT_PTH_FILE, word_filepath=DEFAULT_WORD_FILE, ac_kernel_name='hard_freq_dist', use_default_model=True):

        if use_default_model:
            self.model = self.load_model(pth_filepath)
        else:
            self.change_model()

        self.word_data = self.load_wordfile(word_filepath)
        self.autocorrector = sym_spell.initialize()
        self.ac_kernel = Autocorrect_kernel.kernels[ac_kernel_name]

    def predict_realtime(self, word_df, G=7, K=10, verbose=False):
        """
        Run the entire prediction pipeline and predict the word

        @param word_df (panda object): the raw input loaded using utils.load_subject
        @return predictions (list): a list of predictions depending number of words in word_df
        """

        model = self.model
        wordxs, _ = data_utils.get_calibrated_yprs_samples(
            word_df, resampled=False, flatten=False,
            is_word_samples=True, keep_idx_and_td=True
        )

        predictions = []
        for idx in range(len(wordxs)):
            x = wordxs[idx]
            final_word = self.predict_single(
                x, G, K, verbose
            )
            if verbose:
                print("Prediction:", final_word)

            predictions.append(final_word)

        return predictions

    def predict_single(self, x, G=7, K=10, verbose=False):
        confidence_map = []
        trajs = word_search.word_search(x, G, K, self.model)
        for i, (likelihood, traj) in enumerate(trajs):
            word = ''
            for seg_begin, seg_end, pred, prob in traj:
                word += chr(pred+ord('A'))
            confidence_map.append((likelihood, word))

        alpha_score, final_word = self.summerize_final_word(
            confidence_map, verbose=verbose)
        return final_word

    def predict_testfiles(self, G=7, K=10):
        """
        @param G(int): segment split granularity
        @param K(int): word search top k choices
        """
        model = self.model
        wordxs, wordys = self.word_data

        correct = 0

        for idx in range(len(wordys)):
            x = wordxs[idx]
            y = wordys[idx]
            confidence_map = []

            trajs = word_search.word_search(x, G, K, model)
            print(f'({idx}) Predicting [{y}]')

            for i, (likelihood, traj) in enumerate(trajs):
                word = ''
                for seg_begin, seg_end, pred, prob in traj:
                    word += chr(pred+ord('A'))
                # print(f'  ({i}) likelihood {likelihood}', word)
                confidence_map.append((likelihood, word))

            alpha_score, final_word = self.summerize_final_word(
                confidence_map, verbose=True)
            print(f'    result: alpha_score {alpha_score}', final_word)

            if y.lower() == final_word.lower():
                correct += 1

        total_accuracy = correct * 1.0 / len(wordys)
        print("Correct: {}, Total: {}".format(correct, len(wordys)))
        print("Total accuracy: {}".format(total_accuracy))

        return total_accuracy

    def change_model(self, new_file_path=None):

        if not new_file_path:
            print('Select weight files to load')
            pth_files_paths = get_all_pth_files()
            for idx, path in enumerate(pth_files_paths):
                print(f'[{idx}] {path[0]}')
            selected_file_path = None
            while not selected_file_path:
                try:
                    n = int(input('type a number: '))
                    selected_file_path = pth_files_paths[n]
                    print(selected_file_path)
                except KeyboardInterrupt:
                    quit()
                except:
                    pass
            print()
            new_file_path = selected_file_path

        self.model = self.load_model(new_file_path)

    def change_wordfile(self, new_word_dir=None):

        if not new_word_dir:
            print('Select uppercase word data to test with')
            word_data_dirs = get_all_word_data_dirs()
            for idx, path in enumerate(word_data_dirs):
                print(f'[{idx}] {path[0]}')
            selected_word_dir = None
            while not selected_word_dir:
                try:
                    n = int(input('type a number: '))
                    selected_word_dir = word_data_dirs[n]
                    print(selected_word_dir)
                except KeyboardInterrupt:
                    quit()
                except:
                    pass
            print()
            new_word_dir = selected_word_dir

        self.word_data = self.load_wordfile(new_word_dir)

    def change_ac_kernel(self, ac_kernel_name):

        self.ac_kernel = Autocorrect_kernel.kernels[ac_kernel_name]

    def load_wordfile(self, selected_word_dir):

        word_df = data_utils.load_subject(selected_word_dir[1])
        wordxs, wordys = data_utils.get_calibrated_yprs_samples(
            word_df, resampled=False, flatten=False,
            is_word_samples=True, keep_idx_and_td=True
        )

        return wordxs, wordys

    def load_model(self, selected_file_path):

        model_class, description, model_param, train_param, train_time, extension = \
            selected_file_path[0].split('.')
        model_param_list = model_param.split('-')
        for i in range(len(model_param_list)):
            model_param_list[i] = int(model_param_list[i])
        model_param_list[-1] = bool(model_param_list[-1])
        model_param_list[-2] = bool(model_param_list[-2])
        train_param_list = train_param.split('-')

        print(f'\n[CURRENT MODEL]')
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
        model = globals()[model_class](*model_param_list).to(DEVICE)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(selected_file_path[1]))
        else:
            model.load_state_dict(torch.load(
                selected_file_path[1], map_location=torch.device('cpu')))

        return model

    def summerize_final_word(self, confidence_map, verbose=False):
        predictor = self.autocorrector
        kernel_func = self.ac_kernel

        # obtain just the top result
        if self.ac_kernel == None:
            ac_word, ac_dist, ac_freq = predictor.auto_correct(
                confidence_map[0][1], verbose=verbose)
            # print("top_1:",confidence_map[0][1],ac_word)
            return confidence_map[0][0], ac_word

        new_confidence_map = collections.defaultdict(float)

        for confidence, word in confidence_map:
            ac_word, ac_dist, ac_freq = predictor.auto_correct(
                word, verbose=verbose)
            alpha_score = kernel_func(confidence, ac_freq, ac_dist, beta=1)
            new_confidence_map[ac_word] += alpha_score

            # print (ac_word, alpha_score)
        return max(new_confidence_map.values()), max(new_confidence_map, key=new_confidence_map.get)


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


def main():
    pipeline = Pipeline()
    pipeline.change_model()
    pipeline.change_wordfile()
    pipeline.predict_testfiles()


if __name__ == "__main__":
    main()
