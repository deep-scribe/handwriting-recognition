import beam
import rnn_final
import rnn_bilstm
import data_utils
import segmentation
import word_search
import data_flatten
import numpy as np
import torch
from pprint import pprint
import data_loader
import cnn
import random
import lstm_encdec

# MODEL_WEIGHT_PATH = '../saved_model/rnn_bilstm/rnn_bilstm_random_resampled_0.pth'
MODEL_WEIGHT_PATH = '../saved_model/LSTM_char_classifier.100-5-100-27-0-1.1500-1-3.03-05-17:39.pth'

'''
Test the feasibility to use trajectory_search to reconstruct word
methodology:
    1. load the 26 char handwriting from one subject
    2. for each char in the word to test, randomly pick one sequence of that char
    3. forward each resampled char sequence, OBSERVE rnn predict correctly mostly
    4. vstack all char sequence
    5. run split and traj search, 
    6. OBSERVE the resultant traj_dict[0] contains top trajectories
        MATCHING the expected word and does not deviate

preliminary conclusion:
    1. rnn forward is very accurate
    2. when rnn forward is accurate and confident, trajectory search produces
        MATCHING result, or result that CAN BE AUTOCORRECTED
    3. when rnn forward is not confident, or when word is too long for small split
        num, traj search breaks and cannot be recovered by autocorrect.

next step:
    1. improve rnn confidence by using upper case letters
    2. re-train model with handwriting from only us three.
    3. investigate why the current test set breaks, it is possible that the 
        handwriting on the current test set is different from the one we trained
'''


if __name__ == "__main__":

    # model = rnn_final.get_net(MODEL_WEIGHT_PATH)
    model = lstm_encdec.get_net(MODEL_WEIGHT_PATH)

    # word_df = data_utils.load_subject('../data_words/kevin_mar3')
    word_df = data_utils.load_subject('../data_words/kevin_tip')
    wordxs, wordys = data_utils.get_calibrated_yprs_samples(
        word_df, resampled=False, flatten=False,
        is_word_samples=True, keep_idx_and_td=True
    )

    for idx in range(len(wordys)):
        x = wordxs[idx]
        y = wordys[idx]

        NUM_PART = len(x) // 12
        print(NUM_PART)
        trajs = word_search.word_search(x, NUM_PART, 10, model)
        print(f'predicting {y}')

        for i, (likelihood, traj) in enumerate(trajs):
            word = ''
            for seg_begin, seg_end, pred, prob in traj:
                word += chr(pred+ord('A'))
            print(f', pred {i}: likelihood {likelihood}', word)

    TARGET_WORDS = [
        'exams',
        'cabin',
        'axe',
        'awe',
        'focus',
        'kanji',
        'honest',
        'longest',
        'something',
        'exfoliation',
        'interesting',
        'international'
    ]

    # char_df = data_utils.load_subject('../data_upper/kevin_tip_char_2')
    # calibration_yprs = data_utils.get_yprs_calibration_vector(char_df)

    # for target_word in TARGET_WORDS:
    #     print('-'*80)
    #     print(f'checking word {target_word}')
    #     print('-'*80)
    #     sample_chs = []
    #     for i, ch in enumerate(target_word):
    #         sample_df = data_utils.get_random_sample_by_label(char_df, ch)
    #         sample_yprs = sample_df[data_utils.YPRS_COLUMNS].to_numpy()
    #         # calibrate
    #         sample_yprs = sample_yprs - calibration_yprs - sample_yprs[0]

    #         # check each char predicted is correct
    #         # resample to test forward
    #         sample_id_col = sample_df[data_utils.ID_COLUMN].to_numpy().reshape(
    #             (-1, 1))
    #         sample_td_col = sample_df[data_utils.TIME_DELTA_COLUMN].to_numpy().reshape(
    #             (-1, 1))
    #         yprs_with_id_td = np.hstack(
    #             (sample_id_col, sample_td_col, sample_yprs))
    #         sample_yprs = data_flatten.resample_sequence(
    #             yprs_with_id_td,
    #             is_flatten_ypr=False,
    #             feature_num=100
    #         )
    #         yhat = lstm_encdec.get_prob(model, torch.tensor([sample_yprs.T]))
    #         pred = chr(97+np.argmax(yhat))
    #         print(f'expect [{ch}] predict [{pred}]')

    #         sample_chs.append(yprs_with_id_td)

    #         # insert noise
    #         num_noise_frame = random.randint(4, 10)
    #         noise_start = random.randint(0, yprs_with_id_td.shape[0])
    #         noise = yprs_with_id_td[
    #             noise_start:noise_start+num_noise_frame, :]
    #         sample_chs.append(noise)

    #     # concat each char sequence to word sequence
    #     x = np.vstack(sample_chs)

    #     NUM_PART = len(x) // 12
    #     trajs = word_search.word_search(x, NUM_PART, 10, model)

    #     for i, (likelihood, traj) in enumerate(trajs):
    #         word = ''
    #         for seg_begin, seg_end, pred, prob in traj:
    #             word += chr(pred+97)
    #         print(f'pred {i}:', word)
