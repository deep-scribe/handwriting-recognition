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

MODEL_WEIGHT_PATH = '../saved_model/cnn/cnn_1.pth'

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

    model = cnn.get_net(MODEL_WEIGHT_PATH)

    # import data_loader_upper
    # import data_loader
    # xs, ys = data_loader_upper.verified_subjects_calibrated_yprs(
    #     flatten=False)
    # xs = torch.tensor(xs)
    # xs = torch.transpose(xs, -1, -2)
    # print(xs.shape)
    # pred = cnn.get_prob(model, xs)
    # pred = np.argmax(pred, axis=1)
    # wrong = np.array(pred) != np.array(ys)
    # print(wrong)
    # print(np.sum(wrong))

    word_df = data_utils.load_subject('../data_words/russell')
    wordxs, wordys = data_utils.get_calibrated_yprs_samples(
        word_df, resampled=False, flatten=False,
        is_word_samples=True, keep_idx_and_td=True
    )

    for idx in range(len(wordys)):
        x = wordxs[idx]
        y = wordys[idx]
        if 'A' in y:
            continue
        NUM_PART = 40
        trajs = word_search.word_search(x, NUM_PART, 10, model)
        print(f'predicting {y}')

        for i, (likelihood, traj) in enumerate(trajs):
            word = ''
            for seg_begin, seg_end, pred, prob in traj:
                word += chr(pred+97)
            print(f', pred {i}:', word)

    TARGET_WORDS = [
        'word',
        'does',
        'not',
        'ei',
        'in',
        'it',
        'focus',
        'beam',
        'kami',
        'kanji',
        'cabin'
    ]

    char_df = data_utils.load_subject('../data_upper/russell')
    calibration_yprs = data_utils.get_yprs_calibration_vector(char_df)

    for target_word in TARGET_WORDS:
        print('-'*80)
        print(f'checking word {target_word}')
        print('-'*80)
        sample_chs = []
        for ch in target_word:
            sample_df = data_utils.get_random_sample_by_label(char_df, ch)
            sample_yprs = sample_df[data_utils.YPRS_COLUMNS].to_numpy()
            # calibrate
            sample_yprs = sample_yprs - calibration_yprs - sample_yprs[0]

            # check each char predicted is correct
            # resample to test forward
            sample_id_col = sample_df[data_utils.ID_COLUMN].to_numpy().reshape(
                (-1, 1))
            sample_td_col = sample_df[data_utils.TIME_DELTA_COLUMN].to_numpy().reshape(
                (-1, 1))
            yprs_with_id_td = np.hstack(
                (sample_id_col, sample_td_col, sample_yprs))
            sample_yprs = data_flatten.resample_sequence(
                yprs_with_id_td,
                is_flatten_ypr=False,
                feature_num=100
            )
            yhat = cnn.get_prob(model, torch.tensor([sample_yprs.T]))
            pred = chr(97+np.argmax(yhat))
            print(f'expect [{ch}] predict [{pred}]')

            sample_chs.append(yprs_with_id_td)

        # concat each char sequence to word sequence
        x = np.vstack(sample_chs)

        NUM_PART = 25
        trajs = word_search.word_search(x, NUM_PART, 10, model)

        for i, (likelihood, traj) in enumerate(trajs):
            word = ''
            for seg_begin, seg_end, pred, prob in traj:
                word += chr(pred+97)
            print(f'pred {i}:', word)
