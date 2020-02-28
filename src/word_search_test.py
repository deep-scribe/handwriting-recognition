import beam
import rnn_bilstm
import data_utils
import segmentation
import data_flatten
import numpy as np
import torch
from pprint import pprint

MODEL_WEIGHT_PATH = '../saved_model/rnn_bilstm/rnn_bilstm_random_resampled_0.pth'

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
    TARGET_WORDS = [
        'cat',
        'dog',
        'memory',
        'nexus',
        'fog',
        'entity',
        'word',
        'exfloiate',
        'internationalization'
    ]

    model = rnn_bilstm.get_net(MODEL_WEIGHT_PATH)

    char_df = data_utils.load_subject('../data/russell_11_7')
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
            yhat = rnn_bilstm.get_prob(model, torch.tensor([sample_yprs.T]))
            pred = chr(97+np.argmax(yhat))
            print(f'expect [{ch}] predict [{pred}]')

            sample_chs.append(yprs_with_id_td)

        # concat each char sequence to word sequence
        x = np.vstack(sample_chs)

        NUM_PART = 30
        x_split, bounds = segmentation.split_to_resampled_segments(
            x, NUM_PART
        )
        x_split = torch.tensor(x_split)
        probs = rnn_bilstm.get_prob(model, x_split)
        logit_dict = {
            bounds[i]: np.array(probs[i]) for i in range(len(bounds))
        }
        trajectory_dict = beam.trajectory_search(logit_dict, 5, NUM_PART)

        for i, (likelihood, traj) in enumerate(trajectory_dict[0]):
            word = ''
            for seg_begin, seg_end, pred, prob in traj:
                word += chr(pred+97)
            print(f'pred {i}:', word)
