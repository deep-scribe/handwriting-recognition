import beam
import rnn_bilstm
import data_utils
import segmentation
import numpy as np
import torch
from pprint import pprint

MODEL_WEIGHT_PATH = '../saved_model/rnn_bilstm/rnn_bilstm_random_resampled_0.pth'

if __name__ == "__main__":
    NUM_PART = 15

    df = data_utils.load_subject('../data_words/words_mini_easy_2')
    xs, ys = data_utils.get_calibrated_yprs_samples(
        df=df,
        resampled=False,
        flatten=False,
        is_word_samples=True,
        keep_idx_and_td=True
    )
    xs_split, bounds = segmentation.split_to_resampled_segments(
        xs[1], NUM_PART
    )
    xs_split = torch.tensor(xs_split)

    model = rnn_bilstm.get_net(MODEL_WEIGHT_PATH)
    probs = rnn_bilstm.get_prob(model, xs_split)

    logit_dict = {
        bounds[i]: np.array(probs[i]) for i in range(len(bounds))
    }
    trajectory_dict = beam.trajectory_search(logit_dict, 10, NUM_PART)

    print(ys[1])
    for likelihood, traj in trajectory_dict[0]:
        word = ''
        for seg_begin, seg_end, pred, prob in traj:
            word += chr(pred+97)
        print('pred:', word)
