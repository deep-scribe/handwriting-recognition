import beam
import rnn_bilstm
import data_utils
import segmentation
import numpy as np
import torch
import segmentation
import rnn_bilstm
from pprint import pprint


def word_search(x, n, k, model, is_flatten_ypr=False, feature_num=100):
    x_split, bounds = segmentation.split_to_resampled_segments(
        x, n, is_flatten_ypr, feature_num
    )
    x_split = torch.tensor(x_split)
    probs = rnn_bilstm.get_prob(model, x_split)
    logit_dict = {
        bounds[i]: np.array(probs[i]) for i in range(len(bounds))
    }
    trajectory_dict = beam.trajectory_search(logit_dict, k, n)

    return trajectory_dict[0]
