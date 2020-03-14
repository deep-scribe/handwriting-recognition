import beam
# import rnn_bilstm
import data_utils
import segmentation
import numpy as np
import torch
import segmentation
import math
# import rnn_final
from pprint import pprint


def get_prob(net, input):
    if torch.cuda.is_available():
        input = input.cuda()
    else:
        net.cpu()
    net.eval()
    with torch.no_grad():
        logit = net(input.float())
        prob = F.log_softmax(logit, dim=-1)
    return logit

def word_search(x, g, k, model, is_flatten_ypr=False, feature_num=100):
    '''
    @param x: the sample
    @param g: granularity, how many parts to split per expected letter
    @oaran k: keep top k choices
    '''
    n = math.ceil(len(x) / 75) * g

    x_split, bounds = segmentation.split_to_resampled_segments(
        x, n, is_flatten_ypr, feature_num
    )
    x_split = torch.tensor(x_split)
    probs = get_prob(model, x_split)
    logit_dict = {
        bounds[i]: np.array(probs[i]) for i in range(len(bounds))
    }
    trajectory_dict = beam.trajectory_search(logit_dict, k, n)

    return trajectory_dict[0]
