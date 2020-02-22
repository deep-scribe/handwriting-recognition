import numpy as np
from pprint import pprint


def top_k_logit(logit, k):
    '''
    @param logits: np.array, shape (n_class,)
    @param k: int, num top choices to keep
    @return list, [(idx, prob) ...] of top k choices, sorted
    '''
    assert k > 0
    assert k <= logit.shape[0]
    assert len(logit.shape) == 1
    sorted_idx = list(np.argsort(logit))[::-1]
    top_k_idx = sorted_idx[:k]
    return [(idx, logit[idx]) for idx in top_k_idx]


def logit_dict_to_top_k_logit_dict(logit_dict, k):
    '''
    @param logit_dict: {(seg_begin, seg_end): logit ...} where logit is np.array(n_class,)
    @param k: int, num top choices to keep
    @return {(seg_begin, seg_end): [(idx, prob) ...]}
    '''
    return {
        key: top_k_logit(logit_dict[key], k) for key in logit_dict
    }


if __name__ == "__main__":
    print('-'*80)
    print('test logit_dict_to_top_k_logit_dict')
    print('-'*80)
    logit_dict = {}
    for i in range(4):
        for j in range(i+1, 4):
            logit_dict[(i, j)] = np.array([k*.1 for k in range(6)])
    pprint(logit_dict)
    pprint(logit_dict_to_top_k_logit_dict(logit_dict, 2))
