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


def logit_to_top_k_logit_dict(logit_dict, k):
    '''
    @param logit_dict: {(seg_begin, seg_end): logit ...} where logit is np.array(n_class,)
    @param k: int, num top choices to keep
    @return {(seg_begin, seg_end): [(idx, prob) ...]}
    '''
    return {
        bound: top_k_logit(logit_dict[bound], k) for bound in logit_dict
    }


def top_k_trajectory_from_seg(trajectory_dict, top_k_logit_dict, k, seg_begin, max_seg_bound):
    '''
    populate trajectory_dict with the top k trajectories, starting from seg_begin, 
    ending at max_seg_bound.
    assumes all seg_begin+1 and beyond best trajectories are already populated with DP
    i.e. call this in reverse seg_begin order

    @param trajectory_dict, dict containing all best trajectories between seg_begin+1
        and seg_end, updated to also contain best trajectory between seg_begin and seg_end
    @param top_k_logit_dict: {(seg_begin, seg_end): [(class_idx, prob) ...]}
    @param k: int, num top choices to keep
    @param seg_begin: int, at which point we begin searching for the best trajectory to end
    @param max_seg_bound: int, largest possible segment bound, for example, in a sequence 
        segmented to 3 equal chunks, segment bounds are [0,1,2,3], max_seg_bound is 3
    '''

    if max_seg_bound not in trajectory_dict:
        trajectory_dict[max_seg_bound] = [(None, [])]

    # ensure all the required future segment combination has been already DP'd
    for i in range(seg_begin+1, max_seg_bound):
        assert i in trajectory_dict

    candidates = []  # [(prob, [(seg_begin, seg_end, classidx, prob), ...])]

    for seg_end in range(seg_begin+1, max_seg_bound+1):
        top_k_logit = top_k_logit_dict[(seg_begin, seg_end)]
        for this_class_idx, this_prob in top_k_logit:
            this_trajectory = (seg_begin, seg_end,
                               this_class_idx, this_prob)
            for _, later_trajectory in trajectory_dict[seg_end]:
                combined_trajectory = [this_trajectory] + later_trajectory[:]
                all_prob_list = [this_prob] + \
                    [prob for _, _, _, prob in later_trajectory]
                avg_prob = sum(all_prob_list) / len(all_prob_list)
                candidates.append((avg_prob, combined_trajectory))
    candidates.sort(reverse=True)
    candidates = candidates[:k]
    trajectory_dict[seg_begin] = candidates


def beam_search_top_k_trajectory(logit_dict, k, max_seg_bound):
    '''
    search through logit dict for top k trajectories, return a dictionary of 
    at most top-k trajectories starting at each seg_bound, sorted in descending 
    order of avg_seg_prob for each list of trajectories starting at seg_bound

    @param logit_dict: {(seg_begin, seg_end): logit ...} where logit is np.array(n_class,)
    @param k: int, num top choices to keep
    @param max_seg_bound: int, largest possible segment bound, for example, in a sequence 
        segmented to 3 equal chunks, segment bounds are [0,1,2,3], max_seg_bound is 3
    @return trajectory dict: {
        seg_begin: [
           (avg_seg_prob, 
            [(subseg_begin, subseg_end, subseg_class_idx, subseg_prob), 
             (subseg_begin, subseg_end, subseg_class_idx, subseg_prob), 
             (subseg_begin, subseg_end, subseg_class_idx, subseg_prob)
             ... list continues until subseg_end=max_seg_bound ]
           ),
           (..), 
           (..), 
           ... list of at most k trajectory tuples
        ], 
        ... dict of all possible seg_begin in [0, 1, ..., max_seg_bound]
    }
    '''
    top_k_logit_dict = logit_to_top_k_logit_dict(logit_dict, k)

    trajectory_dict = {}
    for seg_begin in range(max_seg_bound-1, -1, -1):
        top_k_trajectory_from_seg(
            trajectory_dict, top_k_logit_dict,
            k=k, seg_begin=seg_begin, max_seg_bound=max_seg_bound,
        )

    return trajectory_dict


if __name__ == "__main__":
    print('-'*80)
    print('logit_to_top_k_logit_dict')
    print('-'*80)
    logit_dict = {}
    for i in range(4):
        for j in range(i+1, 4):
            logit_dict[(i, j)] = np.array([np.random.rand() for k in range(6)])
    top_k_logit_dict = logit_to_top_k_logit_dict(logit_dict, 4)
    pprint(top_k_logit_dict)

    print('-'*80)
    print('top_k_trajectory_from_seg, empty trajectory, 0-3 segs, seg_begin=2, k=4')
    print('-'*80)
    trajectory_dict = {}
    top_k_trajectory_from_seg(
        trajectory_dict, top_k_logit_dict,
        k=4, seg_begin=2, max_seg_bound=3
    )
    pprint(trajectory_dict)

    print('-'*80)
    print('top_k_trajectory_from_seg, 0-3 segs, seg_begin=1, k=4')
    print('-'*80)
    top_k_trajectory_from_seg(
        trajectory_dict, top_k_logit_dict,
        k=4, seg_begin=1, max_seg_bound=3
    )
    pprint(trajectory_dict)

    print('-'*80)
    print('top_k_trajectory_from_seg, 0-3 segs, seg_begin=0, k=4')
    print('-'*80)
    top_k_trajectory_from_seg(
        trajectory_dict, top_k_logit_dict,
        k=4, seg_begin=0, max_seg_bound=3
    )
    pprint(trajectory_dict)

    print('-'*80)
    print('beam_search_top_k_trajectory, 0-3 segs, all segs, k=4')
    print('-'*80)
    trajectory_dict_2 = beam_search_top_k_trajectory(
        logit_dict, k=4, max_seg_bound=3)
    pprint(trajectory_dict_2)
