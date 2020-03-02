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
    sorted_class_idx = list(np.argsort(logit))[::-1]
    top_k_class_idx = sorted_class_idx[:k]
    return [(class_idx, logit[class_idx]) for class_idx in top_k_class_idx]


def top_k_candidates(seg_begin, seg_end, logit, k):
    top_k_logits = top_k_logit(logit, k)
    return [
        (seg_begin, seg_end, class_idx, prob) for class_idx, prob in top_k_logits
    ]


def logit_to_candidates(logit_dict, k):
    '''
    @param logit_dict: {(seg_begin, seg_end): logit ...} where logit is np.array(n_class,)
    @param k: int, num top choices to keep
    @return {(seg_begin, seg_end): [(idx, prob) ...]}
    '''
    return {
        (seg_begin, seg_end):
        top_k_candidates(
            seg_begin, seg_end,
            logit_dict[(seg_begin, seg_end)], k
        )
        for seg_begin, seg_end in logit_dict
    }


def partial_trajectory_search(trajectory_dict, candidate_dict, k, seg_begin, max_seg_bound):
    '''
    populate trajectory_dict with the top k trajectories, starting from seg_begin, 
    ending at max_seg_bound.
    assumes all seg_begin+1 and beyond best trajectories are already populated with DP
    i.e. call this in reverse seg_begin order

    @param trajectory_dict, dict containing all best trajectories between seg_begin+1
        and seg_end, updated to also contain best trajectory between seg_begin and seg_end
    @param candidate_dict: {(seg_begin, seg_end): [(class_idx, prob) ...]}
    @param k: int, num top choices to keep
    @param seg_begin: int, at which point we begin searching for the best trajectory to end
    @param max_seg_bound: int, largest possible segment bound, for example, in a sequence 
        segmented to 3 equal chunks, segment bounds are [0,1,2,3], max_seg_bound is 3
    '''

    # no trajectory needed to go from max_seg_bound to end
    if max_seg_bound not in trajectory_dict:
        trajectory_dict[max_seg_bound] = [(None, [])]

    # ensure all the required future segment combination has been already DP'd
    for i in range(seg_begin+1, max_seg_bound):
        assert i in trajectory_dict

    # [(prob, [(seg_begin, seg_end, classidx, prob), ...])]
    found_trajs = []

    for seg_end in range(
        seg_begin+1,
        max_seg_bound+1
    ):
        candidates = candidate_dict[(seg_begin, seg_end)]

        # for each prediction in segment (seg_begin, seg_end)
        for _, _, this_class_idx, this_prob in candidates:
            this_candidate = (seg_begin, seg_end,
                              this_class_idx, this_prob)
            # for each top trajectory spanning (seg_end, max_seg)
            for _, later_trajectory in trajectory_dict[seg_end]:
                # merge current segment with trajectory leading to end, append to candidate
                combined_trajectory = [this_candidate] + later_trajectory[:]
                all_prob_list = [this_prob] + \
                    [prob for _, _, _, prob in later_trajectory]
                avg_prob = 1
                for p in all_prob_list:
                    avg_prob = avg_prob * p
                found_trajs.append((avg_prob, combined_trajectory))

    # keep the top k candidate for update trajectory_dict
    found_trajs.sort(reverse=True)
    found_trajs = found_trajs[:k]
    trajectory_dict[seg_begin] = found_trajs


def trajectory_search(logit_dict, k, max_seg_bound):
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
    candidate_dict = logit_to_candidates(logit_dict, k)

    trajectory_dict = {}
    for seg_begin in range(max_seg_bound-1, -1, -1):
        partial_trajectory_search(
            trajectory_dict, candidate_dict,
            k=k, seg_begin=seg_begin, max_seg_bound=max_seg_bound,
        )

    return trajectory_dict


if __name__ == "__main__":
    print('-'*80)
    print('logit_to_candidates')
    print('-'*80)
    logit_dict = {}
    for i in range(4):
        for j in range(i+1, 4):
            logit_dict[(i, j)] = np.array([np.random.rand() for k in range(6)])
    candidate_dict = logit_to_candidates(logit_dict, 4)
    pprint(candidate_dict)

    print('-'*80)
    print('partial_trajectory_search, empty trajectory, 0-3 segs, seg_begin=2, k=4')
    print('-'*80)
    trajectory_dict = {}
    partial_trajectory_search(
        trajectory_dict, candidate_dict,
        k=4, seg_begin=2, max_seg_bound=3
    )
    pprint(trajectory_dict)

    print('-'*80)
    print('partial_trajectory_search, 0-3 segs, seg_begin=1, k=4')
    print('-'*80)
    partial_trajectory_search(
        trajectory_dict, candidate_dict,
        k=4, seg_begin=1, max_seg_bound=3
    )
    pprint(trajectory_dict)

    print('-'*80)
    print('partial_trajectory_search, 0-3 segs, seg_begin=0, k=4')
    print('-'*80)
    partial_trajectory_search(
        trajectory_dict, candidate_dict,
        k=4, seg_begin=0, max_seg_bound=3
    )
    pprint(trajectory_dict)

    print('-'*80)
    print('trajectory_search, 0-3 segs, all segs, k=4')
    print('-'*80)
    trajectory_dict_2 = trajectory_search(
        logit_dict, k=4, max_seg_bound=3)
    pprint(trajectory_dict_2)
