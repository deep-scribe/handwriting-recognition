import data_utils
import data_flatten
import numpy as np
from pprint import pprint


def split_to_resampled_segments(x, n, is_flatten_ypr=False, feature_num=100):
    '''
    @param x: a sequence, (num_frames, 5), columns are (idx, td, yaw, pitch, roll)
        obtain by data_utils.get_calibrated_yprs_samples(.., flatten=False, keep_idx_and_td=True)
    @param n: number of equal parts to split x into 
    @param is_flatten_ypr: if true, flatten to 1d vector
    @param feature_num: num of resultant feature of each resampled segments

    @return two lists that share index
        [np.array(3, feature_num), or np.array(feature_num) if flatten, ...]
        [(seg_begin, seg_end), ...]
    '''
    nframes, _ = x.shape
    segments = []
    bounds = []

    for seg_begin in range(n):
        frame_begin = int(nframes / n * seg_begin)
        for seg_end in range(seg_begin+1, n+1):
            frame_end = int(nframes / n * seg_end)
            seg_frames = x[frame_begin:frame_end, :]
            resampled_seg_frames = data_flatten.resample_sequence(
                data_sequence=seg_frames,
                is_flatten_ypr=is_flatten_ypr,
                feature_num=feature_num,
                label_name=""
            )
            segments.append(resampled_seg_frames.T)
            bounds.append((seg_begin, seg_end))

    return segments, bounds


if __name__ == "__main__":
    import time
    import paths
    import os
    NUM_PART = 15

    df = data_utils.load_subject(os.path.join(
        paths.DATA_UPPER_WORDS_HEAD, 'words_mini_easy'))
    xs, ys = data_utils.get_calibrated_yprs_samples(
        df=df,
        resampled=False,
        flatten=False,
        is_word_samples=True,
        keep_idx_and_td=True
    )

    begin = time.time() * 1000
    xs_split, bounds = split_to_resampled_segments(xs[0], NUM_PART)
    elapsed = time.time() * 1000 - begin

    print(np.array(xs_split).shape)
