import pandas as pd
import os
import data_utils
import data_flatten

VERIFIED_SUBJECTS = [
    'haobin_11_22',
    'kevin_11_7',
    'russell_11_7',
    'kelly_11_7',
    'russell_11_20_stand',
]

YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]
DATA_PATH = '../data/'


def verified_subjects_calibrated_yprs(resampled=False, flatten=False):
    allxs = []
    allys = []

    dfs = data_utils.load_all_subjects(DATA_PATH, VERIFIED_SUBJECTS)

    for subject in dfs:
        print(subject)
        df = dfs[subject]
        xs, ys = data_utils.get_calibrated_yprs_samples(
            df, resampled=resampled, flatten=flatten)

        allxs.extend(xs)
        allys.extend(ys)

    return allxs, allys


if __name__ == "__main__":
    xs, ys = verified_subjects_calibrated_yprs(resampled=True, flatten=False)
    print(xs[0])
    print(xs[0].shape)
