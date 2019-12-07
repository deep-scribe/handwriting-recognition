import pandas as pd
import os
import data_utils
import data_flatten
import random
import data_augmentation

VERIFIED_SUBJECTS = [
    'albert',
    'chaonan_12_5',
    'daniel',
    'haobin_11_22',
    'isa_12_5',
    'janet',
    'joanne',
    'kelly_11_7',
    'kevin_11_7',
    'ruocheng',
    'russell_11_20_stand',
    'russell_11_7',
    'solomon',
    'wenzhou_12_5',
    'yiheng_11_30',
    'yongxu_11_30',
]

YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]
DATA_PATH = '../data/'


def verified_subjects_calibrated_yprs(resampled=False, flatten=False, subjects=None):
    if subjects == None:
        subjects = VERIFIED_SUBJECTS
    dfs = data_utils.load_all_subjects(DATA_PATH, subjects)

    allxs = []
    allys = []

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
