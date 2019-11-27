import pandas as pd
import os
import data_utils

VERIFIED_SUBJECTS = [
    'haobin_11_22',
    'kevin_11_7',
    'russell_11_7',
    'kelly_11_7',
    'russell_11_20_stand',
]

YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]
DATA_PATH = '../data/'


def load_verified_subjects_calibrated_yprs():
    allxs = []
    allys = []

    dfs = data_utils.load_all_subjects(DATA_PATH, VERIFIED_SUBJECTS)

    for subject in dfs:
        print(subject)
        df = dfs[subject]
        xs, ys = data_utils.get_calibrated_yprs_samples(df)

        allxs.extend(xs)
        allys.extend(ys)

    return allxs, allys


if __name__ == "__main__":
    xs, ys = load_verified_subjects_calibrated_yprs()
    print(xs[0])
    print(ys)
    print(len(xs))
    print(len(ys))
