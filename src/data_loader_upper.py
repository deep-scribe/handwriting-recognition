import pandas as pd
import os
import data_utils
import data_flatten
import random
import data_augmentation
import numpy as np
from sklearn.model_selection import train_test_split

# modify this list to match dirs in ../data that contain
# valid data, one subject per dir
VERIFIED_SUBJECTS = [
    # 'kevin',
    # 'russell',
    'kevin_tip_first',
    'kevin_tip_second',
    'kevin_tip_char_2',
    'kevin_mar3',
    'russell_upper_2',
    'russell_upper_3',
    'russell_upper_4',
    'russell_upper_5',
]

YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]
DATA_PATH = '../data_upper/'


def verified_subjects_calibrated_yprs(resampled=True, flatten=True, keep_idx_and_td=False, subjects=None):
    '''
    load yaw, pitch, roll with verified subjects
    use param subjects to override
    return (xs, ys), lists of all samples correspondingly
    '''
    if subjects == None:
        subjects = VERIFIED_SUBJECTS
    dfs = data_utils.load_all_subjects(DATA_PATH, subjects)

    allxs = []
    allys = []

    for subject in dfs:
        print('Processing', subject)
        df = dfs[subject]
        xs, ys = data_utils.get_calibrated_yprs_samples(
            df, resampled=resampled, flatten=flatten, keep_idx_and_td=keep_idx_and_td)

        allxs.extend(xs)
        allys.extend(ys)

    return allxs, allys


def load_all_classic_random_split(dev_prop, test_prop, resampled=True, flatten=True, keep_idx_and_td=False,):
    xs, ys = verified_subjects_calibrated_yprs(
        resampled=resampled, flatten=flatten, keep_idx_and_td=keep_idx_and_td)
    xs = np.array(xs)
    ys = np.array(ys)
    trainx, devtestx, trainy, devtesty = train_test_split(
        xs, ys, test_size=(dev_prop+test_prop))
    devx, testx, devy, testy = train_test_split(
        devtestx, devtesty, test_size=(test_prop/(test_prop+dev_prop)))

    return trainx, devx, testx, trainy, devy, testy


def load_subject_classic_random_split(dev_prop, test_prop, subjects=None, resampled=True, flatten=True, keep_idx_and_td=False):
    xs, ys = verified_subjects_calibrated_yprs(
        resampled=resampled, flatten=flatten, keep_idx_and_td=keep_idx_and_td, subjects=subjects)
    xs = np.array(xs)
    ys = np.array(ys)
    trainy = []
    print(ys)
    while not (all(x in trainy for x in range(27))):
        trainx, devtestx, trainy, devtesty = train_test_split(
            xs, ys, test_size=(dev_prop+test_prop))
    devx, testx, devy, testy = train_test_split(
        devtestx, devtesty, test_size=(test_prop/(test_prop+dev_prop)))

    return trainx, devx, testx, trainy, devy, testy
