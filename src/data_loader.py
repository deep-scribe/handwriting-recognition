import pandas as pd
import os
import data_utils
import data_flatten
import random
import data_augmentation
import numpy as np
from sklearn.model_selection import train_test_split

VERIFIED_SUBJECTS = [
    'albert',
    'canon_12_5',
    'daniel',
    'haobin_11_22',
    'isa_12_5',
    'janet',
    'joanne',
    'jq_12_6',
    'kelly_11_7',
    'kevin_11_7',
    'ruocheng',
    'russell_11_20_stand',
    'russell_11_7',
    'russell_random_12_7',
    'solomon',
    'wenzhou_12_5',
    'yiheng_11_30',
    'yiheng_12_5',
    'yongxu_11_30',
]

YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]
DATA_PATH = '../data/'


def verified_subjects_calibrated_yprs(resampled=True, flatten=True, subjects=None):
    if subjects == None:
        subjects = VERIFIED_SUBJECTS
    dfs = data_utils.load_all_subjects(DATA_PATH, subjects)

    allxs = []
    allys = []

    for subject in dfs:
        print('Processing', subject)
        df = dfs[subject]
        xs, ys = data_utils.get_calibrated_yprs_samples(
            df, resampled=resampled, flatten=flatten)

        allxs.extend(xs)
        allys.extend(ys)

    return allxs, allys


# TRAIN_PROP = 0.8
DEV_PROP = 0.1
TRAIN_PROP = 0.1


def load_all_classic_random_split(resampled=True, flatten=True):
    xs, ys = verified_subjects_calibrated_yprs(
        resampled=resampled, flatten=flatten)
    xs = np.array(xs)
    ys = np.array(ys)
    print('Splitting out test set')
    trainx, devtestx, trainy, devtesty = train_test_split(
        xs, ys, test_size=(DEV_PROP+TRAIN_PROP))
    print('Splitting out dev and train set')
    devx, testx, devy, testy = train_test_split(
        devtestx, devtesty, test_size=(TRAIN_PROP/(TRAIN_PROP+DEV_PROP)))

    return trainx, devx, testx, trainy, devy, testy


# use two subjects as test and two subjects as dev
def load_all_subject_split(resampled=True, flatten=True):
    shuffled_subjects = VERIFIED_SUBJECTS[:]
    random.shuffle(shuffled_subjects)
    train_subjects = shuffled_subjects[:-4]
    dev_subjects = shuffled_subjects[-4:-2]
    test_subjects = shuffled_subjects[-2:]
    print('train_subjects', train_subjects)
    print('dev_subjects', dev_subjects)
    print('test_subjects', test_subjects)

    trainx, trainy = verified_subjects_calibrated_yprs(
        resampled=resampled, flatten=flatten, subjects=train_subjects)
    devx, devy = verified_subjects_calibrated_yprs(
        resampled=resampled, flatten=flatten, subjects=dev_subjects)
    testx, testy = verified_subjects_calibrated_yprs(
        resampled=resampled, flatten=flatten, subjects=test_subjects)
    trainx = np.array(trainx)
    devx = np.array(devx)
    testx = np.array(testx)
    trainy = np.array(trainy)
    devy = np.array(devy)
    testy = np.array(testy)

    return trainx, devx, testx, trainy, devy, testy


if __name__ == "__main__":
    # xs, ys = verified_subjects_calibrated_yprs(resampled=True, flatten=False)
    # print(xs[0])
    # print(xs[0].shape)
    trainx, devx, testx, trainy, devy, testy = load_all_classic_random_split()
    print(trainx.shape, devx.shape, testx.shape,
          trainy.shape, devy.shape, testy.shape)

    trainx, devx, testx, trainy, devy, testy = load_all_subject_split()
    print(trainx.shape, devx.shape, testx.shape,
          trainy.shape, devy.shape, testy.shape)
