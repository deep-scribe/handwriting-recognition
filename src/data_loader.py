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

DATA_PATH = '../data/'


def load_verified_subjcects():
    df = data_utils.load_all_subjects(DATA_PATH, VERIFIED_SUBJECTS)
    return df


if __name__ == "__main__":
    df = load_verified_subjcects()
    print(data_utils.get_random_sample_by_label(df, 'a'))
