import pandas as pd
import os
import string
import random
import numpy as np
import data_flatten

RAW_COLUMNS = [
    'id', 'td',
    'yaw', 'pitch', 'roll',
    'ax', 'ay', 'az',
    'gx', 'gy', 'gz',
    'qx', 'qy', 'qz', 'qw'
]

CALIBRATION_LABEL_NAME = 'calibration'
CALIBRATION_FILENAME = CALIBRATION_LABEL_NAME + '.csv'

ID_COLUMN = RAW_COLUMNS[0]
TIME_DELTA_COLUMN = RAW_COLUMNS[1]
YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]

LEGAL_LABELS = 'abcdefghijklmnopqrstuvwxyz'
LABEL_TO_INDEX = {
    LEGAL_LABELS[i]: i for i in range(len(LEGAL_LABELS))
}


def load_one_char_csv(filename):
    '''
    loads a {char}.csv
    return a pandas dataframe
    '''
    df = pd.read_csv(
        filename,
        header=None,
        names=RAW_COLUMNS
    )

    # handle encountering '#' to remove the previous writing sequence
    ids_to_remove = set()
    for row_number_of_pound in list(df[df.astype(str)['id'] == '#'].index):
        if row_number_of_pound == 0:
            continue
        row_number_to_remove = row_number_of_pound - 1
        ids_to_remove.add(df.iloc[row_number_to_remove]['id'])

    for i in df.id.unique():
        if len(df[df['id'] == i]) < 25:
            ids_to_remove.add(i)

    for id_to_remove in ids_to_remove:
        df = df[df['id'] != id_to_remove]

    return df


def append_identifiers(df, label, subject_id):
    '''
    Append a columns to the dataframe `df`
    df: a dataframe to modify
    label: str label
    subject_id: str subject identifier
    '''
    df['subject_id'] = subject_id
    df['label'] = label


def load_subject(subject_path):
    '''
    load all {char}.csv from a directory, append label,
    return a merged pd dataframe
    '''

    dfs = []

    for root, dirs, files in os.walk(subject_path):
        for filename in files:
            if '.csv' not in filename:
                continue
            label = filename.replace('.csv', '')
            chardf = load_one_char_csv(
                os.path.join(subject_path, filename)
            )
            append_identifiers(chardf, label=label, subject_id=subject_path)
            dfs.append(chardf)

    return pd.concat(dfs, ignore_index=True)


def load_all_subjects(parent_path, subject_paths):
    '''
    load csvs of all subdirs
    return one single pandas dataframe
    '''
    dfs = {}

    for subject_path in subject_paths:
        # print(subject_path)
        dfs[subject_path] = load_subject(
            os.path.join(parent_path, subject_path)
        )

    return dfs


def get_random_sample_by_label(df, label):
    '''
    one random sample sequence matching the label
    return a pd dataframe with frames that share an id
    '''
    rows = df[df['label'] == label]
    ids = list(set(rows['id'].tolist()))
    subjects = list(set(rows['subject_id'].tolist()))
    sample = rows[
        rows['id'] == random.choice(ids)
    ]
    sample = sample[
        sample['subject_id'] == random.choice(subjects)
    ]
    return sample


def get_all_samples_by_label(df, label):
    '''
    given by the label
    return a dictionary of sample_id: sample pd dataframes
    '''
    rows = df[df['label'] == label]
    ids = list(set(rows['id'].tolist()))
    samples = {}
    for i in ids:
        samples[i] = rows[rows['id'] == i]
    return samples


def get_yprs_calibration_vector(df):
    '''
    given a df of a subject (i.e. df returned by load_subject())
    return a vector of the mean of all calibration rows of yprs cols
    '''
    calibrationdf = df[df['label'] == CALIBRATION_LABEL_NAME]
    if calibrationdf.empty:
        print(
            f'[WARN] data_utils.get_yprs_calibration_vector: no calibration data in df, returning [0,0,0]'
        )
        return np.zeros(3)

    calibrationyprs = calibrationdf[YPRS_COLUMNS].to_numpy()
    calibrationyprs = np.mean(calibrationyprs, axis=0)
    return calibrationyprs


def get_calibrated_yprs_samples(df, resampled, flatten):
    '''
    given a df of a subject (i.e. df returned by load_subject())
    return (xs, ys)
    where xs is a list of np.array((num_frames, 3)) of the yprs of the sequence
    where ys is a list of int of labels (0-25)
    '''

    xs = []
    ys = []

    calibrationyprs = get_yprs_calibration_vector(df)

    for label_idx in range(len(LEGAL_LABELS)):
        label_ch = LEGAL_LABELS[label_idx]

        samples = get_all_samples_by_label(df, label_ch)
        for sample_id in samples:
            sampledf = samples[sample_id]
            sample_yprs = sampledf[YPRS_COLUMNS].to_numpy()
            # calibrate
            sample_yprs = sample_yprs - calibrationyprs - sample_yprs[0]

            if resampled:
                sample_id_col = sampledf[ID_COLUMN].to_numpy().reshape((-1, 1))
                sample_td_col = sampledf[TIME_DELTA_COLUMN].to_numpy().reshape(
                    (-1, 1))
                sequence = np.hstack(
                    (sample_id_col, sample_td_col, sample_yprs))

                if sequence.shape[0] <= 2:
                    continue

                sample_yprs = data_flatten.resample_sequence(
                    sequence, is_flatten_ypr=flatten
                )

            xs.append(sample_yprs)
            ys.append(label_idx)

    return xs, ys


def main():
    df = load_subject('../data/kelly_11_7')
    print(df)
    s = get_random_sample_by_label(df, 'a')
    print(s)
    print(s[['ax', 'ay', 'az']].to_numpy())


if __name__ == "__main__":
    main()
