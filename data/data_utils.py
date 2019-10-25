import pandas as pd
import os
import string
import random

RAW_COLUMNS = [
    'id', 'td',
    'yaw', 'pitch', 'roll',
    'ax', 'ay', 'az',
    'gx', 'gy', 'gz',
    'mx', 'my', 'mz',
]


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

    # TODO: handle encountering '#' to remove the previous writing sequence

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

    for ch in string.ascii_lowercase:
        filename = f'{ch}.csv'
        chardf = load_one_char_csv(
            os.path.join(subject_path, filename)
        )
        append_identifiers(chardf, label=ch, subject_id=subject_path)
        dfs.append(chardf)

    return pd.concat(dfs, ignore_index=True)


def load_all_subjects(parent_path):
    '''
    load csvs of all subdirs 
    return one single pandas dataframe
    '''
    dfs = []

    for root, dirs, files in os.walk(parent_path):
        for subject_path in dirs:
            dfs.append(load_subject(
                os.path.join(parent_path, subject_path)
            ))

    return pd.concat(dfs, ignore_index=True)


def get_random_sample_by_label(df, label):
    '''
    one random sample sequence matching the label
    return a pd dataframe with frames that share an id
    '''
    rows = df[df['label'] == label]
    ids = list(set(rows['id'].tolist()))
    sample = rows[rows['id'] == random.choice(ids)]
    return sample


# def main():
#     df = load_all_subjects('raw_data')
#     print(df)
#     s = get_sample(df, 'a')
#     print(s)
#     print(s[['ax', 'ay', 'az']].to_numpy())


# if __name__ == "__main__":
#     main()
