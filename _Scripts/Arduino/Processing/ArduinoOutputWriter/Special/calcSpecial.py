import pandas as pd
import os
import string
import random
import numpy as np

RAW_COLUMNS = [
    'id', 'td',
    'yaw', 'pitch', 'roll',
    'ax', 'ay', 'az',
    'gx', 'gy', 'gz',
    'q1', 'q2', 'q3', "q4"
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


df = load_one_char_csv("./Sample_Still.csv")

print(df.shape)

print(np.mean(df.ax))
print(np.mean(df.ay))
print(np.mean(df.az))
print(np.mean(df.q1))
print(np.mean(df.q2))
print(np.mean(df.q3))
print(np.mean(df.q4))
