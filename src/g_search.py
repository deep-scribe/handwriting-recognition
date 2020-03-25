import pipeline
import data_utils
import pandas as pd
import numpy as np
import json
import os
import paths
from pprint import pprint

G_RANGE = (3, 10)
K = 20
SAVE_PATH = os.path.join(paths.OUTPUT, 'g_search')
FILENAME = f'k_{K}.json'


def search():
    xs = []
    ys = []
    for subject_name in [
        'kevin_30',
        'russell_30'
    ]:
        subject = os.path.join(paths.DATA_UPPER_WORDS_HEAD, subject_name)
        xss, yss = data_utils.get_calibrated_yprs_samples(
            data_utils.load_subject(subject),
            resampled=False, flatten=False,
            is_word_samples=True, keep_idx_and_td=True
        )
        xs.extend(xss)
        ys.extend(yss)
    ys = np.array(ys)

    p = pipeline.Pipeline(use_default_model=False)

    a = {}  # G:acc
    w = {G: [] for G in range(*G_RANGE)}  # G:[(yhat, y),...]
    for G in range(*G_RANGE):
        print(f'running G={G}')

        preds = []
        for i, x in enumerate(xs):
            yhat = p.predict_single(x, G, K=K, verbose=False).upper()
            preds.append(yhat)
            w[G].append((yhat, ys[i]))
            print(f'y=[{ys[i]}] yhat=[{yhat}]')

        acc = sum(np.array(preds) == ys) / len(ys)
        print(f'G={G} acc={acc}')
        a[G] = acc

    pprint(a)

    with open(os.path.join(SAVE_PATH, FILENAME), 'w+') as f:
        json.dump({
            'acc': a,
            'words': w
        }, f)


def plot():
    pass


if __name__ == "__main__":
    search()
