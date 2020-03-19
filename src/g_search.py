import pipeline
import data_utils
import pandas as pd
import numpy as np
import json
from pprint import pprint

SAVE_PATH = '../output/g_search/acc.json'
G_RANGE = (3, 11)


def main():
    xs = []
    ys = []
    for subject in [
        '../data_words/kevin_30',
        '../data_words/russell_30'
    ]:
        xss, yss = data_utils.get_calibrated_yprs_samples(
            data_utils.load_subject(subject),
            resampled=False, flatten=False,
            is_word_samples=True, keep_idx_and_td=True
        )
        xs.extend(xss)
        ys.extend(yss)
    ys = np.array(ys)

    p = pipeline.Pipeline(use_default_model=False)

    d = {}  # G:acc
    w = {G: [] for G in range(*G_RANGE)}  # G:[(yhat, y),...]
    for G in range(*G_RANGE):
        print(f'running G={G}')

        preds = []
        for i, x in enumerate(xs):
            yhat = p.predict_single(x, G, K=10, verbose=False).upper()
            preds.append(yhat)
            w[G].append((yhat, ys[i]))
            print(f'y=[{ys[i]}] yhat=[{yhat}]')

        acc = sum(np.array(preds) == ys) / len(ys)
        print(f'G={G} acc={acc}')
        d[G] = acc

    pprint(d)

    with open(SAVE_PATH, 'w+') as f:
        json.dump(d, f)


if __name__ == "__main__":
    main()
