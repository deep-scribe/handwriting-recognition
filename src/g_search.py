import pipeline
import data_utils
import pandas as pd
import numpy as np
import json
from pprint import pprint

SAVE_PATH = '../output/g_search/acc.json'
G_RANGE = (7, 11)


def main():
    dfs = [data_utils.load_subject(subject) for subject in [
        '../data_words/kevin_30',
        # '../data_words/russell_30'
    ]]
    df = pd.concat(dfs)
    # df = data_utils.load_subject('../data_words/small')
    xs, ys = data_utils.get_calibrated_yprs_samples(
        df,
        resampled=False, flatten=False,
        is_word_samples=True, keep_idx_and_td=True
    )
    ys = np.array(ys)

    p = pipeline.Pipeline(use_default_model=False)

    d = {}  # G:acc
    for G in range(*G_RANGE):
        print(f'running G={G}')

        preds = []
        for i, x in enumerate(xs):
            yhat = p.predict_single(x, G, K=10, verbose=False).upper()
            preds.append(yhat)
            print(f'y=[{ys[i]}] yhat=[{yhat}]')

        acc = sum(np.array(preds) == ys) / len(ys)
        print(f'G={G} acc={acc}')
        d[G] = acc

    pprint(d)

    with open(SAVE_PATH, 'w+') as f:
        json.dump(d, f)


if __name__ == "__main__":
    main()
