import pipeline
import data_utils
import pandas as pd
import numpy as np
import json
from pprint import pprint

SAVE_PATH = '../output/g_search/acc.json'
G_RANGE = (3, 13)


def main():
    dfs = [data_utils.load_subject(subject) for subject in [
        '../data_words/kevin_30', '../data_words/russell_30']]
    df = pd.concat(dfs)
    # df = data_utils.load_subject('../data_words/small')
    _, ys = data_utils.get_calibrated_yprs_samples(
        df, False, False, is_word_samples=True)
    ys = np.array(ys)

    p = pipeline.Pipeline(use_default_model=False)

    d = {}  # G:acc
    for G in range(*G_RANGE):
        print(f'running G={G}')
        preds = p.predict_realtime(df, G=G, K=10)
        preds = np.array([w.upper() for w in preds])
        acc = np.sum(preds == ys) / len(ys)
        d[G] = acc

    pprint(d)

    with open(SAVE_PATH, 'w+') as f:
        json.dump(d, f)


if __name__ == "__main__":
    main()
