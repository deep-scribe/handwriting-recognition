import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt
import json
import sym_spell

SAVE_PATH = '../output/g_search/'
COLORS = [
    '#8c1515',
    '#003262',
    '#009b76',
    '#b26f16',
]


def main():
    ds = load()  # {K: {acc: [], words: []}}

    acc_plot(ds)
    edit_dist_plot(ds)


def acc_plot(ds):
    plt.clf()
    fig, ax = plt.subplots()
    for i, K in enumerate(reversed(sorted(ds))):
        acc = ds[K]['acc']
        xs = [G for G in acc]
        ys = [acc[G] for G in acc]
        ax.plot(xs, ys, label=f'K={K}', marker='.', color=COLORS[i])
    ax.set(xlabel='G', ylabel='accuracy (higher is better)',
           title='Test Accuracy')
    ax.legend()
    fig.savefig(os.path.join(SAVE_PATH, 'gk_acc.png'))
    plt.clf()


def edit_dist_plot(ds):
    plt.clf()
    fig, ax = plt.subplots()
    for i, K in enumerate(reversed(sorted(ds))):
        words = ds[K]['words']
        xs = [int(G) for G in words]
        ys = [
            sum(sym_spell.editDistance(yhat, y)
                for yhat, y in words[G]) / len(words[G])
            for G in words
        ]
        ax.plot(xs, ys, label=f'K={K}', marker='.', color=COLORS[i])
    ax.set(xlabel='G', ylabel='mean edit distance to label (lower is better)',
           title='Edit Distance')
    ax.legend()
    fig.savefig(os.path.join(SAVE_PATH, 'gk_edit_dist.png'))
    plt.clf()


def load():
    ds = {}
    root, dirs, files = list(os.walk(SAVE_PATH))[0]
    for filename in files:
        if filename.split('.')[-1] == 'json':
            with open(os.path.join(SAVE_PATH, filename)) as f:
                ds[int(filename.split('.')[0].split('_')[-1])] = json.load(f)
    return ds  # {K: {acc: [], words: []}}


if __name__ == "__main__":
    main()
