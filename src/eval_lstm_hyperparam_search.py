import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import lstm
import torch
import json
import os
from pprint import pprint

FILENAME_DESCRIPTION = 'hyperparam_search_1'
MODEL_WEIGHT_PATH = '../saved_model/'
MODEL_HIST_PATH = '../output/'


def main():
    files = get_filepaths()
    populate_devacc(files)
    populate_model_param(files)
    files = sorted(files, key=lambda file: file['devacc'], reverse=True)
    files = files[:50]
    pprint(files[0])
    show_plot(files)


def show_plot(files):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlabel('lstm_hidden_dim')
    ax.set_ylabel('lstm_n_layers')
    ax.set_zlabel('fc_hidden_dim')
    ax.scatter3D(
        [file['lstm_hidden_dim'] for file in files],
        [file['lstm_n_layers'] for file in files],
        [file['fc_hidden_dim'] for file in files],
        c=[file['devacc'] for file in files],
        cmap='Reds',
        depthshade=0,
    )
    plt.show()


def populate_model_param(files):
    for file in files:
        _, _, paramstrs, _, _ = file['prefix'].split('.')
        _, lstm_hidden_dim, lstm_n_layers, fc_hidden_dim, _, _, _ = paramstrs.split(
            '-')
        file['lstm_hidden_dim'] = int(lstm_hidden_dim)
        file['lstm_n_layers'] = int(lstm_n_layers)
        file['fc_hidden_dim'] = int(fc_hidden_dim)


def populate_devacc(files):
    for file in files:
        json_path = file['json_path']
        with open(json_path) as f:
            j = json.load(f)
            # print()
            file['devacc'] = max(j['devacc'])


def get_filepaths():
    l = []
    root, dirs, files = list(os.walk(MODEL_WEIGHT_PATH))[0]
    for name in files:
        if FILENAME_DESCRIPTION in name:
            if name.split('.')[-1] == 'pth':
                no_ext = '.'.join(name.split('.')[:-1])
                json_filename = no_ext+'.json'
                json_path = os.path.join(MODEL_HIST_PATH, json_filename)
                pth_path = os.path.join(MODEL_HIST_PATH, name)
                print(json_filename)
                if os.path.exists(json_path):
                    l.append({
                        'prefix': no_ext,
                        'json_path': json_path,
                        'pth_path': pth_path
                    })
    return l


if __name__ == "__main__":
    main()
