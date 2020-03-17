import data_utils
import data_flatten
import torch
import os
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from art import tprint, text2art
import time

DEMO_FILEPATH = 'demo_data'
PTH_PATH = 'rnn.pth'


class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 26, bias=True)

    def forward(self, x):
        # init_h = torch.randn(self.n_layers, x.shape[0], self.hidden_dim).cuda()
        # init_c = torch.randn(self.n_layers, x.shape[0], self.hidden_dim).cuda()
        init_h = torch.randn(self.n_layers, x.shape[0], self.hidden_dim)
        init_c = torch.randn(self.n_layers, x.shape[0], self.hidden_dim)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, (init_h, init_c))
        out = self.fc(out[:, -1, :])
        return out


MODEL = Net(input_dim=3, hidden_dim=100, n_layers=5)
MODEL.load_state_dict(torch.load(PTH_PATH, map_location=torch.device('cpu')))


class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        re_eval()


def re_eval():
    df = data_utils.load_subject(DEMO_FILEPATH)

    x, y = data_utils.get_calibrated_yprs_samples(
        df, resampled=True, flatten=False)
    if len(x) != 1 or len(y) != 1:
        s = 'Error, try again.'

    try:
        dataloader = torch.utils.data.DataLoader(
            [(x[0].T, y[0])], batch_size=1, shuffle=False
        )

        with torch.no_grad():
            for x, y in dataloader:
                output = MODEL(x.float())
                _, pred = torch.max(output.data, 1)
                char = data_utils.LEGAL_LABELS[pred.data]
                art = text2art(char)
                s = str(art)

        print('\n' * 100)
        print(s)

    except:
        print("Empty File")


if __name__ == "__main__":
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path='demo_data/', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
