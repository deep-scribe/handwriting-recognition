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
import lstm
import lstm_siamese

DEMO_FILEPATH = 'demo_data'
PTH_PATH = '../saved_model/da/'
# PTH_FILENAME = 'LSTM_char_classifier.best.3-275-8-88-27-0-1.1500-3-3.03-15-09-25.pth'
# PTH_FILENAME = 'LSTM_char_classifier.rus_kev_upper.3-200-3-200-27-0-1.1500-3-3.03-11-03-57.pth'
# PTH_FILENAME = 'LSTM_char_classifier.transfer.3-200-3-200-27-0-1-1500-1.3.03-15-19-04.pth'
PTH_FILENAME = 'LSTM_char_classifier.rus_kev_upper.3-200-3-200-27-0-1.1500-1-3.03-13-02-20.pth'
# MODEL_CONFIG = (3, 275, 8, 88, 27, False, True)
MODEL_CONFIG = (3, 200, 3, 200, 27, False, True)


# MODEL = lstm.LSTM_char_classifier(*MODEL_CONFIG)
MODEL = lstm_siamese.LSTM_char_classifier(*MODEL_CONFIG)
MODEL.load_state_dict(torch.load(
    os.path.join(PTH_PATH, PTH_FILENAME),
    map_location=torch.device('cpu')))


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
                output, _ = MODEL(x.float())
                _, pred = torch.max(output.data, 1)
                char = data_utils.LEGAL_LABELS[pred.data]
                art = text2art(char.upper())
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
