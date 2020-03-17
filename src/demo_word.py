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
import pipeline

DEMO_FILEPATH = 'demo_data/'
PIPELINE = pipeline.Pipeline(use_default_model=False)


class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        re_eval()


def re_eval():
    df = data_utils.load_subject(DEMO_FILEPATH)

    try:
        pred = PIPELINE.predict_realtime(df, G=7, K=10)
        art = str(text2art(pred[0].upper()))
        print('\n' * 100)
        print(art)
    except:
        print("Empty File")


if __name__ == "__main__":
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path=DEMO_FILEPATH, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
