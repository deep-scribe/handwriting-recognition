import numpy as np
from utils import dtw, l2
from data import data_utils, data_visualizer, data_augmentation, data_flatten

train_dirs = ["kevin_11_7", "russell_11_7"]
test_dirs = ["kelly_11_7"]

root = "./data/"

train_dict = {}
for dir in train_dirs:
    loaded_dataset = load_data_dict_from_file(dir, calibrate=True)
    flattened_dataset = resample_dataset(loaded_dataset, is_flatten_ypr=True, feature_num=100)
    for label_name, data_sequences in flattened_dataset.items():
        
