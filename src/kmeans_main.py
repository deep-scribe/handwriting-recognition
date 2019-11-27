import sys
import numpy as np
from kmeans_utils import dtw, l2
import data_utils
import data_visualizer
import data_augmentation
import data_flatten
from data_flatten import load_data_dict_from_file, resample_dataset
from kmeans_core import KMeans
import os


train_dirs = ["../data/kevin_11_7", "../data/russell_11_7"]
test_dirs = ["../data/kelly_11_7"]

root = "../data/"

train_data, train_labels = [], []  # shape should roughly be (40*26, 100, 3)
test_data, test_labels = [], []  # shape should roughly be (20*26, 100, 3)

for dir in train_dirs:
    loaded_dataset = load_data_dict_from_file(dir, calibrate=True)
    flattened_dataset = resample_dataset(
        loaded_dataset, is_flatten_ypr=False, feature_num=100)

    for label_name, data_sequences in flattened_dataset.items():
        train_data.extend(data_sequences)
        train_labels.extend([label_name]*len(data_sequences))

train_data = np.asarray(train_data)
train_labels = np.asarray(train_labels)

print("train_data", train_data.shape)
print("train_labels", train_labels.shape)


for dir in test_dirs:
    loaded_dataset = load_data_dict_from_file(dir, calibrate=True)
    flattened_dataset = resample_dataset(
        loaded_dataset, is_flatten_ypr=False, feature_num=100)

    for label_name, data_sequences in flattened_dataset.items():
        test_data.extend(data_sequences)
        test_labels.extend([label_name]*len(data_sequences))

test_data = np.asarray(test_data)
test_labels = np.asarray(test_labels)

print("test_data", test_data.shape)
print("test_labels", test_labels.shape)

for k in range(26, 52):
    kmeans = KMeans(k, dtw, medoids=True)
    kmeans.fit(train_data, train_labels, verbos=False)
    label_prediction = kmeans.predict_labels(test_data)
    # print(np.mean([kmeans.predict_labels(train_data)[i]==train_labels[i] for i in range(len(train_labels))]))
    acc = np.mean([label_prediction[i] == test_labels[i]
                   for i in range(len(test_labels))])
    print("K: ", k, acc)
