import sys
import numpy as np
from kmeans_utils import dtw, l2
import data_utils
import data_visualizer
import data_augmentation
import data_flatten
from data_flatten import load_data_dict_from_file, resample_dataset
from kmeans_core import KMeans


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def shuffle(all_dirs, train_split=0.8):
    data, labels = [], []

    for dir in all_dirs:
        loaded_dataset = load_data_dict_from_file(dir, calibrate=True)
        flattened_dataset = resample_dataset(
            loaded_dataset, is_flatten_ypr=True, feature_num=100)

        for label_name, data_sequences in flattened_dataset.items():
            data.extend(data_sequences)
            labels.extend([label_name]*len(data_sequences))

    data = np.asarray(data)
    labels = np.asarray(labels)

    data, labels = unison_shuffled_copies(data, labels)
    cutoff_index = int(len(data) * train_split)

    train_data = data[:cutoff_index]
    train_labels = labels[:cutoff_index]
    test_data = data[cutoff_index:]
    test_labels = labels[cutoff_index:]

    return train_data, train_labels, test_data, test_labels


train_dirs = ["../data/kevin_11_7", "../data/russell_11_7"]
test_dirs = ["../data/kelly_11_7"]
all_dirs = train_dirs + test_dirs

train_data, train_labels, test_data, test_labels = shuffle(all_dirs)

print("train data:", train_data.shape)
print("train labels:", train_labels.shape)
print("test data:", test_data.shape)
print("test labels:", test_labels.shape)

for k in range(26, 52):
    kmeans = KMeans(k, l2, medoids=False)
    kmeans.fit(train_data, train_labels, verbos=False)
    label_prediction = kmeans.predict_labels(test_data)
    # print(np.mean([kmeans.predict_labels(train_data)[i]==train_labels[i] for i in range(len(train_labels))]))
    acc = np.mean([label_prediction[i] == test_labels[i]
                   for i in range(len(test_labels))])
    print("K: ", k, acc)
