from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split

import csv
import numpy as np
import numpy

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

# true label index
A_TRUE_LABEL = 0
B_TRUE_LABEL = 1
C_TRUE_LABEL = 2
D_TRUE_LABEL = 3
E_TRUE_LABEL = 4

RANDOM_STATE = 42
TEST_RATE = 0.2

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):
    def __init__(self,
                 motionData,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=False,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid motionData dtype %r, expected uint8 or float32' %
                            dtype)

        assert motionData.shape[0] == labels.shape[0], (
            'motionData.shape: %s labels.shape: %s' % (motionData.shape, labels.shape))
        self._num_examples = motionData.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            motionData = motionData.astype(numpy.float32)
            motionData = numpy.multiply(motionData, 1.0 / 255.0)

        self._motionData = motionData
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def motionData(self):
        return self._motionData

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._motionData = self.motionData[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            motionData_rest_part = self._motionData[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._motionData = self.motionData[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            motionData_new_part = self._motionData[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((motionData_rest_part, motionData_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._motionData[start:end], self._labels[start:end]


def cluster_selection_label(letter_a_set, letter_b_set, letter_c_set, letter_d_set, letter_e_set):
    # return np.array(letter_a_set + letter_b_set)
    return np.array(letter_a_set + letter_b_set + letter_c_set + letter_d_set + letter_e_set)


def cluster_selection_data(letter_a_set, letter_b_set, letter_c_set, letter_d_set, letter_e_set):
    return np.concatenate((letter_a_set, letter_b_set, letter_c_set, letter_d_set, letter_e_set), axis=0)
    # return np.concatenate((letter_a_set, letter_b_set), axis=0)


def read_format_input(read_file_name):
    with open(read_file_name, 'rb') as f:
        reader = csv.reader(f)
        raw_data_list = list(reader)
    return raw_data_list


def read_data_sets(train_dir=None,
                   fake_data=False,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url="NO_URL"):
    # raw data file names
    DATA_SET_A = 'run_letter_a_format.csv'
    DATA_SET_B = 'run_letter_b_format.csv'
    DATA_SET_C = 'run_letter_c_format.csv'
    DATA_SET_D = 'run_letter_d_format.csv'
    DATA_SET_E = 'run_letter_e_format.csv'

    raw_list_a = read_format_input(DATA_SET_A)
    format_list_a = np.array(raw_list_a).astype(None)
    y_data_set_a = [A_TRUE_LABEL] * format_list_a.shape[0]

    raw_list_b = read_format_input(DATA_SET_B)
    format_list_b = np.array(raw_list_b).astype(None)
    y_data_set_b = [B_TRUE_LABEL] * format_list_b.shape[0]

    raw_list_c = read_format_input(DATA_SET_C)
    format_list_c = np.array(raw_list_c).astype(None)
    y_data_set_c = [C_TRUE_LABEL] * format_list_c.shape[0]

    raw_list_d = read_format_input(DATA_SET_D)
    format_list_d = np.array(raw_list_d).astype(None)
    y_data_set_d = [D_TRUE_LABEL] * format_list_d.shape[0]

    raw_list_e = read_format_input(DATA_SET_E)
    format_list_e = np.array(raw_list_e).astype(None)
    y_data_set_e = [E_TRUE_LABEL] * format_list_e.shape[0]

    x_data_set = cluster_selection_data(format_list_a, format_list_b, format_list_c, format_list_d, format_list_e)
    y_data_set = cluster_selection_label(y_data_set_a, y_data_set_b, y_data_set_c, y_data_set_d, y_data_set_e)

    # apply one-hot encoding:
    y_data_set = dense_to_one_hot(y_data_set, num_classes=5)

    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set,
                                                        test_size=TEST_RATE, random_state=RANDOM_STATE)

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(x_train, y_train, **options)
    validation = DataSet(x_train, y_train, **options)
    test = DataSet(x_test, y_test, **options)

    return base.Datasets(train=train, validation=validation, test=test);
