# import third-party packages
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split

import numpy as np
import logging
import csv
import datetime
import time
import random
import sys


# constant for logger
CURRENT_TIMESTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
RESULT_OUTPUT = 'run_svm_result_' + CURRENT_TIMESTAMP + '.log'
LOGGER_FORMAT_HEADER = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
CUTOFF_LINE = '--------------------------------------------------------------------------------------------------'
IS_ENABLE_FILE_LOGGING = False

# general constant
SPLIT_RANDOM_STATE = 42
TEST_SIZE = 0.25

# constant for rnn training
learning_rate = 0.001
training_steps = 1000 # TODO: change the steps to 10000 for better result
batch_size = 128
display_step = 50

# constant rnn network parameters
num_input = 3  # we only read one set of yaw pitch row
timesteps = 100  # timesteps - we have 100 data point for each char
num_hidden = 128  # hidden layer num of features
num_classes = 5  # number of data class - using abc for now

# raw data file names
DATA_SET_A = 'run_letter_a_format.csv'
DATA_SET_B = 'run_letter_b_format.csv'
DATA_SET_C = 'run_letter_c_format.csv'
DATA_SET_D = 'run_letter_d_format.csv'
DATA_SET_E = 'run_letter_e_format.csv'

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# bookkeeping logic for setup logger and random sample generator
LOGGER = logging.getLogger('cogs181_runtime')
LOGGER.setLevel(logging.DEBUG)
formatter = logging.Formatter(LOGGER_FORMAT_HEADER)

# setup file logging if user enable
if IS_ENABLE_FILE_LOGGING:
    fh = logging.FileHandler(RESULT_OUTPUT)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

random.seed(SPLIT_RANDOM_STATE)


def read_format_input(read_file_name):
    with open(read_file_name, 'r') as f:
        reader = csv.reader(f)
        raw_data_list = list(reader)

    return raw_data_list


def render_raw_data():
    raw_a_x = np.array(read_format_input(DATA_SET_A)).astype(None)
    raw_b_x = np.array(read_format_input(DATA_SET_B)).astype(None)
    raw_c_x = np.array(read_format_input(DATA_SET_C)).astype(None)
    raw_d_x = np.array(read_format_input(DATA_SET_D)).astype(None)
    raw_e_x = np.array(read_format_input(DATA_SET_E)).astype(None)
    raw_x = np.concatenate((raw_a_x, raw_b_x, raw_c_x, raw_d_x, raw_e_x), axis=0)

    raw_a_y = np.array([[1, 0, 0, 0, 0]] * len(raw_a_x)).astype(None)
    raw_b_y = np.array([[0, 1, 0, 0, 0]] * len(raw_b_x)).astype(None)
    raw_c_y = np.array([[0, 0, 1, 0, 0]] * len(raw_c_x)).astype(None)
    raw_d_y = np.array([[0, 0, 0, 1, 0]] * len(raw_d_x)).astype(None)
    raw_e_y = np.array([[0, 0, 0, 0, 1]] * len(raw_e_x)).astype(None)
    raw_y = np.concatenate((raw_a_y, raw_b_y, raw_c_y, raw_d_y, raw_e_y), axis=0)

    train_x, test_x, train_y, test_y = \
        train_test_split(raw_x, raw_y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE)

    return train_x, train_y, test_x, test_y


# TODO: make render_batch generate next set of unique batch instead of repeat some same index
def render_batch(batch_size, x_data, y_data):
    if len(x_data) != len(y_data):
        sys.exit("Error: cannot render batch with different len of x and y.")

    batch_index = random.sample(range(0, len(x_data)), batch_size)
    render_x = []
    render_y = []

    for index in batch_index:
        render_x.append(x_data[index])
        render_y.append(y_data[index])

    render_x = np.array(render_x).astype(None)
    render_y = np.array(render_y).astype(None)
    return render_x, render_y


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def training_engine(train_x, train_y, test_x, test_y):
    logits = RNN(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # assign vars to default value
    init = tf.global_variables_initializer()

    # start tf training
    with tf.Session() as sess:

        # run the initializer
        sess.run(init)

        for step in range(1, training_steps + 1):
            batch_x, batch_y = render_batch(batch_size, train_x, train_y)
            # reshape data to get 100 seq of 3 elements (y,p,r)
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # run optimization operation by using backprop
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:

                # calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                training_step_log = ("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
                LOGGER.debug(training_step_log)

        LOGGER.debug("Optimization Finished!")

        # calculate accuracy on testing set
        test_data = test_x.reshape((-1, timesteps, num_input))
        test_label = test_y
        test_acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
        test_log = "Testing Accuracy:" + str(test_acc)
        LOGGER.debug(test_log)


def main():
    LOGGER.debug("Start reading formatted input")
    train_x, train_y, test_x, test_y = render_raw_data()
    LOGGER.debug("End reading formatted input")

    training_engine(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
