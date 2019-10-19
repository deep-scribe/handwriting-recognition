# ------------------------------------------------------------------
#
# CNN Testing
#
# ------------------------------------------------------------------


from __future__ import print_function

import os
import csv

# Set GPU number.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import MNIST data
import input_data

fullsets = input_data.read_data_sets()

import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, shape=[None,300])
y_ = tf.placeholder(tf.float32, shape=[None,5])


# Handy functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution and Pooling:
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME', use_cudnn_on_gpu=True)

# reduces array dimension by dividing 2
def max_pool_1x1(x):
    return tf.nn.pool(x, window_shape=[2],
                          pooling_type="MAX", padding='SAME', strides=[2])

# reduce reduce no dimension
def max_pool_1x1_m5(x):
    return tf.nn.pool(x, window_shape=[15],
                            pooling_type="MAX", padding='SAME', strides=[1])

def max_pool_1x1_d5(x):
    return tf.nn.pool(x, window_shape=[5],
                            pooling_type="MAX", padding='SAME', strides=[5])

#*********************** First Convolution Layer *************************************
# filter / kernel tensor of shape [filter_width, in_channels, out_channels]
W_conv1 = weight_variable([60, 1, 32])
b_conv1 = bias_variable([32])

# input tensor of shape [batch, in_width, in_channels] if data_format is "NHWC"
x_motionData = tf.reshape(x, [-1, 300, 1])

# apply elu function and max pool
h_conv1 = tf.nn.elu(conv1d(x_motionData, W_conv1) + b_conv1)
h_pool1 = max_pool_1x1(h_conv1)

#*********************** Second Convolution Layer ************************************
W_conv2 = weight_variable([60, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.elu(conv1d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_1x1(h_conv2)

#*********************** Third Convolution Layer ************************************
W_conv3 = weight_variable([60, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.elu(conv1d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_1x1_m5(h_conv3)

#*********************** Fourth Convolution Layer ************************************
W_conv4 = weight_variable([60, 64, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.elu(conv1d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_1x1_m5(h_conv4)

#*********************** Fifth Convolution Layer ************************************
W_conv5 = weight_variable([60, 64, 128])
b_conv5 = bias_variable([128])

h_conv5 = tf.nn.elu(conv1d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_1x1_d5(h_conv5)


# Densely Connected Layer with 1024 neurons: (300/2/2 = 75 / 5 = 15)
W_fc1 = weight_variable([15 * 128, 64*64])
b_fc1 = bias_variable([64*64])

h_pool5_flat = tf.reshape(h_pool5, [-1, 15 * 128])
h_fc1 = tf.nn.elu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

# Dropout before the readout layer to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer:
W_fc2 = weight_variable([64*64, 5])
b_fc2 = bias_variable([5])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# apply regularization
regularizer = tf.nn.l2_loss(W_fc2)
cross_entropy = tf.reduce_mean(cross_entropy + 0.01 * regularizer)

adam_optimizer = tf.train.AdamOptimizer(1e-4)

train_step = adam_optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# for plotting
iteration_list = []
train_loss_list = []
test_accuracy_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = fullsets.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy));
            print("current learning rate is: " + str(adam_optimizer._lr))

            train_loss_list.append(
                cross_entropy.eval(feed_dict={x: fullsets.test.motionData, y_: fullsets.test.labels, keep_prob: 1.0}));

            test_accuracy_list.append(
                accuracy.eval(feed_dict={x: fullsets.test.motionData, y_: fullsets.test.labels, keep_prob: 1.0}));

            iteration_list.append(i);

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: fullsets.test.motionData, y_: fullsets.test.labels, keep_prob: 1.0}))

    plt.ylabel('training loss')
    plt.xlabel('num of trainig epoch/iteration')
    plt.title('ConvNet Training on fullsets')
    plt.plot(iteration_list[1:], train_loss_list[1:], color='r', label='train loss')
    plt.legend(loc='upper right')
    plt.savefig('./Problem2_figure_Training_Loss.png')

    plt.gcf().clear()

    plt.ylabel('test accuracy')
    plt.xlabel('num of trainig epoch/iteration')
    plt.title('ConvNet Training on fullsets')
    plt.plot(iteration_list[1:], test_accuracy_list[1:], color='y', label='test accuracy')
    plt.legend(loc='lower right')
    plt.savefig('./Problem2_figure_Test_Acc.png')

    myData = []
    myData.append(iteration_list)
    myData.append(train_loss_list)
    myData.append(test_accuracy_list)
    myFile = open('5_Layer_MAX_elu.csv','w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(myData)
