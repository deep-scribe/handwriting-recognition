import data_loader
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import os
import sys
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

import data_flatten

import warnings
warnings.filterwarnings('ignore')


'''
autoencoder to smoothen data
given a list of training samples, train one model for each of yaw, pitch, roll
use that model to pre-process all training / testing data
'''

# change these param for input data, make sure CODE_SIZE < FEATURE_NUM
FEATURE_NUM = 100
HIDDEN_SIZE = 128
CODE_SIZE = 64

# if you want to use logistic regression
# AUTOENCODER_LOSS ='binary_crossentropy'
# if you want to use linear regression
AUTOENCODER_LOSS = 'mean_squared_error'


def restore_ypr(ypr_sequence):
    '''
    inverse function to normalize_ypr
    '''
    ypr_max = 180.00
    ypr_min = -179.99

    ypr_sequence *= (ypr_max - ypr_min)
    ypr_sequence += ypr_min

    return ypr_sequence


def normalize_ypr(ypr_sequence):
    '''
    normalize the values to be in range(0-1)
    '''
    # print (np.max(ypr_sequence), np.min(ypr_sequence))

    ypr_max = 180.00
    ypr_min = -179.99

    ypr_sequence -= ypr_min
    ypr_sequence /= (ypr_max - ypr_min)

    # print (np.max(ypr_sequence), np.min(ypr_sequence))
    # print ()

    return ypr_sequence


def compare_dataset(d1, d2, verbose=False):
    for key in d1:
        if not np.array_equal(d1[key], d2[key]):
            if verbose:
                print("ERROR: the two dataset has different elements or shape")
            return False
    return True


def load_data(subject_path, is_flatten_ypr=False, feature_num=300, preprocess=True):
    '''
    If proprocess = True, then call data_flatten
    '''
    loaded_dataset = data_flatten.load_data_dict_from_file(
        subject_path, calibrate=True, verbose=False)

    if preprocess:
        loaded_dataset = data_flatten.resample_dataset(
            loaded_dataset, is_flatten_ypr=is_flatten_ypr, feature_num=feature_num)

    return loaded_dataset


def restore_ypr_data(yaws, pitchs, rolls, labels):
    '''
    Invert function of separate ypr data
    '''

    restored_dataset = collections.defaultdict(list)

    i = 0

    for key in labels:
        merged_ypr = np.column_stack((yaws[i], pitchs[i], rolls[i]))
        restored_dataset[key].append(merged_ypr)
        i += 1

    for key in restored_dataset:
        restored_dataset[key] = np.array(restored_dataset[key])

    return restored_dataset


def separate_ypr_data(dataset):

    yaws, pitchs, rolls, labels = [], [], [], []

    for key in dataset:
        for sequence in dataset[key]:
            # Sequence shape is [100,3]
            yaws.append(sequence.T[0])
            pitchs.append(sequence.T[1])
            rolls.append(sequence.T[2])
            labels.append(key)

    return np.array(yaws), np.array(pitchs), np.array(rolls), np.array(labels)


def restore_ypr_sample(yaws, pitchs, rolls):

    restored_sequence = []

    for i in range(len(yaws)):
        restored_sequence.append(
            np.column_stack((yaws[i], pitchs[i], rolls[i])))

    return np.array(restored_sequence)


def separate_ypr_sample(data_sequence):
    yaws, pitchs, rolls, labels = [], [], [], []

    for sample in data_sequence:
        # sample shape is [100,3]
        yaws.append(sample.T[0])
        pitchs.append(sample.T[1])
        rolls.append(sample.T[2])

    return np.array(yaws), np.array(pitchs), np.array(rolls)


def __shallow_Autoencoder(input_size=100, code_size=64):
    '''
    one layer
    '''
    input_ypr = Input(shape=(input_size,))
    code = Dense(code_size, activation='relu')(input_ypr)
    output_ypr = Dense(input_size, activation='sigmoid')(code)

    return Model(input_ypr, output_ypr)


def __2_layer_Autoencoder(input_size=100, hidden_size=128, code_size=64):
    '''
    two layer 
    '''
    input_ypr = Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(input_ypr)
    code = Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_size, activation='relu')(code)
    output_ypr = Dense(input_size, activation='sigmoid')(hidden_2)

    return Model(input_ypr, output_ypr)


def Denoising_Autoencoder(ypr_train_noisy, input_size=100, hidden_size=128, code_size=64, verbose=False):

    normalized_train_noisy = normalize_ypr(ypr_train_noisy)

    if verbose:
        print("Creating Autoencoder model")

    # autoencoder = __shallow_Autoencoder(input_size, code_size)
    autoencoder = __2_layer_Autoencoder(input_size, hidden_size, code_size)

    autoencoder.compile(optimizer='adam', loss=AUTOENCODER_LOSS)
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    if verbose:
        print("Training Autoencoder...")
    autoencoder.fit(normalized_train_noisy, normalized_train_noisy, epochs=10)

    if verbose:
        print("Generate new ypr...")

    new_ypr = autoencoder.predict(normalized_train_noisy)

    new_ypr = restore_ypr(new_ypr)

    return new_ypr, autoencoder


def autoencode_by_letter(dataset):
    reconstructed_dataset = {}

    for key in dataset:
        yaws, pitchs, rolls = separate_ypr_sample(dataset[key])

        print(yaws.shape, pitchs.shape, rolls.shape)

        new_yaws = Denoising_Autoencoder(
            yaws, input_size=FEATURE_NUM, hidden_size=128, code_size=64, verbose=False)
        new_pitchs = Denoising_Autoencoder(
            pitchs, input_size=FEATURE_NUM, hidden_size=128, code_size=64, verbose=False)
        new_rolls = Denoising_Autoencoder(
            rolls, input_size=FEATURE_NUM, hidden_size=128, code_size=64, verbose=False)
        reconstructed_dataset[key] = restore_ypr_sample(
            new_yaws, new_pitchs, new_rolls)

    return reconstructed_dataset


def autoencode_as_whole(dataset):

    yaws, pitchs, rolls, labels = separate_ypr_data(dataset)

    print("yprl shapes:", yaws.shape, pitchs.shape, rolls.shape, labels.shape)

    new_yaws, yaws_encoder = Denoising_Autoencoder(
        yaws, input_size=FEATURE_NUM, hidden_size=HIDDEN_SIZE, code_size=CODE_SIZE, verbose=False)
    new_pitchs, pitchs_encoder = Denoising_Autoencoder(
        pitchs, input_size=FEATURE_NUM, hidden_size=HIDDEN_SIZE, code_size=CODE_SIZE, verbose=False)
    new_rolls, rolls_encoder = Denoising_Autoencoder(
        rolls, input_size=FEATURE_NUM, hidden_size=HIDDEN_SIZE, code_size=CODE_SIZE, verbose=False)

    print("after: yprl shapes:", new_yaws.shape,
          new_pitchs.shape, new_rolls.shape, labels.shape)

    reconstructed_dataset = restore_ypr_data(
        new_yaws, new_pitchs, new_rolls, labels)

    return reconstructed_dataset


def __as_denoise_test(dataset):
    yaws, pitchs, rolls, labels = separate_ypr_data(dataset)

    yaws_c = yaws.copy()
    pitchs_c = pitchs.copy()
    rolls_c = rolls.copy()

    _, _, _, ypr_encoder = ae_denoise(
        yaws, pitchs, rolls, FEATURE_NUM, HIDDEN_SIZE, CODE_SIZE)
    new_yaws, new_pitchs, new_rolls = ae_predict(
        yaws_c, pitchs_c, rolls_c, ypr_encoder)

    reconstructed_dataset = restore_ypr_data(
        new_yaws, new_pitchs, new_rolls, labels)

    return reconstructed_dataset


def ae_denoise(yaws, pitchs, rolls, feature_num=100, hidden_size=128, code_size=64):
    '''
    yaws: shape = (total_sequences, feature_num)
    pitch: shape = (total_sequences, feature_num)
    rolls: shape = (total_sequences, feature_num)

    total_sequences means the numebr of all data sequences in entire dataset, regardless of the labels.
    feature_num is the data samples per sequence, after resampling.

    return: denoised data with exactly same shape
    '''

    yaws = np.copy(yaws)
    pitchs = np.copy(pitchs)
    rolls = np.copy(rolls)

    new_yaws, yaws_encoder = Denoising_Autoencoder(
        yaws, input_size=feature_num, hidden_size=hidden_size, code_size=code_size, verbose=False)

    new_pitchs, pitchs_encoder = Denoising_Autoencoder(
        pitchs, input_size=feature_num, hidden_size=hidden_size, code_size=code_size, verbose=False)

    new_rolls, rolls_encoder = Denoising_Autoencoder(
        rolls, input_size=feature_num, hidden_size=hidden_size, code_size=code_size, verbose=False)

    ypr_encoder = np.array([yaws_encoder, pitchs_encoder, rolls_encoder])

    # return new_yaws, new_pitchs, new_rolls, ypr_encoder
    return new_yaws, new_pitchs, new_rolls, ypr_encoder


def ae_predict(yaws, pitchs, rolls, ypr_encoder):
    '''
    yaws: need to have the same feature number as when you train it
    ypr_encoder: the returned model
    '''

    yaws_encoder, pitchs_encoder, rolls_encoder = ypr_encoder[0], ypr_encoder[1], ypr_encoder[2]

    output = []

    for encoder, ypr_noisy in [(yaws_encoder, np.copy(yaws)), (pitchs_encoder, np.copy(pitchs)), (rolls_encoder, np.copy(rolls))]:
        normalized_noisy = normalize_ypr(ypr_noisy)
        res = encoder.predict(normalized_noisy)
        new_ypr = restore_ypr(res)
        output.append(new_ypr)

    new_yaws, new_pitchs, new_rolls = output[0], output[1], output[2]

    return new_yaws, new_pitchs, new_rolls


def main():
    '''
    In terminal, run {python autoencoder.py "flat_ypr_testdata"}
    '''
    trainx, devx, testx, trainy, devy, testy = data_loader.load_all_classic_random_split(
        True, False)
    dataset = restore_ypr_data(
        testx[:, :, 0], testx[:, :, 1], testx[:, :, 2], testy)

    _, _, _, ypr_encoder = ae_denoise(
        trainx[:, :, 0], trainx[:, :, 1], trainx[:, :, 2])
    new_yaws, new_pitchs, new_rolls = ae_predict(
        testx[:, :, 0], testx[:, :, 1], testx[:, :, 2], ypr_encoder)

    reconstructed_dataset = restore_ypr_data(
        new_yaws, new_pitchs, new_rolls, testy)

    # reconstructed_dataset = autoencode_by_letter(dataset)
    # reconstructed_dataset = autoencode_as_whole(dataset)
    # reconstructed_dataset = __as_denoise_test(dataset)

    color_red = '#980000'
    color_blue = '#003262'

    # if compare_dataset(restored_dataset, dataset):
    #     print('same!')

    print("shape of the reconstructed dataset:",
          reconstructed_dataset[0].shape)
    # print(dataset[0][0])
    # print(reconstructed_dataset[0][0])

    for key in reconstructed_dataset:

        plt.subplot(2, 2, 1)
        plt.plot(dataset[key][2].T[1], label='Original', color=color_blue)
        # plt.legend(loc="upper right")
        plt.ylabel('Degrees')
        plt.subplot(2, 2, 2)
        plt.plot(dataset[key][7].T[1], label='Original', color=color_blue)
        # plt.legend(loc="upper right")
        plt.subplot(2, 2, 3)
        plt.plot(dataset[key][12].T[1], label='Original', color=color_blue)
        # plt.legend(loc="upper right")
        plt.ylabel('Degrees')
        plt.subplot(2, 2, 4)
        plt.plot(dataset[key][18].T[1], label='Original', color=color_blue)
        # plt.legend(loc="upper right")

        plt.subplot(2, 2, 1)
        plt.plot(reconstructed_dataset[key][2].T[1],
                 label='Denoised', color=color_red)
        # plt.legend(loc="upper right")
        # plt.xlim(0, 100)
        # plt.ylim(-40, 60)
        # plt.gca().set_aspect('equal', adjustable='box')

        plt.subplot(2, 2, 2)
        plt.plot(reconstructed_dataset[key][7].T[1],
                 label='Denoised', color=color_red)
        # plt.legend(loc="upper right")
        # plt.xlim(0, 100)
        # plt.ylim(-40, 60)
        # plt.gca().set_aspect('equal', adjustable='box')

        plt.subplot(2, 2, 3)
        plt.plot(reconstructed_dataset[key][12].T[1],
                 label='Denoised', color=color_red)
        # plt.legend(loc="upper right")
        # plt.xlim(0, 100)
        # plt.ylim(-40, 60)
        # plt.gca().set_aspect('equal', adjustable='box')

        plt.subplot(2, 2, 4)
        plt.plot(reconstructed_dataset[key][18].T[1],
                 label='Denoised', color=color_red)
        # plt.legend(loc="upper right")

        letter_name = chr(key + 97)

        sub_title = 'Letter ' + letter_name + '. Pitch data'
        plt.suptitle(sub_title)

        plt.xlim(0, 100)
        plt.ylim(-40, 60)
        plt.gca().set_aspect('equal', adjustable='box')

        file_name = 'DAE/letter_' + letter_name + ".png"
        plt.savefig(file_name, dpi=200)
        plt.clf()


if __name__ == "__main__":
    main()
