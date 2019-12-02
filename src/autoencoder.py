import os
import sys
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

import data_flatten

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam


FEATURE_NUM = 300
# if you want to use logistic regression
AUTOENCODER_LOSS ='binary_crossentropy'
# if you want to use linear regression   
# AUTOENCODER_LOSS ='mean_squared_error'

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
    loaded_dataset = data_flatten.load_data_dict_from_file(subject_path, calibrate=True, verbose=False)

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

    yaws, pitchs, rolls, labels = [],[],[],[]
    
    for key in dataset:
        for sample in dataset[key]:
            # sample shape is [100,3]
            yaws.append(sample.T[0])
            pitchs.append(sample.T[1])
            rolls.append(sample.T[2])
            labels.append(key)

    return np.array(yaws), np.array(pitchs), np.array(rolls), np.array(labels)

def restore_ypr_sample(yaws,pitchs,rolls):

    restored_sequence = []

    for i in range(len(yaws)):
        restored_sequence.append(
            np.column_stack((yaws[i], pitchs[i], rolls[i])))

    return np.array(restored_sequence)

def separate_ypr_sample(data_sequence):
    yaws, pitchs, rolls, labels = [],[],[],[]

    for sample in data_sequence:
        # sample shape is [100,3]
        yaws.append(sample.T[0])
        pitchs.append(sample.T[1])
        rolls.append(sample.T[2])

    return np.array(yaws), np.array(pitchs), np.array(rolls)

def __shallow_Autoencoder(input_size = 300, code_size = 64):
    '''
    one layer
    '''
    input_ypr = Input(shape=(input_size,))
    code = Dense(code_size, activation='relu')(input_ypr)
    output_ypr = Dense(input_size, activation='sigmoid')(code)

    return Model(input_ypr, output_ypr)

def __2_layer_Autoencoder(input_size = 300, hidden_size = 128, code_size = 64):
    '''
    two layer 
    '''
    input_ypr = Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(input_ypr)
    code = Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_size, activation='relu')(code)
    output_ypr = Dense(input_size, activation='sigmoid')(hidden_2)

    return Model(input_ypr, output_ypr)
    

def Denoising_Autoencoder(ypr_train_noisy, input_size = 300, hidden_size = 128, code_size = 64, verbose=False):

    normalized_train_noisy = normalize_ypr(ypr_train_noisy)

    input_ypr = Input(shape=(input_size,))

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

    # plt.subplot(1,2,1)
    # plt.plot(ypr_train_noisy[0])
    # plt.ylabel('Degrees')
    # plt.subplot(1,2,2)
    # plt.plot(new_ypr[0])
    # plt.suptitle('Compare before and after denoise, y/p/r data')
    # plt.show()

    return new_ypr

def autoencode_by_letter(dataset):
    reconstructed_dataset = {}

    for key in dataset:
        yaws, pitchs, rolls = separate_ypr_sample(dataset[key])

        print(yaws.shape, pitchs.shape, rolls.shape)

        new_yaws = Denoising_Autoencoder(yaws, input_size=FEATURE_NUM, hidden_size = 128, code_size = 64, verbose=False)
        new_pitchs = Denoising_Autoencoder(pitchs, input_size=FEATURE_NUM, hidden_size = 128, code_size = 64, verbose=False)
        new_rolls = Denoising_Autoencoder(rolls, input_size=FEATURE_NUM, hidden_size = 128, code_size = 64, verbose=False)
        reconstructed_dataset[key] = restore_ypr_sample(new_yaws, new_pitchs, new_rolls)

    return reconstructed_dataset

def autoencode_as_whole(dataset):

    yaws, pitchs, rolls, labels = separate_ypr_data(dataset)

    print(yaws.shape, pitchs.shape, rolls.shape, labels.shape)

    new_yaws = Denoising_Autoencoder(yaws, input_size=FEATURE_NUM, hidden_size = 128, code_size = 64, verbose=False)
    new_pitchs = Denoising_Autoencoder(pitchs, input_size=FEATURE_NUM, hidden_size = 128, code_size = 64, verbose=False)
    new_rolls = Denoising_Autoencoder(rolls, input_size=FEATURE_NUM, hidden_size = 128, code_size = 64, verbose=False)

    reconstructed_dataset = restore_ypr_data(new_yaws, new_pitchs, new_rolls, labels)

    return reconstructed_dataset

def main():
    '''
    In terminal, run {python autoencoder.py "flat_ypr_testdata"}
    '''
    if len(sys.argv) != 2:
        print('Usage: python autoencoder.py <subject_path>')
        quit()

    subject_path = sys.argv[1]
    dataset = load_data(subject_path, feature_num=FEATURE_NUM)

    # reconstructed_dataset = autoencode_by_letter(dataset)
    reconstructed_dataset = autoencode_as_whole(dataset)

    # if compare_dataset(restored_dataset, dataset):
    #     print('same!')

    print("shape of the reconstructed dataset:",reconstructed_dataset['a'].shape)

    for key in reconstructed_dataset:
        plt.subplot(2,2,1)
        plt.plot(dataset[key][2].T[1])
        plt.ylabel('Degrees')
        plt.subplot(2,2,2)
        plt.plot(dataset[key][7].T[1])
        plt.subplot(2,2,3)
        plt.plot(dataset[key][12].T[1])
        plt.ylabel('Degrees')
        plt.subplot(2,2,4)
        plt.plot(dataset[key][18].T[1])

        plt.subplot(2,2,1)
        plt.plot(reconstructed_dataset[key][2].T[1])
        plt.subplot(2,2,2)
        plt.plot(reconstructed_dataset[key][7].T[1])
        plt.subplot(2,2,3)
        plt.plot(reconstructed_dataset[key][12].T[1])
        plt.subplot(2,2,4)
        plt.plot(reconstructed_dataset[key][18].T[1])
        sub_title = 'Letter ' + key + '. Pitch data'
        plt.suptitle(sub_title)

        file_name = 'DAE_output/letter_' + key + ".png"
        plt.savefig(file_name)
        plt.clf()



if __name__ == "__main__":
    main()

