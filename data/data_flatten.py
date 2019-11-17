import os
import numpy as np
import sys
import data_utils
from scipy import interpolate


INDEX_TIME_COLUMNS = [
    'id', 'td',
]

YPRS_COLUMNS = [
    'yaw', 'pitch', 'roll',
]

##
# Private Functions
def __linear_interpolate_1d(sample_timestamp, sample_ypr, interpol_time_axis):
    interpol_function = interpolate.interp1d(sample_timestamp, sample_ypr)
    interpol_ypr = interpol_function(interpol_time_axis)
    return interpol_ypr

def __get_calibrated_delta(initial_calibration_vec, sample_frames):
    '''
    calibrate sample frames by subtracting calibration delta, then subtract
    0th frame to get delta w.r.t to begin of sequence
    '''
    return (sample_frames - initial_calibration_vec) - sample_frames[0]

def __create_time_stamps(time_list):
    '''
    time_list: np.array with shape=(n_samples,)

    return np.array with same shape but elems are accumulated time stamps
    '''
    output = np.zeros(shape=(time_list.shape[0],))
    acc_sum = 0

    for i, time_delta in enumerate(time_list):
        output[i] = acc_sum + time_delta
        acc_sum += time_delta

    return output

##
# Public Functions
def load_data_dict_from_file(subject_path, calibrate=True):
    '''
    subject_path: string of a path containing all csv recorded by a subject,
    it is required that the file is named as {label_name}.csv
    this path should also contain calibration.csv

    calibrate: most likely you want to deal with the calibrated data
    
    return dictionary with len()=26, key is label, value is
        1D np.array with shape=(num_writing_events, ), and
        each element is 2D np.array with shape=(num_data_samples_of_event, 5)
        where 5 is for: [index,time_delta,y,p,r]

    NOTE: num_data_samples_of_event differs for different events
    '''

    df = data_utils.load_subject(subject_path)

    calibrationdf = df[df['label'] == data_utils.CALIBRATION_LABEL_NAME]
    if calibrationdf.empty:
        print('!' * 80)
        print(
            f'WARN: no {data_utils.CALIBRATION_FILENAME}, using default [0,0,0] to calibrate yprs.'
        )
        print('!' * 80)
        calibrationyprs = np.zeros(3)
    else:
        calibrationyprs = calibrationdf[YPRS_COLUMNS].to_numpy()
        calibrationyprs = np.mean(calibrationyprs, axis=0)

    # only walk the current layer in directory
    depth, walk_depth = 0, 1

    for root, dirs, files in os.walk(subject_path):

        if depth >= walk_depth:
            break

        depth += 1
        dataset_dict = {}

        for filename in files:
            if '.csv' not in filename:
                continue
            if filename == data_utils.CALIBRATION_FILENAME:
                continue

            label_name = filename.replace('.csv', '')
            samplesdf = df[df['label'] == label_name]
            sampleids = samplesdf.id.unique()

            total_lines = samplesdf.shape[0]
            num_samples = sampleids.shape[0]

            print(f'Processing label {label_name}, with {num_samples} samples, '
                  f'from {total_lines} lines...')

            data_sequences = []

            for i, sample_id in enumerate(sampleids):
                # each writing event:
                sampledf = samplesdf[samplesdf['id'] == sample_id]
                sample_yprs = sampledf[YPRS_COLUMNS].to_numpy()

                if calibrate:
                    sample_yprs = __get_calibrated_delta(calibrationyprs, sample_yprs)

                sample_idx_td = sampledf[INDEX_TIME_COLUMNS].to_numpy()
                sample_idx_td_yprs = np.concatenate((sample_idx_td, sample_yprs), axis=1)

                # print("yprs",sample_yprs.shape)
                # print("idx_td_yprs",sample_idx_td_yprs.shape)

                data_sequences.append(sample_idx_td_yprs)

            # e.g.: label a has 20 data_sequences,
            # each sequence has various # of data samples,
            # each data sample has 5 columns [index, time, yaw, pitch, roll]
            dataset_dict[label_name] = np.asarray(data_sequences)

    print("Successfully loaded yaw pitch roll data set in dictionary format from",len(dataset_dict),"files.\n")

    return dataset_dict

def resample_sequence(data_sequence, is_flatten_ypr=True, feature_num=100):
    '''
    data_sequence: np.array with shape=(n_samples, 5),
    where 5 is for [index, time_delta, y, p, r]

    return np.array with shape=(3*feature_num,)
    '''

    yaw_list = data_sequence.T[2]
    pitch_list = data_sequence.T[3]
    roll_list = data_sequence.T[4]

    time_stamps = __create_time_stamps(data_sequence.T[1])
    time_lower_bound = time_stamps[0]
    time_upper_bound = time_stamps[-1]
    time_axis = np.linspace(time_lower_bound, time_upper_bound, num=feature_num)

    interpol_yaw = __linear_interpolate_1d(time_stamps, yaw_list, time_axis)
    interpol_pitch = __linear_interpolate_1d(time_stamps, pitch_list, time_axis)
    interpol_roll = __linear_interpolate_1d(time_stamps, roll_list, time_axis)

    merged_ypr = np.column_stack((interpol_yaw,interpol_pitch,interpol_roll))

    if not is_flatten_ypr:
        return merged_ypr

    return merged_ypr.flatten()


def resample_dataset(data, is_flatten_ypr=True, feature_num=100):
    '''
    data: data dictionary as returned by load_data_dict_from_file

    return: data dictionary with flattened data, shape=(m, 3*feature_num)
    '''
    resampled_output = {}

    for label_name, data_sequences in data.items():
        new_sequences = []

        for data_seq in data_sequences:
            new_sequences.append(
                resample_sequence(data_seq, is_flatten_ypr, feature_num)
            )

        resampled_output[label_name] = np.asarray(new_sequences)

    return resampled_output


def example():
    '''
    In terminal, run {python data_flatten.py "flat_ypr_testdata"}
    '''
    if len(sys.argv) != 2:
        print('Usage: python data_flatten.py <subject_path>')
        quit()

    subject_path = sys.argv[1]

# Example 1:
# If you want to resample and flatten the data
# For data for each label_name, you get shape=(20,300)
    loaded_dataset = load_data_dict_from_file(subject_path, calibrate=True)
    flattened_dataset = resample_dataset(loaded_dataset, is_flatten_ypr=True, feature_num=100)
    print ("Sanity check for Example 1, ypr data is flattened...")

    # the shape of should be (20, 3 * 100)
    assert(flattened_dataset['a'].shape == (20, 300))
    print (flattened_dataset['a'].shape)

    # peek one data sample from letter m
    assert(flattened_dataset['m'][10].shape == (300,))
    print(flattened_dataset['m'][10])

    # peek first three ypr from letter z
    print(flattened_dataset['z'][19][:3])

# Example 2:
# If you only want to resample but don't flatten the data
# For data for each label_name, you get shape=(20,100,3)
    resampled_dataset = resample_dataset(loaded_dataset, is_flatten_ypr=False, feature_num=100)
    print("\nSanity check for Example 2, ypr data is NOT flattened...")

    # the shape of should be (20, 100, 3)
    assert(resampled_dataset['a'].shape == (20, 100, 3))
    print (resampled_dataset['a'].shape)

    # peek one data sequence from letter m, should be ypr in 100 rows
    assert(resampled_dataset['m'][10].shape == (100,3))
    print(resampled_dataset['m'][10])

    # peek first three ypr from letter z
    print(resampled_dataset['z'][19][0])

if __name__ == "__main__":
    example()