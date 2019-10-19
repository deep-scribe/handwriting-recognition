import csv
import numpy as np
from scipy import interpolate

# tune params for pre-process data
RE_SAMPLE_RATE_COUNT = 100  # up-sample or down-sample rate for interpolation
SIX_DOF_REDUCTION = False  # false for 3 dof output, true for 6 dof output

# raw data file names
RAW_FILE_NAME_A = 'run_letter_a.csv'
RAW_FILE_NAME_B = 'run_letter_b.csv'
RAW_FILE_NAME_C = 'run_letter_c.csv'
SAVE_FILE_NAME_A = 'run_letter_a_format.csv'
SAVE_FILE_NAME_B = 'run_letter_b_format.csv'
SAVE_FILE_NAME_C = 'run_letter_c_format.csv'

# raw data index reference
# index 0: sample counter
# index 1: time interval between samples (in ms)
# index 2-4: yaw, pitch, row
# index 5-7: ax, ay, az
# index 9-11: gx, gy, gz
# index 12-14: mx, my, mz
REDUCTION_SAMPLE_INDEX_LIST = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
NON_REDUCTION_SAMPLE_INDEX_LIST = [0, 12, 13, 14]

SAMPLE_NON_TIME_INDEX_LIST = [1, 2, 3]  # index for yaw, pitch and roll
SAMPLE_NON_YAW_INDEX = [1, 2]  # index for pitch and roll
SAMPLE_NON_PITCH_INDEX = [0, 2]  # index for yaw and roll
SAMPLE_NON_ROLL_INDEX = [0, 1]  # index for yaw and pitch
SAMPLE_TIME_INDEX_LIST = [0]  # index for timestamp
SAMPLE_COUNT_INDICATOR_INDEX = 0  # index for sample counter


def read_raw_input(read_file_name):
    with open(read_file_name, 'r') as f:
        reader = csv.reader(f)
        raw_data_list = list(reader)
    return raw_data_list


def feature_reduction(sample, reduction_index_list):
    return np.delete(sample, reduction_index_list, axis=None)


def render_single_sample_with_reduction(sample, sample_rate_count):
    temp_format_list = []

    for index, value in enumerate(sample):
        current_sample = np.array(value).astype(None)

        if index > 0:
            current_sample[1] += float(sample[index - 1][1])
            sample[index] = current_sample

        temp_format_list.append(feature_reduction(current_sample, REDUCTION_SAMPLE_INDEX_LIST))

    format_list = np.array(temp_format_list)
    format_list_time_stamp = np.delete(format_list, SAMPLE_NON_TIME_INDEX_LIST, axis=1).flatten()
    format_list_data = np.delete(format_list, SAMPLE_TIME_INDEX_LIST, axis=1)
    format_list_yaw = np.delete(format_list_data, SAMPLE_NON_YAW_INDEX, axis=1).flatten()
    format_list_pitch = np.delete(format_list_data, SAMPLE_NON_PITCH_INDEX, axis=1).flatten()
    format_list_roll = np.delete(format_list_data, SAMPLE_NON_ROLL_INDEX, axis=1).flatten()

    timestamp_lower_bound = format_list_time_stamp[0]
    timestamp_upper_bound = format_list_time_stamp[-1]
    interpol_time_axis = np.linspace(timestamp_lower_bound, timestamp_upper_bound, num=sample_rate_count)

    interpol_yaw = ypr_interpolate(format_list_time_stamp, format_list_yaw, interpol_time_axis)
    interpol_pitch = ypr_interpolate(format_list_time_stamp, format_list_pitch, interpol_time_axis)
    interpol_roll = ypr_interpolate(format_list_time_stamp, format_list_roll, interpol_time_axis)

    interpol_merge_ypr = np.column_stack((interpol_yaw, interpol_pitch, interpol_roll))

    return interpol_merge_ypr.flatten()


def ypr_interpolate(sample_timestamp, sample_ypr, interpol_time_axis):
    interpol_function = interpolate.interp1d(sample_timestamp, sample_ypr)
    interpol_ypr = interpol_function(interpol_time_axis)
    return interpol_ypr


def render_output(raw_data_list, output_file_name):
    writer = csv.writer(open(output_file_name, 'w'))
    max_list_index = int(raw_data_list[-1][0])
    print('{}{}'.format('Reading file with sample count: ', max_list_index))

    # use 1 as starting index due to sample index start with 1
    for sample_index in range(1, max_list_index + 1):
        target_raw_sample = [row for row in raw_data_list if str(sample_index) in row[SAMPLE_COUNT_INDICATOR_INDEX]]
        render_single_sample_raw = np.array(target_raw_sample).astype(None)

        if SIX_DOF_REDUCTION:
            render_single_sample_result = render_single_sample_with_reduction(render_single_sample_raw,
                                                                              RE_SAMPLE_RATE_COUNT)
        else:
            render_single_sample_result = render_single_sample_with_reduction(render_single_sample_raw,
                                                                              RE_SAMPLE_RATE_COUNT)

        writer.writerow(render_single_sample_result)


def main():
    print('reading raw input on letter a')
    raw_list_a = read_raw_input(RAW_FILE_NAME_A)
    print('reading raw input on letter b')
    raw_list_b = read_raw_input(RAW_FILE_NAME_B)
    print('reading raw input on letter c')
    raw_list_c = read_raw_input(RAW_FILE_NAME_C)
    print('saving rendered output on letter a')
    render_output(raw_list_a, SAVE_FILE_NAME_A)
    print('saving rendered output on letter b')
    render_output(raw_list_b, SAVE_FILE_NAME_B)
    print('saving rendered output on letter c')
    render_output(raw_list_c, SAVE_FILE_NAME_C)


if __name__ == "__main__":
    main()
