import numpy as np
import math
import data_utils
import data_visualizer
import string
import shutil
import os
import sys


'''
data augmentation pipeline to construct more training samples by applying
- noise to each frame
- stretch to each axis to all frames in axis
- rotation to all frames in sequence
'''


def quaternion_to_rotation_matrix(q):
    a, b, c, d = q
    return np.array([[a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                     [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                     [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])


def add_noise(sample_yprs, noise_mean=0.0, noise_std=1.0):
    '''
    With an input matrix, add noise~N(noise_mean,noise_std^2) to each entry

    Parameter:
        sample_yprs: input yall, pitch, roll matrix with dimension (N,3)
        noise_mean: the mean of the noise normal distribution, defaul 0
        noise_std: the standard deviation of the noise normal distribution, default 1

    Return:
        the new matrix with noise, dimension (N,3)
    '''
    for i in range(sample_yprs.shape[0]):
        eps = np.random.randn(3)*noise_std + noise_mean
        sample_yprs[i] += eps
    return sample_yprs


def rotate_by_vector(sample_yprs, theta, rotation_axis=[0, 0, 1]):
    '''
    With an input matrix, rotate the object counter-clockwisely around the same rotation axis by the same angle at each time stamp

    Parameter:
        sample_yprs: input yall, pitch, roll matrix with dimension (N,3)
        rotation_axis: the axis around which the input should rotate.
                        It is an unit vector (x, y ,z)
        theta: the angle of rotation in degree

    Return:
        the new matrix after rotation, dimension (N,3)
    '''

    rotation_axis = np.array(rotation_axis)
    q = np.hstack([np.array(math.cos(np.radians(theta/2.0))),
                   rotation_axis * math.sin(np.radians(theta/2.0))])
    # print(np.matmul(quaternion_to_rotation_matrix(q), sample_yprs.T).T.shape)
    return np.matmul(quaternion_to_rotation_matrix(q), sample_yprs.T).T


def stretch(sample_yprs, ky, kp, kr):
    '''
    With an input matrix, stretch the object with constants ky, kp, kr

    Parameter:
        sample_yprs: input yall, pitch, roll matrix with dimension (N,3)
        k*: stretching constants

    Return:
        the new matrix after stretching, dimension (N,3)
    '''
    return sample_yprs*np.array([ky, kp, kr])


PNG_DIRNAME_YPRS_2D = 'yprs_2d_pngs_augmented'
PNG_DIRNAME_YPRS_3D = 'yprs_3d_pngs_augmented'
YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]


def augment(sample_yprs, rotate=True, noise=True, stretching=True, theta_range=5):
    '''
    sample_ypr shape=(N,3)
    '''
    if rotate:
        theta = np.random.randn()*theta_range
        sample_yprs = rotate_by_vector(sample_yprs, theta)
    if stretching:
        ky, kp, kr = np.random.randn(3)*0.3 + 1
        sample_yprs = stretch(sample_yprs, ky, kp, kr)
    if noise:
        sample_yprs = add_noise(sample_yprs)
    # print(len(traces.keys()))
    return sample_yprs


def dump_augmented_yprs_pngs(subject_path):
    '''
    subject_path: string of a path containing all csv recorded by a subject
    this path should also contain calibration.csv

    visualize yaw pitch roll by drawing scatter plot of the end of a unit vector
    rotated by the delta yaw pitch roll with respect to each starting postiion of the
    sample sequence

    dump 2D and 3D pngs to corresponding subdir inside subject_path
    '''
    data_visualizer.create_dir_remove_old(
        os.path.join(subject_path, PNG_DIRNAME_YPRS_2D))
    data_visualizer.create_dir_remove_old(
        os.path.join(subject_path, PNG_DIRNAME_YPRS_3D))
    print(PNG_DIRNAME_YPRS_2D)
    print(PNG_DIRNAME_YPRS_3D)

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

    for root, dirs, files in os.walk(subject_path):
        for filename in files:
            if '.csv' not in filename:
                continue
            if filename == data_utils.CALIBRATION_FILENAME:
                continue

            label_name = filename.replace('.csv', '')
            samplesdf = df[df['label'] == label_name]
            sampleids = samplesdf.id.unique()

            print(f'Processing label {label_name}...')

            for sample_id in sampleids:
                sampledf = samplesdf[samplesdf['id'] == sample_id]
                sample_yprs = sampledf[YPRS_COLUMNS].to_numpy()
                sample_yprs = data_visualizer.get_calibrated_delta(
                    calibrationyprs, sample_yprs
                )

                sample_yprs = augment(sample_yprs)

                # TODO: this looks like a runtime bottleneck
                # make this faster
                trace = []
                for i in range(sample_yprs.shape[0]):
                    trace.append(data_visualizer.rotate_to_world_axes(
                        data_visualizer.get_unit_vector(), sample_yprs[i]
                    ))
                trace = np.array(trace)

                data_visualizer.output_2d_scatter(
                    trace[:, 0],
                    trace[:, 1],
                    os.path.join(
                        subject_path,
                        PNG_DIRNAME_YPRS_2D,
                        f'{label_name}_{sample_id}_aug.png'
                    )
                )

                data_visualizer.output_3d_scatter(
                    trace[:, 0],
                    trace[:, 1],
                    trace[:, 2],
                    os.path.join(
                        subject_path,
                        PNG_DIRNAME_YPRS_3D,
                        f'{label_name}_{sample_id}_aug.png'
                    )
                )


def main():
    print('Loading one subject')
    df = data_utils.load_subject('../data/kevin_11_7')

    print('one sample sequence chosen in the pandas dataframe')
    row = data_utils.get_random_sample_by_label(df, 'a')
    print(row)

    print('yprs of one sample sequence')
    yprs = row[YPRS_COLUMNS].to_numpy()
    print(yprs)

    print('augmented yprs of one sample sequence')
    augmented = augment(yprs)
    print(augmented)

    print('sum of squared distance from original to yprs')
    sum_sq_dist = np.sum((yprs - augmented) ** 2)
    print(sum_sq_dist)


if __name__ == "__main__":
    main()
