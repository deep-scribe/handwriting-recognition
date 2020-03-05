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


def noise_stretch_rotate_augment(
    train_x, train_y, augment_prop=1, is_already_flattened=True, resampled=True
):
    '''
    use default data augmentation setting to append to the TRAIN_SET
    augment_prop * len(train_set) number of samples
    please augment TRAIN_SET only
    return the augmented x and ys
    '''
    augmented_xs = []
    augmented_ys = []

    for p in range(augment_prop):
        for i in range(train_x.shape[0]):
            x = train_x[i]
            y = train_y[i]

            if is_already_flattened:
                unflattened_x = x.reshape(int(x.shape[0] / 3), 3)
            else:
                unflattened_x = x
            augmented_x = augment(unflattened_x)

            if is_already_flattened:
                augmented_xs.append(augmented_x.flatten())
            else:
                augmented_xs.append(augmented_x)
            augmented_ys.append(y)
    if resampled:
        return np.vstack((train_x, np.array(augmented_xs))), np.append(train_y, np.array(augmented_ys))
    else:
        return np.append(train_x, augmented_xs), np.append(train_y, augmented_ys)


def get_concat_augment(xs, ys, augment_prop, frame_prop=0.10):
    '''
    x shape=(N,3)
    do this before shape augment
    '''
    assert augment_prop >= 1

    aug_xs = []
    aug_ys = []
    num_orig = len(xs)

    for _ in range(augment_prop):
        for i, x in enumerate(xs):
            y = ys[i]
            aug_ys.append(y)

            nframes, _ = x.shape
            x_front_src = xs[np.random.choice(num_orig)]
            x_back_src = xs[np.random.choice(num_orig)]

            front_noise_frame_prop = np.random.random() * frame_prop
            front_noise_frame_num = int(
                front_noise_frame_prop * len(x_front_src))
            back_noise_frame_prop = np.random.random() * frame_prop
            back_noise_frame_num = int(
                back_noise_frame_prop * len(x_back_src))

            front_noise = x_front_src[:front_noise_frame_num, :]
            back_noise = x_back_src[len(x_back_src) - back_noise_frame_num:, :]

            augmented_x = np.vstack([front_noise, x, back_noise])
            aug_xs.append(augmented_x)

    return np.array(aug_xs), np.array(aug_ys)


def get_trim_augment(xs, ys, augment_prop, frame_prop=0.1):
    '''
    x shape=(N,3)
    do this before shape augment
    '''

    aug_xs = []
    aug_ys = []

    for _ in range(augment_prop):
        for i, x in enumerate(xs):
            nframes = len(x)
            y = ys[i]
            aug_ys.append(y)

            front_trim_frame_num = int(
                np.random.random() * frame_prop * nframes)
            back_trim_frame_num = int(
                np.random.random() * frame_prop * nframes)

            augmented_x = x[
                front_trim_frame_num:
                nframes - back_trim_frame_num, :]
            aug_xs.append(augmented_x)

    return np.array(aug_xs), np.array(aug_ys)


def get_nonclass_samples(xs, count, nonclass_idx=26):
    lenxs = len(xs)
    nonclass_xs = []

    # nonclass by taking small partial of a sample
    for _ in range(count // 2):
        x = xs[np.random.choice(lenxs)]
        lenx = len(x)
        num_frame = np.random.choice(lenx//4) + 2
        begin_frame = np.random.choice(lenx - num_frame - 1)
        partx = x[begin_frame: begin_frame + num_frame, :]
        nonclass_xs.append(partx)

    # nonclass by concating 2-4 chars
    for _ in range(count // 2):
        cat = []
        n = np.random.randint(2, 5)
        for _ in range(n):
            cat.append(xs[np.random.choice(lenxs)])
        nonclass_x = np.vstack(cat)
        nonclass_xs.append(nonclass_x)

    return np.array(nonclass_xs), np.array([nonclass_idx for _ in range(len(nonclass_xs))])


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
