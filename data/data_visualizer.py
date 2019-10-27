from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import string
import shutil
import os
import sys
import data_utils


def get_rotation_matrix(yaw, pitch, roll):
    '''
    return np.array: a rotation matrix given yaw pitch roll
    '''
    sinalpha = np.sin(np.radians(yaw))
    cosalpha = np.cos(np.radians(yaw))
    sinbeta = np.sin(np.radians(pitch))
    cosbeta = np.cos(np.radians(pitch))
    singamma = np.sin(np.radians(roll))
    cosgamma = np.cos(np.radians(roll))
    rotationmatrix = np.array([
        [
            cosalpha*cosbeta,
            cosalpha*sinbeta*singamma - sinalpha * singamma,
            cosalpha*sinbeta*cosgamma + sinalpha*singamma
        ],
        [
            sinalpha*cosbeta,
            sinalpha*sinbeta*singamma + cosalpha*cosgamma,
            sinalpha*sinbeta*cosgamma - cosalpha*singamma,
        ],
        [
            -sinbeta,
            cosbeta*singamma,
            cosbeta*cosgamma
        ]
    ])
    return rotationmatrix


def rotate_to_world_axes(vector_to_rotate, yaw_pitch_roll_vec):
    '''
    rotate a vector by a 3d vector denoting yaw pitch roll
    '''
    yaw, pitch, roll = yaw_pitch_roll_vec.tolist()
    rotmat = get_rotation_matrix(yaw, pitch, roll)
    return np.matmul(rotmat, vector_to_rotate)


def create_dir_remove_old(path):
    '''
    created directory denoted by string path, force remove old one if exist
    '''
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


PNG_DIRNAME_YPRS_2D = 'yprs_2d_pngs'
PNG_DIRNAME_YPRS_3D = 'yprs_3d_pngs'
YPRS_COLUMNS = ['yaw', 'pitch', 'roll', ]


def dump_yprs_pngs(subject_path):
    '''
    subject_path: string of a path containing all csv recorded by a subject
    this path should also contain calibration.csv

    visualize yaw pitch roll by drawing scatter plot of the end of a unit vector
    rotated by the delta yaw pitch roll with respect to each starting postiion of the
    sample sequence

    dump 2D and 3D pngs to corresponding subdir inside subject_path
    '''
    create_dir_remove_old(os.path.join(subject_path, PNG_DIRNAME_YPRS_2D))
    create_dir_remove_old(os.path.join(subject_path, PNG_DIRNAME_YPRS_3D))

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
                sample_yprs = get_calibrated_delta(
                    calibrationyprs, sample_yprs
                )

                # TODO: this looks like a runtime bottleneck
                # make this faster
                trace = []
                for i in range(sample_yprs.shape[0]):
                    trace.append(rotate_to_world_axes(
                        get_unit_vector(), sample_yprs[i]
                    ))
                trace = np.array(trace)

                output_2d_scatter(
                    trace[:, 0],
                    trace[:, 1],
                    os.path.join(
                        subject_path,
                        PNG_DIRNAME_YPRS_2D,
                        f'{label_name}_{sample_id}.png'
                    )
                )

                output_3d_scatter(
                    trace[:, 0],
                    trace[:, 1],
                    trace[:, 2],
                    os.path.join(
                        subject_path,
                        PNG_DIRNAME_YPRS_3D,
                        f'{label_name}_{sample_id}.png'
                    )
                )


def output_2d_scatter(xs, ys, path):
    '''
    draw a scatter plot given datapoints and output to path
    '''
    plt.clf()
    fig = plt.figure()
    plt.axes().scatter(
        xs,
        ys,
        c=[i for i in range(xs.shape[0])],
        cmap='hot'
    )
    fig.savefig(path)
    plt.close(fig)


def output_3d_scatter(xs, ys, zs, path):
    '''
    draw a scatter plot given datapoints and output to path
    '''
    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(
        xs,
        ys,
        zs,
        c=[i for i in range(xs.shape[0])],
        cmap='hot'
    )
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    fig.savefig(path)
    plt.close(fig)


def get_unit_vector():
    return np.array([0, 0, 1])


def get_calibrated_delta(initial_calibration_vec, sample_frames):
    '''
    calibrate sample frames by subtracting calibration delta, then subtract
    0th frame to get delta w.r.t to begin of sequence
    '''
    return (sample_frames - initial_calibration_vec) - sample_frames[0]


def main():
    if len(sys.argv) != 2:
        print('Usage: python data_visualizer.py <subject_path>')
        quit()

    subject_path = sys.argv[1]
    dump_yprs_pngs(subject_path)


if __name__ == "__main__":
    main()
