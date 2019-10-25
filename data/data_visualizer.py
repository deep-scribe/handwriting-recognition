from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import string

import data_utils


def get_rotation_matrix(yaw, pitch, roll):
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
    yaw, pitch, roll = yaw_pitch_roll_vec.tolist()
    rotmat = get_rotation_matrix(yaw, pitch, roll)
    return np.matmul(rotmat, vector_to_rotate)


for ch in string.ascii_lowercase:
    # def plot_all_samples(sample_path, outpath):
    df = data_utils.load_all_subjects('raw_data')
    sampledfs = data_utils.get_all_samples_by_label(df, ch)

    for sampleid in sampledfs:
        sampledf = sampledfs[sampleid]

        # think this should be velocity
        accelerations = sampledf[['ax', 'ay', 'az']].to_numpy()
        yprs = sampledf[['yaw', 'pitch', 'roll']].to_numpy()

        frame_durations = sampledf[['td']].to_numpy()

        numkeypoint = accelerations.shape[0]+1

        # think this should be integral of position
        positions = np.zeros((numkeypoint, accelerations.shape[1]))
        # think this should be position
        velocities = np.zeros((numkeypoint, accelerations.shape[1]))

        for i in range(accelerations.shape[0]):
            accelerations[i] = rotate_to_world_axes(accelerations[i], yprs[i])

        for i in range(1, numkeypoint-1):
            acceleration = accelerations[i]
            acceleration[2] -= 1000

            old_velocity = velocities[i]
            old_position = positions[i]
            delta_t = frame_durations[i]

            new_velocity = old_velocity + delta_t * \
                (acceleration + (acceleration - accelerations[i-1] / 2))
            velocities[i+1] = new_velocity

            new_position = old_position + delta_t * \
                (old_velocity + (old_velocity - velocities[i-1]) / 2)
            positions[i+1] = new_position

        # print(velocities)

        plt.clf()
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            velocities[:, 0],
            velocities[:, 1],
            velocities[:, 2],
            c=[i for i in range(velocities.shape[0])],
            cmap='hot'
        )
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.savefig(f'3d_{ch}_{sampleid}.png')

        plt.clf()
        fig = plt.figure()
        plt.axes().scatter(
            velocities[:, 0],
            velocities[:, 1],
            c=[i for i in range(velocities.shape[0])],
            cmap='hot'
        )
        plt.savefig(f'2d_{ch}_{sampleid}.png')
