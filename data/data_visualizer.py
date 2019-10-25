from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

import data_utils


df = data_utils.load_all_subjects('raw_data')
sampledf = data_utils.get_random_sample_by_label(df, 'o')

# think this should be velocity
accelerations = sampledf[['ax', 'ay', 'az']].to_numpy()
yprs = sampledf[['yaw', 'pitch', 'roll']].to_numpy()

frame_durations = sampledf[['td']].to_numpy()

numkeypoint = accelerations.shape[0]+1

# think this should be integral of position
positions = np.zeros((numkeypoint, accelerations.shape[1]))
# think this should be position
velocities = np.zeros((numkeypoint, accelerations.shape[1]))

for i in range(1, numkeypoint-1):
    acceleration = accelerations[i]
    acceleration -= accelerations[0]
    yaw, pitch, roll = (yprs[0]-yprs[i]).tolist()

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

    acceleration = np.matmul(rotationmatrix, acceleration, )

    old_velocity = velocities[i]
    old_position = positions[i]
    delta_t = frame_durations[i]

    new_velocity = old_velocity + delta_t * \
        (acceleration + (acceleration - accelerations[i-1] / 2))
    velocities[i+1] = new_velocity

    new_position = old_position + delta_t * \
        (new_velocity + (new_velocity - old_velocity) / 2)
    positions[i+1] = new_position


# print(velocities)

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
plt.show()

fig = plt.figure()
plt.scatter(
    velocities[:, 0],
    velocities[:, 1],
    c=[i for i in range(velocities.shape[0])],
    cmap='hot'
)
plt.show()
