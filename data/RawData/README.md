## The Structure of the Raw Data

|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| ID    |  TD  | yaw    | pitch   | raw    | ax     | ay      | az      | gx    | gy    | gz   | mx  | my  | mz  |
| ----- | :--: | ------ | ------- | ------ | ------ | ------- | ------- | ----- | ----- | ---- | ---- | ---- | ---- |
| **1** |  23  | 25.306 | -22.912 | -7.991 | 436.95 | -173.28 | 1092.65 | -5.17 | -10.1 | 1.24 | -243 | 386  | 93   |

\[index, time_diff, yaw, pitch, roll, 1000\*ax, 1000\*ay, 1000\*az, gx, gy, gz, mx, my, mz\]
 

* ID: the same ID index means that this belongs to the same handwritting motion.
* TD: Time Difference/Delta of this Frame
* ax : acceleration on x-axis
* ay : acceleration on y-axis
* az : acceleration on z-axis
* Yaw: rotation along z-axis (pen stick as axis, vertical)
* Pitch: rotation along y-axis (left-right axis)
* Roll: rotation along x-axis (front-back axis)
* mx: magnetometer x-axis
* my: magnetometer y-axis
* mz: magnetometer z-axis


More about Yaw Pitch Roll:
* Define output variables from updated quaternion---these are Tait-Bryan angles, commonly used in aircraft orientation.
* In this coordinate system, the positive z-axis is down toward Earth.
* **Yaw** is the angle between Sensor x-axis and Earth magnetic North (or true North if corrected for local declination, looking down on the sensor positive yaw is counterclockwise.
* **Pitch** is angle between sensor x-axis and Earth ground plane, toward the Earth is positive, up toward the sky is negative.
* **Roll** is angle between sensor y-axis and Earth ground plane, y-axis up is positive roll.
* These arise from the definition of the homogeneous rotation matrix constructed from quaternions.
* Tait-Bryan angles as well as Euler angles are non-commutative; that is, the get the correct orientation the rotations must be applied in the correct order which for this configuration is yaw, pitch, and then roll.
* For more see http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles which has additional links.
